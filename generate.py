import argparse
from datetime import datetime
import os
import librosa
import numpy as np
import tensorflow as tf
from scipy.interpolate import interpolate

from wavenet import WaveNet

SAMPLES = 16000
SAMPLE_RATE = 16000
DILATIONS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
             1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
             1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
             1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
             1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
RESIDUAL_CHANNELS = 32
DILATION_CHANNELS = 32
QUANTIZATION_CHANNELS = 256
SKIP_CHANNELS = 512


def get_arguments():
    parser = argparse.ArgumentParser(description='WaveNet generation script')

    parser.add_argument('--hop_length', type=int, default=128,
                        help='Size of hop in samples')
    parser.add_argument('--n_mfcc', type=int, default=25,
                        help='Number of mel cepstrum coefficients to return')
    parser.add_argument('--n_fft', type=int, default=512,
                        help='The size of the fft bin used to calculate mfcc')
    parser.add_argument('--use_aux_features', type=bool, default=False,
                        help='whether to use auxiliary features as input')
    parser.add_argument('--aux_source', type=str, default=None,
                        help='A file from which to generate aux_features')
    parser.add_argument('--mean_sub', type=bool, default=False,
                        help='whether to subtract the mean from mfccs')
    parser.add_argument('--normalise', type=bool, default=False,
                        help='whether to normalise mfccs to variance 1')
    parser.add_argument('--mean_file', type=str, default=None)
    parser.add_argument('--std_file', type=str, default=None)
    parser.add_argument('--checkpoint', type=str,
                        help='Which model checkpoint to generate from')
    parser.add_argument('--samples',type=int, default=SAMPLES,
                        help='How many waveform samples to generate')
    parser.add_argument('--wav_out_path', type=str, default=None,
                        help='Path to output wav file')

    return parser.parse_args()


def calculate_mel_cepstrum(audio, sample_rate, hop_length, n_mfcc, n_fft,
                            mean_sub, normalise, mean_file, std_file):
    # add extra band as one will be removed later
    n_mfcc = n_mfcc + 1
    # 1) fft
    fft = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length, center=False)
    # 2) pendogram
    pend = np.abs(fft)**2
    # 3) mel spaced filterbanks
    melspec = librosa.feature.melspectrogram(S=pend, n_mels=n_mfcc, n_fft=n_fft)
    # 4) log of mel spaced filter filterbanks
    melspec_log = librosa.power_to_db(melspec)
    # 5) DCT to get mel_cepstrum coefficients
    melcep = librosa.feature.mfcc(S=melspec_log, n_mfcc=n_mfcc)
    # drop first band
    melcep = melcep[1:]
    # append zeros at either end
    melcep = np.pad(melcep,((0,0),(1,1)),mode='constant')
    print("mean_sub: ", mean_sub)
    if mean_sub:
        mean = np.load(mean_file)
        melcep = melcep - mean
    if normalise:
        std = np.load(std_file)
        for i in range(std.shape[0]):
            if std[i] > 0:
                melcep[i] = melcep[i] / std[i]
    # interpolate
    # get x values: the centers of each fft frame, plus first and last samples
    num_x = melcep.shape[1]
    x = np.zeros(num_x)
    for i in range(melcep.shape[1] - 2):
        x[i + 1] = (n_fft/2) + (i * hop_length)
    x[-1] = audio.shape[0]
    # create an array to hold interploated values
    melcep_interp = np.zeros([audio.shape[0],melcep.shape[0]])
    # list of all sample values to produce interpolated y
    s = np.arange(audio.shape[0])
    for i in range(melcep.shape[0]):
        f = interpolate.interp1d(x,melcep[i])
        melcep_interp[:,i] = f(s)
    # trim
    audio_length = audio.shape[0]
    melcep_interp = melcep_interp[:audio_length]
    return melcep_interp


def main():
    args = get_arguments()
    # load audio file to use as source of aux features
    if args.use_aux_features:
        feature_source, _ = librosa.load(args.aux_source, sr=SAMPLE_RATE,
                            mono=False)
        # if stereo file, take left channel
        if len(feature_source.shape) > 1:
            feature_source = feature_source[0]
        melcep = calculate_mel_cepstrum(feature_source,
                                        sample_rate=SAMPLE_RATE,
                                        hop_length=args.hop_length,
                                        n_mfcc=args.n_mfcc,
                                        n_fft=args.n_fft,
                                        mean_sub=args.mean_sub,
                                        normalise=args.normalise,
                                        mean_file=args.mean_file,
                                        std_file=args.std_file)

    sess = tf.Session()

    wavenet = WaveNet(
        dilations=DILATIONS,
        residual_channels=RESIDUAL_CHANNELS,
        dilation_channels=DILATION_CHANNELS,
        quantization_channels=QUANTIZATION_CHANNELS,
        skip_channels=SKIP_CHANNELS,
        use_aux_features=args.use_aux_features,
        n_mfcc=args.n_mfcc)

    # placeholders for samples and aux input
    if args.use_aux_features:
        aux_input = tf.placeholder(dtype=tf.float32, shape=(1, None, args.n_mfcc))
        # make samples generated equal to length of aux source
        if melcep.shape[0] < args.samples + wavenet.receptive_field:
            args.samples = melcep.shape[0] - wavenet.receptive_field
    else:
        aux_input = None
    samples = tf.placeholder(tf.int32)

    output = wavenet.generate(samples, aux_input, use_aux_features=args.use_aux_features)

    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)

    # make seed waveform
    waveform = [QUANTIZATION_CHANNELS / 2] * (wavenet.receptive_field - 1)
    waveform.append(np.random.randint(QUANTIZATION_CHANNELS))

    last_step_print = datetime.now()
    # generate waveform
    for step in range(args.samples):
        if len(waveform) > wavenet.receptive_field:
            window = waveform[-wavenet.receptive_field:]
        else:
            window = waveform

        if args.use_aux_features:
            window_aux = melcep[step:step + wavenet.receptive_field]
            window_aux = np.array([window_aux])
            prediction = sess.run([output], feed_dict={samples: window,
                                aux_input: window_aux})[0]
        else:
            prediction = sess.run([output], feed_dict={samples: window})[0]

        sample = np.random.choice(np.arange(QUANTIZATION_CHANNELS), p=prediction)
        waveform.append(sample)

        # print progress each second
        now = datetime.now()
        time_since_print = now - last_step_print
        if time_since_print.total_seconds() > 1.:
            print("Completed samples:", step, "of", args.samples)
            last_step_print = now


    # Decode samples and save as wave file
    decode_in = tf.placeholder(dtype=tf.float32)
    mu = QUANTIZATION_CHANNELS - 1
    signal = 2 * (tf.to_float(decode_in) / mu) - 1
    magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
    decoded = tf.sign(signal) * magnitude

    out = sess.run(decoded, feed_dict={decode_in: waveform})
    librosa.output.write_wav(args.wav_out_path, out, SAMPLE_RATE)
    print("Sound file generated at", args.wav_out_path)


if __name__ == '__main__':
    main()
