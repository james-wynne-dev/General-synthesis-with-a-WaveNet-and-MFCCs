import argparse
import os
import sys
import time
import librosa
import random
import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline
from scipy.interpolate import interpolate
from datetime import datetime

from wavenet import WaveNet

CHECKPOINT_EVERY = 1000
NUM_STEPS = 300000
LEARNING_RATE = 1e-3
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAMPLE_SIZE = 18000
MAX_TO_KEEP = 300
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
    parser = argparse.ArgumentParser(description='WaveNet example network')

    parser.add_argument('--hop_length', type=int, default=128,
                        help='Size of hop in samples')
    parser.add_argument('--n_mfcc', type=int, default=25,
                        help='Number of mel cepstrum coefficients to return')
    parser.add_argument('--n_fft', type=int, default=512,
                        help='The size of the fft bin used to mfcc')
    parser.add_argument('--use_aux_features', type=bool, default=False,
                        help='whether to use auxiliary features as input')
    parser.add_argument('--mean_sub', type=bool, default=False,
                        help='whether to subtract the mean from mfccs')
    parser.add_argument('--normalise', type=bool, default=False,
                        help='whether to normalise mfccs to variance 1')
    parser.add_argument('--mean_file', type=str, default=None)
    parser.add_argument('--std_file', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None,
                        help='The directory of audio files.')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps')
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE,
                        help='Size of data chunk to process in each step')
    parser.add_argument('--max_checkpoints', type=int, default=MAX_TO_KEEP,
                        help='max checkpoints to keep')
    parser.add_argument('--save_to', type=str, default=None,
                        help='Directory to save checkpoints')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory to restore from')
    return parser.parse_args()


def get_audio_data(dir, sample_rate, hop_length, n_mfcc, n_fft, mean_sub,
                    normalise, mean,std,receptive_field,sample_size):
    files = os.listdir(dir)
    mean = np.load(mean)
    std = np.load(std)
    def generator():
        while True:
            random.shuffle(files)
            for filename in files:
                filename = dir + "/" + filename
                audio, _ = librosa.load(filename, sr=sample_rate, mono=False)
                # test if file mono, and if not take first channel
                if len(audio.shape) > 1:
                    audio = audio[0]
                aux_features = calculate_mel_cepstrum(audio, sample_rate,
                                hop_length, n_mfcc, n_fft, mean_sub, normalise,
                                mean,std)
                audio = audio.reshape(-1, 1)
                # pad audio and aux
                audio = np.pad(audio, [[receptive_field, 0], [0, 0]],
                               'constant')

                aux_features = np.pad(aux_features,
                                    [[receptive_field, 0], [0, 0]],
                                    'constant')
                # loop to cut pieces
                while len(audio) > receptive_field:
                    piece = audio[:(receptive_field +
                                    sample_size), :]
                    audio = audio[sample_size:, :]
                    aux_piece = aux_features[:(receptive_field +
                                            sample_size), :]
                    aux_features = aux_features[sample_size:, :]

                    yield piece, aux_piece
    return generator


def calculate_mel_cepstrum(audio, sample_rate, hop_length, n_mfcc, n_fft,
                            mean_sub, normalise, mean, std):
    # print("calculate_mel_cepstrum2")
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
    if mean_sub:
        melcep = melcep - mean
    if normalise:
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


def encode(audio, aux, n_mfcc):
    # mu law encode, for 8-bit mu is 255, range 0-255
    mu = tf.to_float(QUANTIZATION_CHANNELS - 1)
    # 1) magnitude of x * mu
    numerator = tf.minimum(tf.abs(audio), 1.0) * mu
    # 2) ln(1 + x)
    numerator = tf.log1p(numerator)
    # 5) denominator = log(QUANTIZATION_CHANNELS + 1)
    denominator = tf.log1p(mu)
    # 6) numerator / denominator
    result = (numerator / denominator)
    # 7) * sign(x)
    result = tf.sign(audio) * result
    # 8) Quantize = zero crossing is 127
    result = tf.to_int32(((result + 1) / 2 ) * mu)

    # one hot encode
    encoded = tf.one_hot(result, QUANTIZATION_CHANNELS, dtype=tf.float32)
    # reshape to [batch, width, channels]
    encoded = tf.reshape(encoded, [1, -1, QUANTIZATION_CHANNELS])

    # reshape aux to [batch, width, channels]
    aux = tf.reshape(aux, [1, -1, n_mfcc])
    return encoded, aux


def store_and_restore(save_to, restore_from):
    if not (bool(save_to) ^ bool(restore_from)):
        raise ValueError("Specify save_to or restore_from"
                        " but not both")
    if restore_from:
        ckpt = tf.train.latest_checkpoint(restore_from)
        if not ckpt:
            raise ValueError("There are no checkpoints in "
                            + restore_from)
        else:
            training_step = int(ckpt.split("ckpt")[-1])
            # save new checkpoint to same directory as restore from
            return restore_from, ckpt, training_step
    return save_to, None, 0


def main():
    args = get_arguments()
    # if checkpoint found training_step will be updated, else it's 0
    save_to, ckpt, training_step = store_and_restore(args.save_to, args.restore_from)

    # Get wavenet object
    wavenet = WaveNet(dilations=DILATIONS,
                    residual_channels=RESIDUAL_CHANNELS,
                    dilation_channels=DILATION_CHANNELS,
                    skip_channels=SKIP_CHANNELS,
                    quantization_channels=QUANTIZATION_CHANNELS,
                    use_aux_features=args.use_aux_features,
                    n_mfcc=args.n_mfcc)

    # create dataset
    data_iterator = get_audio_data(args.data_dir,
                            mean_sub=args.mean_sub,
                            normalise=args.normalise,
                            mean=args.mean_file,
                            std=args.std_file,
                            sample_rate=SAMPLE_RATE,
                            hop_length=args.hop_length,
                            n_mfcc=args.n_mfcc,
                            n_fft=args.n_fft,
                            receptive_field=wavenet.receptive_field,
                            sample_size=SAMPLE_SIZE)
    data = tf.data.Dataset.from_generator(data_iterator, (tf.float32,tf.float32))
    data = data.map(lambda audio, aux: encode(audio, aux, n_mfcc=args.n_mfcc))
    data = data.prefetch(30)
    iterator = data.make_initializable_iterator()
    next_element = iterator.get_next()

    # build wavenet and return loss
    loss = wavenet.loss(input_batch=next_element[0], aux_input=next_element[1])
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE,
                                  epsilon=1e-4)

    # add trainable variables to optimizer
    trainable = tf.trainable_variables()
    optim = optimizer.minimize(loss, var_list=trainable)



    writer = tf.summary.FileWriter(save_to)
    # save graph if first training run
    if not args.restore_from:
        writer.add_graph(tf.get_default_graph())
    summaries = tf.summary.merge_all()

    # Saver and variable initializer
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=args.max_checkpoints)
    init = tf.global_variables_initializer()
    sess = tf.Session()

    last_saved_step = training_step
    step = None
    try:
        sess.run(init)
        if ckpt:
            saver.restore(sess, ckpt)
        sess.run(iterator.initializer)

        for step in range(training_step + 1, args.num_steps):
            start_time = time.time()
            summary, loss_value, _ = sess.run([summaries, loss, optim])
            writer.add_summary(summary, step)
            duration = time.time() - start_time
            print('step {:d} - loss = {:.3f}, ({:.3f} sec/step)'
                  .format(step, loss_value, duration))
            if step % CHECKPOINT_EVERY == 0:
                saver.save(sess, save_to + "model.ckpt" + str(step))
                last_saved_step = step

    except KeyboardInterrupt:
        print()
    finally:
        if step > last_saved_step:
            saver.save(sess, save_to + "/model.ckpt" + str(step))


if __name__ == '__main__':
    main()
