import argparse
import librosa
import numpy as np
import os


def get_arguments():
    parser = argparse.ArgumentParser(description='Calculate norms')
    parser.add_argument('--hop_length', type=int, default=128,
                        help='Size of hop in samples')
    parser.add_argument('--n_mfcc', type=int, default=25,
                        help='Number of mel cepstrum coefficients to return')
    parser.add_argument('--n_fft', type=int, default=512,
                        help='The size of the fft bin used to mfcc')
    parser.add_argument('--data_dir', type=str, default='./Norms',
                        help='The source files for calcuation norms')
    parser.add_argument('--output_dir', type=str, default='./Norms',
                        help='The director to out norms to')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='audio sample rate')
    return parser.parse_args()


def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files


def norms_from_files(directory, output_dir, sample_rate, hop_length, n_mfcc, n_fft):
    files = find_files(directory)
    # average norm and std dev across files
    mean_total = np.zeros([n_mfcc, 1])
    std_total = np.zeros([n_mfcc])
    totalFrames = 0
    for filename in files:
        audio, _ = librosa.load(filename, sr=sample_rate, mono=False)
        if len(audio.shape) > 1:
            audio = audio[0]
        numberOfFrames, mean, std = calculate_mel_cepstrum_norms(audio,
                                                                sample_rate,
                                                                hop_length,
                                                                n_mfcc,
                                                                n_fft)
        totalFrames += numberOfFrames
        mean_total += mean
        std_total += std
    mean = mean_total / totalFrames
    std = std_total / totalFrames
    np.save(output_dir + "_mean_" + "f_" + str(n_fft) + "h_" + str(hop_length)
            + "n_" + str(n_mfcc), mean)
    np.save(output_dir + "_std_" + "f_" + str(n_fft) + "h_" + str(hop_length)
            + "n_" + str(n_mfcc), std)


def calculate_mel_cepstrum_norms(audio, sample_rate, hop_length, n_mfcc, n_fft):
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
    # calulate mean & std
    numberOfFrames = melcep.shape[1]
    mean = np.mean(melcep, axis=1, keepdims=True)
    mean *= numberOfFrames
    std = np.std(melcep, axis=1)
    std *= numberOfFrames
    return numberOfFrames, mean, std

args = get_arguments()

norms_from_files(directory=args.data_dir, output_dir=args.output_dir,
            sample_rate=args.sample_rate, hop_length=args.hop_length,
            n_mfcc=args.n_mfcc, n_fft=args.n_fft)
