import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import math
import os

SAMPLE_RATE = 16000
FFT = 512

dir = "./Data/UsedTestAudio/"
files = os.listdir(dir)
print(files)
for i, file in enumerate(files):
    file
    audio,_ = librosa.core.load(dir + file, sr=16000)
    audio = audio[5117:]
    sftf = librosa.core.stft(audio, FFT)
    plt.figure(i)
    plt.figure(figsize=(8,5))
    librosa.display.specshow(librosa.amplitude_to_db(sftf,ref=np.max),y_axis='log', x_axis='time')
    plt.title(file)

plt.show()
