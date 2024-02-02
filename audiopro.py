import scipy
from scipy.fft import fft 
import librosa as lb
import os
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.model_selection import train_test_split 
from tensorflow.keras

#loading my audio file
audio_files= ['heyjarvis.mp3', 'name.mp3', 'repeat.mp3', 'time.mp3']

mfccs_list = []

for filename in audio_files:
    audio_data, sample_rate = lb.load(filename)

    mfccs = lb.feature.mfcc(y = audio_data, sr = sample_rate)

    mfccs_list.append(mfccs)

    # Plot waveform
    plt.figure(figsize=(10, 4))
    plt.subplot(2, len(audio_files), audio_files.index(filename) + 1)
    plt.plot(np.arange(len(audio_data))/sample_rate, audio_data)
    plt.title(f'Waveform - {filename}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # Visualize spectrogram
    plt.subplot(2, len(audio_files), len(audio_files) + audio_files.index(filename) + 1)
    lb.display.specshow(lb.amplitude_to_db(lb.stft(audio_data), ref=np.max), sr=sample_rate, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram - {filename}')

plt.tight_layout()
plt.show()
