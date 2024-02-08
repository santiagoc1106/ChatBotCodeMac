import scipy
from scipy.fft import fft 
import librosa as lb
import os
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def show_voice():
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
        stft_data = lb.stft(audio_data)
        magnitude_spec = np.abs(stft_data)  # Take the absolute value to retain only magnitude
        log_magnitude_spec = lb.amplitude_to_db(magnitude_spec, ref=np.max)
        lb.display.specshow(log_magnitude_spec, sr=sample_rate, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram - {filename}')



    # Define your audio files
    audio_files = ['heyjarvis.mp3', 'name.mp3', 'repeat.mp3', 'time.mp3']
    plt.tight_layout()
    plt.show()

# Extract MFCC features
# mfccs_list = []
# for filename in audio_files:
#     audio_data, sample_rate = lb.load(filename)
#     mfccs = lb.feature.mfcc(y=audio_data, sr=sample_rate)
#     mfccs_list.append(mfccs)

# # Prepare data for training
# # Assuming you have labels for each audio file, otherwise, you need to define them
# # Let's assume labels are just numerical values for demonstration purposes
# labels = ['heyjarvis.mp3', 'name.mp3', 'repeat.mp3', 'time.mp3']  # Example labels
# X = np.array(mfccs_list)
# y = np.array(labels)

# # Convert labels to numerical values
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(y)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Convert data to PyTorch tensors
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train, dtype=torch.int64)
# y_test_tensor = torch.tensor(y_test, dtype=torch.int64)

# # Create DataLoader for training and testing data
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
# test_loader = DataLoader(test_dataset, batch_size=32)

# # Define your neural network model
# class SimpleNN(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(SimpleNN, self).__init__()
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(input_size, 128)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x

# # Initialize model
# input_size = X_train_tensor.shape[1]
# num_classes = len(set(y))
# model = SimpleNN(input_size=input_size, num_classes=num_classes)

# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters())

# # Train the model
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     for inputs, targets in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
    
#     model.eval()
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for inputs, targets in test_loader:
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, dim=1)
#             total += targets.size(0)
#             correct += (predicted == targets).sum().item()

#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%')

# iris = load_iris()
# X = iris.data
# y = iris.target

# # Split data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define hyperparameters grid
# param_grid_nn = {
#     'input_size': [X_train_tensor.shape[1]],  # Fixed input size based on your data
#     'num_classes': [len(set(y))],             # Fixed number of classes based on your data
#     'hidden_size': [64, 128, 256],             # Hidden layer size options
#     'num_layers': [1, 2, 3],                   # Number of hidden layers options
#     'dropout': [0.0, 0.1, 0.2] 
# }


# # Define a function to create the model
# def create_model(input_size, num_classes, hidden_size, num_layers, dropout):
#     layers = []
#     layers.append(nn.Flatten())
#     for _ in range(num_layers):
#         layers.append(nn.Linear(input_size, hidden_size))
#         layers.append(nn.ReLU())
#         layers.append(nn.Dropout(dropout))
#         input_size = hidden_size
#     layers.append(nn.Linear(hidden_size, num_classes))
#     return nn.Sequential(*layers)

# # Wrap the model creation function into an object that can be passed to GridSearchCV
# model_creator = lambda **kwargs: create_model(**kwargs)
# param_grid_nn['model'] = [model_creator]

# # Perform grid search with cross-validation
# grid_search = GridSearchCV(estimator=None, param_grid=param_grid_nn, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)

# # Get the best hyperparameters and model
# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_

# # Evaluate the best model on the test set
# test_score = best_model.score(X_test, y_test)

# # Function to extract features (spectrograms) from audio files
# def extract_features(audio_files, sample_rate=22050):
#     features_list = []
#     for audio_file in audio_files:
#         # Load audio file
#         audio_data, _ = lb.load(audio_file, sr=sample_rate)
#         # Compute spectrogram
#         spectrogram = lb.feature.melspectrogram(y=audio_data, sr=sample_rate)
#         if max_length is not None:
#             spectrogram = lb.util.fix_length(spectrogram, max_length, axis = 1)
#         features_list.append(spectrogram)
#     return features_list

# # Function to preprocess features (if needed)
# def preprocess_features(features_list):
#     preprocessed_features_list = []
#     for features in features_list:
#         # Apply any preprocessing steps here (if needed)
#         preprocessed_features_list.append(features)
#     return preprocessed_features_list

# def extract_mfcc(audio_files, max_length=None):
#     mfccs_list = []
#     max_len_mfccs = 0
#     for audio_file in audio_files:
#         # Load audio file
#         audio_data, sample_rate = lb.load(audio_file)
#         # Compute MFCCs
#         mfccs = lb.feature.mfcc(y=audio_data, sr=sample_rate)
#         # Pad or truncate MFCCs to max_length
#         max_len_mfccs = max(max_len_mfccs, mfccs.shape[1])
#         mfccs_list.append(mfccs)
#     mfccs_list = [np.pad(mfcc, ((0, 0), (0, max_len_mfccs - mfcc.shape[1])), mode='constant') for mfcc in mfccs_list]
#     return mfccs_list

# def extract_spectrograms(audio_files, sample_rate=22050, max_length=None):
#     spectrograms_list = []
#     for audio_file in audio_files:
#         # Load audio file
#         audio_data, _ = lb.load(audio_file, sr=sample_rate)
#         # Compute spectrogram
#         spectrogram = lb.feature.melspectrogram(y=audio_data, sr=sample_rate)
#         # Pad or truncate spectrogram to max_length
#         if max_length is not None:
#             spectrogram = lb.util.fix_length(spectrogram, max_length, axis=1)
#         spectrograms_list.append(spectrogram)
#     return spectrograms_list

# # Example usage
# audio_files = ['heyjarvis.mp3', 'name.mp3', 'repeat.mp3', 'time.mp3']
# max_length = max(len(lb.feature.melspectrogram(y=lb.load(audio_file, sr = 22050)[0], sr = 22050)[1])for audio_file in audio_files)
# spectrograms = extract_features(audio_files, sample_rate = 22050, max_length = max_length)
# mfccs_list = extract_mfcc(audio_files)
# spectrograms_list = extract_spectrograms(audio_files, max_length=max_length)
# max_len_mfccs = max(mfcc.shape[1] for mfcc in mfccs_list)


# for i, mfcss in enumerate (mfccs_list):
#     print(f"MFCCs shape for '{audio_files[i+1]}': {mfccs.shape}")

# X = np.array(mfccs_list)
# print("Shape of the NumPy array:", X.shape)
# # Visualize spectrograms
# plt.figure(figsize=(10, 4))
# for i, spectrogram in enumerate(spectrograms):
#     plt.subplot(1, len(audio_files), i + 1)
#     lb.display.specshow(lb.power_to_db(spectrogram, ref=np.max), sr=sample_rate, x_axis='time', y_axis='mel')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title(f'Spectrogram - {audio_files[i]}')


# def extract_features_from_mic(sample_rate=22050):
#     # Parameters for audio capture
#     FORMAT = pyaudio.paInt16
#     CHANNELS = 1
#     RATE = sample_rate
#     CHUNK = 1024
    
#     # Initialize PyAudio
#     p = pyaudio.PyAudio()

#     # Open stream for audio capture
#     stream = p.open(format=FORMAT,
#                     channels=CHANNELS,
#                     rate=RATE,
#                     input=True,
#                     frames_per_buffer=CHUNK)

#     # Create empty list to store spectrograms
#     spectrograms = []

#     # Main loop for real-time audio processing
#     try:
#         while True:
#             # Read audio data from stream
#             data = stream.read(CHUNK)
#             audio_data = np.frombuffer(data, dtype=np.int16)

#             # Compute spectrogram
#             spectrogram = lb.feature.melspectrogram(y=audio_data, sr=RATE)

#             # Append spectrogram to the list
#             spectrograms.append(spectrogram)

#             # Display spectrogram (optional)
#             plt.clf()
#             plt.imshow(lb.power_to_db(spectrogram, ref=np.max), aspect='auto', origin='lower', cmap='inferno')
#             plt.colorbar(format='%+2.0f dB')
#             plt.xlabel('Time (frames)')
#             plt.ylabel('Frequency (Hz)')
#             plt.title('Real-time Spectrogram')
#             plt.pause(0.001)

#     except KeyboardInterrupt:
#         print("Recording stopped.")
#         pass

#     # Close the audio stream
#     stream.stop_stream()
#     stream.close()
#     p.terminate()

#     return spectrograms

# # Extract spectrograms from microphone
# spectrograms = extract_features_from_mic()

# plt.figure(figsize=(10, 4))
# for i, spectrogram in enumerate(spectrograms_list):
#     plt.subplot(1, len(spectrograms), i + 1)
#     lb.display.specshow(lb.power_to_db(spectrogram, ref=np.max), sr=sample_rate, x_axis='time', y_axis='mel')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title(f'Real-time Spectrogram {i+1}')


# #Visualize MFCCs
# plt.figure(figsize=(10, 4))
# for i, mfccs in enumerate(mfccs_list):
#     plt.subplot(1, len(mfccs_list), i + 1)
#     lb.display.specshow(mfccs, sr=22050, x_axis='time')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title(f'MFCCs - {audio_files[i]}')

# print("Best hyperparameters:", best_params)
# print("Test set accuracy:", test_score)