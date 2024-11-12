import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling, Conv2D, MaxPool2D, Flatten, Dropout, Dense, Activation


# Step 1: Data Preparation
data_dir = 'C:/1/ECE 172/Project/Project 2/Data'
spectrograms = []
labels = []
# parameters for spectrogram computation

window_size = 256  # Window size for spectrogram computation
overlap = 128  # Number of samples overlap between segments
n_fft = 512  # Number of points used in the FFT
fixed_length = 128  # Define a fixed length for spectrograms

# Loop through each folder (digit) in the data directory
digit_folders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]
for digit_folder in digit_folders:
    digit_path = os.path.join(data_dir, digit_folder)
    # Loop through each audio file in the digit folder
    audio_files = [file for file in os.listdir(digit_path) if file.endswith('.wav')]
    for audio_file in audio_files:
        audio_path = os.path.join(digit_path, audio_file)
        # Read audio file
        audio, fs = librosa.load(audio_path, sr=48000)
        # Compute spectrogram and take absolute value
        S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=overlap, win_length=window_size))

        # Ensure fixed length by padding or truncating
        if S.shape[1] < fixed_length:
            pad_width = fixed_length - S.shape[1]
            S = np.pad(S, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        elif S.shape[1] > fixed_length:
            S = S[:, :fixed_length]

        spectrograms.append(S)
        labels.append(int(digit_folder))

spectrograms = np.array(spectrograms)
labels = np.array(labels)

print('Size of spectrograms array:', spectrograms.shape)
print('Size of labels array:', labels.shape)

# Step 2: Model Architecture

# Define the input shape
input_shape = spectrograms[0].shape

model = Sequential()
model.add(Conv2D(filters=8, kernel_size=(20, 20), activation='relu', input_shape=input_shape))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=16, kernel_size=(10, 10), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=10, activation='softmax'))

# Step 3: Training

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Display the model summary
model.summary()

X_train, X_val, y_train, y_val = train_test_split(spectrograms, labels, test_size=0.2, random_state=42)

# Reshape the input data to match the model's input shape
X_train = X_train.reshape(-1, 257, 128, 1)
X_val = X_val.reshape(-1, 257, 128, 1)


# Step 4: Use at least 10 iterations per Epoch at least 20 Epochs.
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

# Plot training history before optimize
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Step 5: Adjust the Learning Rate and Optimizer

val_loss, val_acc = model.evaluate(X_val, y_val)
print(f'Validation Accuracy: {val_acc:.4f}')

learning_rate = 0.0001
optimizer = Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

# Step 6: Graph
# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()


# Step 7: Confusion Matrix

y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
conf_matrix = confusion_matrix(y_val, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
