import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import librosa
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to load and preprocess audio data
def load_and_preprocess_audio(folder_path):
    spectrograms = []
    labels = []

    # parameters for spectrogram computation
    window_size = 256  # Window size for spectrogram computation
    overlap = 128  # Number of samples overlap between segments
    n_fft = 512  # Number of points used in the FFT
    fixed_length = 128  # Define a fixed length for spectrograms

    # Loop through each folder (digit) in the data directory
    digit_folders = [folder for folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, folder))]

    for digit_folder in digit_folders:
        digit_path = os.path.join(folder_path, digit_folder)

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

            # Convert to RGB image by duplicating the single channel across three channels
            S_rgb = np.stack((S,) * 3, axis=-1)

            spectrograms.append(S_rgb)
            labels.append(int(digit_folder))

    spectrograms = np.array(spectrograms)
    labels = np.array(labels)

    return spectrograms, labels


# Load and preprocess audio data
data_folder = 'C:/1/ECE 172/Project/Project 2/Data'
spectrograms, labels = load_and_preprocess_audio(data_folder)


X_train, X_val, y_train, y_val = train_test_split(spectrograms, labels, test_size=0.2, random_state=42)

input_shape = spectrograms[0].shape # RGB spectrogram

#  dropout rate to prevent overfitting
dropout_rate = 0.5

def create_inceptionv3_model(input_shape, dropout_rate):
    base_model = InceptionV3(include_top=False, input_shape=input_shape, weights='imagenet')

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation='softmax')  # Output layer for 10 classes
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

inceptionv3_model = create_inceptionv3_model(input_shape, dropout_rate)

# Data augmentation
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

history = inceptionv3_model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=20, validation_data=(X_val, y_val))

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('InceptionV3 Accuracy vs. Epoch (with Dropout and Data Augmentation)')
plt.show()

val_loss, val_acc = inceptionv3_model.evaluate(X_val, y_val)
print(f'Validation Accuracy for InceptionV3: {val_acc:.4f}')

y_pred = np.argmax(inceptionv3_model.predict(X_val), axis=-1)

conf_matrix = confusion_matrix(y_val, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for InceptionV3')
plt.show()
