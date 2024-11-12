# Audio Classification using Deep Neural Networks

## Project Overview

This project aims to perform accurate audio classification on spoken digits (0-9) using various Artificial Neural Network models. The dataset contains audio files in `.wav` format, which are processed and classified using the following deep learning models:

- **Deep Convolutional Neural Networks (DCNN)**
- **MobileNet**
- **ResNet50**
- **Inception V3**
- **YOLO (You Only Look Once)**

The audio files are first converted into **spectrograms** as a preprocessing step before being fed into the models.

## Theoretical Background

This project employs deep neural networks for audio classification tasks. The audio data, originally in `.wav` format, is transformed into **spectrograms**, which represent the frequency spectrum over time. This transformation allows the models to better process and classify the spoken digits.

### Models Used:
- **DCNN (Deep Convolutional Neural Network)**: A deep convolutional neural network designed to capture patterns and features from spectrograms.
- **MobileNet**: A lightweight, efficient convolutional neural network model that performs well in mobile and edge devices.
- **ResNet50**: A 50-layer deep residual network optimized for image classification tasks, known for handling the vanishing gradient problem.
- **Inception V3**: A powerful CNN with multiple convolution layers, designed to efficiently process large datasets and provide high accuracy.
- **YOLO (You Only Look Once)**: Primarily a real-time object detection model, adapted here to classify spoken digits by treating spectrograms as images.

## Libraries Used
- **TensorFlow**: A comprehensive open-source framework for machine learning and deep learning.
- **Keras**: High-level neural networks API, running on top of TensorFlow, to simplify building and training deep learning models.
- **Scikit-learn**: A machine learning library for Python, used for data preprocessing and evaluation tasks.
- **NumPy**: A core scientific computing library for Python, providing support for large, multi-dimensional arrays and matrices, along with mathematical functions.
- **Matplotlib**: A plotting library for Python, used for visualizing spectrograms and results.

## Data Preprocessing
1. **Audio Loading**: The `.wav` files containing spoken digits (0-9) are loaded into memory.
2. **Spectrogram Conversion**: The audio signals are converted into spectrograms to represent the frequency spectrum over time.
3. **Normalization**: The spectrograms are normalized to ensure consistent input across all audio files.
4. **Data Splitting**: The dataset is divided into training, validation, and test sets for model evaluation.

## Model Architecture
The audio spectrograms are fed into each of the following deep neural network models for classification:

- **DCNN**: A convolutional neural network designed to detect patterns from spectrograms.
- **MobileNet**: A lightweight model, used here for fast and efficient classification.
- **ResNet50**: A deep residual network that utilizes skip connections to avoid vanishing gradients, suitable for deeper architectures.
- **Inception V3**: Known for its efficient computation and deeper layers, the Inception V3 model is applied to process large amounts of data effectively.
- **YOLO**: Although primarily used for real-time object detection, YOLO is adapted in this project to classify spoken digits based on spectrogram images.

## Results and Evaluation
Each model is evaluated based on performance metrics such as accuracy, precision, recall, and F1 score. The results from all models are compared to determine the most effective approach for classifying spoken digits from audio files.

## Installation

To run this project, you'll need the following dependencies:

- Python 3.x
- TensorFlow
- Keras
- Scikit-learn
- NumPy
- Matplotlib

You can install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
