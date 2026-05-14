# Gujarati Number Predictor

A PyTorch-based Convolutional Neural Network (CNN) designed to recognize handwritten Gujarati digits (0-9).

## Overview
This project trains a custom CNN to classify Gujarati numerals. The dataset consists of images of handwritten Gujarati digits. The neural network architecture includes multiple convolutional layers with Batch Normalization and Dropout to prevent overfitting, topped with an adaptive average pooling layer and a fully connected classifier.

## Features
* **PyTorch CNN Architecture**: 5-block Convolutional Network with Batch Normalization.
* **Data Augmentation**: Robust training pipeline with random rotations, affine transformations, and color jitter to handle various handwriting styles.
* **LR Scheduler & AdamW**: Uses `ReduceLROnPlateau` and `AdamW` optimizer for optimal convergence.
* **Ready-to-use Predictor**: A clean script to feed any image and get the predicted Gujarati digit along with the confidence score.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gujarati-number-predictor.git
   cd gujarati-number-predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Structure
Make sure your dataset is structured like this in the root directory before training:
```
Train Set/
  ├── 0_zero/
  ├── 1_one/
  └── ...
Test Set/
  ├── 0_zero/
  ├── 1_one/
  └── ...
```

## Usage

### 1. Training the Model
To train the model from scratch, simply run:
```bash
python train.py
```
This will augment the data, train the model over the specified epochs, and save the best weights to `gujarati_digit_model_best.pth`.

### 2. Making Predictions
To predict a digit from a single image:
1. Place your image in the directory (e.g., `test.jpg`).
2. Update the `image_path` variable at the bottom of `predit.py` (if necessary).
3. Run the prediction script:
```bash
python predit.py
```

Output Example:
```text
Using device: cuda
Model loaded successfully!

Original Image Size: (256, 256)
Tensor Shape: torch.Size([1, 256, 256])

==========================
Predicted Digit : 7
Confidence      : 92.39%
==========================
```

## Model Architecture
* Input: `256x256` Grayscale image
* 5 x `(Conv2D -> BatchNorm -> ReLU -> MaxPool2D)`
* `AdaptiveAvgPool2D(4, 4)`
* Flatten
* `Linear(4096, 1024) -> ReLU -> Dropout(0.5)`
* `Linear(1024, 256) -> ReLU -> Dropout(0.25)`
* `Linear(256, 10)` (Output classes)

## License
MIT License
