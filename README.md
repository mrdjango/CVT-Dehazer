# Single Image Dehazing with Convolutional Vision Transformer (CVT)

## Description

This project implements a Single Image Dehazing model using a Convolutional Vision Transformer (CVT). The model is designed to remove haze from images, thereby enhancing the clarity and quality of the visual content. The CVT model leverages convolutional layers for feature extraction and self-attention mechanisms to capture long-range dependencies in the image data. The project includes training and testing scripts, as well as utilities for data loading, augmentation, and metric calculation (PSNR and SSIM) to evaluate model performance.

## Table of Contents

- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Dataset Preparation](#dataset-preparation)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Introduction

Image dehazing is a crucial preprocessing step for various computer vision applications, especially those operating in outdoor environments where haze can significantly degrade image quality. This project aims to provide an effective and efficient solution to the image dehazing problem using state-of-the-art transformer-based techniques.

## Model Architecture

The model architecture consists of a Convolutional Vision Transformer (CVT) which combines the strengths of convolutional neural networks (CNNs) and transformer networks. The model includes:
- **Convolutional Layers** for initial feature extraction.
- **Self-Attention Mechanisms** for capturing long-range dependencies.
- **Feedforward Networks** for further processing of features.
- **Upsampling Layers** to reconstruct the dehazed image.

## Dataset Preparation

To train the dehazing model, you need a dataset containing pairs of hazy and clear images. The dataset should be organized in the following structure:
```
data/
├── train/
│   ├── hazy/
│   └── GT/
├── valid/
│   ├── hazy/
│   └── GT/
└── test/
    ├── hazy/
    └── GT/
```

- `hazy/`: Directory containing hazy images.
- `GT/`: Directory containing ground truth clear images.

## Results

The model's performance is evaluated using Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM). The metrics for each epoch are recorded and saved for further analysis.

<table>
  <tr>
    <td>Hazy Image</td>
     <td>Dehazed Image (with 10 epoch)</td>
  </tr>
  <tr>
    <td><img src="/hazy_image.jpg" width=270 height=480></td>
    <td><img src="/dehazed_image.jpg" width=270 height=480></td>
  </tr>
 </table>


### Sample Metrics Table

| Epoch | Loss  | PSNR  | SSIM  |
|-------|-------|-------|-------|
| 1     | 0.49  | 3.50  |0.0018 |
| ...   | ...   | ...   | ...   |
| 50    |0.0199 | 19.11 | 0.62  |

## Dependencies

- Python 3.8 or higher
- PyTorch
- torchvision
- numpy
- scikit-image
- opencv-python
- pandas

Install the required packages using:
```bash
conda install --file requirements.txt  -c pytorch -c nvidia
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to modify the sections and content as needed to better fit your project's specifics.