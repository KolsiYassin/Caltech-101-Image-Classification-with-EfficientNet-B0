# Caltech-101 Image Classification with EfficientNet-B0

Deep Learning Lab project for image classification on the **Caltech-101** dataset using **EfficientNet-B0** and transfer learning.

The project investigates the effect of different training strategies, including data augmentation, class-weighted loss, random initialization, and learning-rate scheduling.

## Overview

The task is to classify images into **101 object categories** from the Caltech-101 dataset.

The main objectives were to:

* Explore and analyze the dataset
* Understand the EfficientNet-B0 architecture
* Establish a baseline training configuration
* Evaluate different training and preprocessing strategies
* Compare validation and test performance
* Identify the most effective training configuration

## Dataset

The project uses [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) through the PyTorch `torchvision.datasets.Caltech101` implementation.

Dataset characteristics:

* **8,677 images**
* **101 classes**
* **31–800 images per class**
* Variable image dimensions
* Both RGB and grayscale images
* Strong class imbalance

The dataset is split using stratified sampling:

| Split      | Images | Percentage |
| ---------- | -----: | ---------: |
| Training   |  6,073 |        70% |
| Validation |  1,302 |        15% |
| Test       |  1,302 |        15% |

The split is generated with a fixed random seed (`42`) and the same split is used across experiments for fair comparison.

## Model

The project uses **EfficientNet-B0**, as specified by the task.

The model uses:

* MBConv blocks
* Depthwise convolutions
* Squeeze-and-Excitation
* Residual connections
* SiLU activation
* Global average pooling
* Dropout
* A final linear classifier

The original ImageNet classifier is replaced with a new classifier producing **101 outputs**.

### Transfer Learning

The main experiments use **ImageNet-pretrained weights**.

A separate experiment trains the same architecture from random initialization to investigate the effect of transfer learning.

## Data Processing

Images are converted to RGB and processed to match the expected EfficientNet-B0 input format.

### Baseline preprocessing

```text
Convert to RGB
      ↓
Resize shorter side to 256
      ↓
Center crop to 224 × 224
      ↓
Convert to tensor
      ↓
ImageNet normalization
```

The validation and test sets use deterministic preprocessing.

Training augmentation was investigated separately through several experiments.

## Training

Baseline configuration:

| Parameter      | Value               |
| -------------- | ------------------- |
| Model          | EfficientNet-B0     |
| Initialization | ImageNet pretrained |
| Optimizer      | Adam                |
| Learning rate  | 0.001               |
| Batch size     | 32                  |
| Epochs         | 10                  |
| Loss           | Cross-Entropy Loss  |
| Weight decay   | 0                   |
| Random seed    | 42                  |

The best checkpoint is selected using validation accuracy.

The test set is kept separate and is only used for final evaluation.

## Experiments

Several training configurations were evaluated.

### 1. Baseline

ImageNet-pretrained EfficientNet-B0 with deterministic preprocessing and standard cross-entropy loss.

### 2. Data Augmentation

Different augmentation strategies were evaluated, including:

* Random resized cropping
* Random horizontal flipping
* Color jitter
* Random rotation
* Random affine transformations

### 3. Class-Weighted Loss

Class weights were calculated from the training split to compensate for the class imbalance.

Only the training set was used to calculate the weights.

### 4. Random Initialization

The same EfficientNet-B0 architecture was trained without ImageNet-pretrained weights.

This experiment investigates the effect of transfer learning.

### 5. Learning-Rate Scheduling

The baseline configuration was extended with `ReduceLROnPlateau`:

```python
torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=2
)
```

The learning rate is reduced when the validation loss stops improving.

## Results

The final results are:

| Configuration                | Best Val. Acc. |  Test Acc. |   Macro F1 |  Test Loss |
| ---------------------------- | -------------: | ---------: | ---------: | ---------: |
| Baseline                     |         91.47% |     89.78% |     0.8595 |     0.4270 |
| Augmentation 1               |         84.02% |     84.72% |     0.8230 |     0.8652 |
| Augmentation 2               |         92.32% |     89.17% |     0.8470 |     0.4577 |
| Augmentation 3               |         93.39% |     90.32% |     0.8596 |     0.4416 |
| Class weighted               |         90.17% |     90.17% |     0.8561 |     0.4102 |
| Random initialization        |         71.89% |     69.35% |     0.5606 |     1.8204 |
| **Learning-rate scheduling** |     **94.93%** | **93.32%** | **0.9035** | **0.2810** |

### Main Findings

* ImageNet pretraining substantially improved performance compared with random initialization.
* Data augmentation had mixed results depending on the augmentation strategy.
* Class-weighted loss produced only a small change in overall performance.
* Learning-rate scheduling produced the strongest result among the tested configurations.
* The final learning-rate scheduling configuration achieved **93.32% test accuracy** and **0.9035 Macro F1**.

## Evaluation Metrics

### Accuracy

The percentage of correctly classified test images.

### Macro F1

F1-score is calculated independently for each class and then averaged across all classes.

Macro F1 is particularly useful for this project because the dataset is strongly imbalanced and gives every class equal importance.

### Test Loss

Cross-entropy loss calculated on the held-out test set.

## License

This project was developed as part of a university Deep Learning Lab course.

The Caltech-101 dataset is provided by the California Institute of Technology. Please refer to the [official dataset source](https://data.caltech.edu/records/mzrjq-6wc02) for dataset licensing and usage information.
