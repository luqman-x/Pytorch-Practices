# 🃏 Playing Card Classification (PyTorch)

## Overview

This project builds an image classification model that identifies **53 classes of playing cards** using transfer learning.

The model is implemented in **PyTorch** and uses a pretrained **EfficientNet-B0** from **timm**.

The goal is to understand and implement a **complete deep learning pipeline**: dataset handling → training → validation → inference.

---

## Dataset

📥 **Download Dataset:**
👉 _(https://drive.google.com/drive/folders/1fwGrdBea1054IeKSR6P4UELBrJ0s-W0m?usp=sharing)_

### Structure

```
dataset/
├── train/
├── valid/
└── test/
```

Each folder contains subfolders representing the class labels (e.g., _"ace of spades"_).

---

## Approach

### 1. Data Handling

- Custom dataset class wrapping `ImageFolder`
- Data transformations:
  - Resize to 128×128
  - Random horizontal flip (augmentation)
  - Normalization (ImageNet stats)

---

### 2. Model

- Backbone: EfficientNet-B0 (pretrained)
- Final layer replaced to match **53 classes**

```python
self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
self.features = nn.Sequential(*list(self.base_model.children())[:-1])

self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(1280, num_classes),
)
```

---

### 3. Training Setup

- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam (lr = 0.001)
- **Batch Size:** 32
- **Scheduler:** ReduceLROnPlateau
- **Early Stopping:** Patience = 7

---

## Training Logic

For each epoch:

1. Forward pass on training batches
2. Compute loss and backpropagate
3. Update weights
4. Evaluate on validation set
5. Adjust learning rate (scheduler)
6. Save best model (early stopping)

---

## Monitoring

The model tracks:

- Training loss
- Validation loss
- Accuracy

Loss curves are plotted after training to observe:

- Convergence
- Overfitting behavior

---

## Inference

The model can:

- Predict class probabilities for a given image
- Visualize predictions using a bar chart

This helps interpret how confident the model is across all classes.

---

## Key Components

- `PlayingcardDataset` → handles data loading
- `SimpleCardClassifier` → model definition
- `EarlyStopping` → prevents overfitting
- Training loop → core learning process

---

## Observations

- Transfer learning significantly improves performance compared to training from scratch
- Early stopping helps avoid overfitting on small datasets
- Validation tracking is essential to detect model generalization

---

## Limitations

- Image size is relatively small (128×128)
- No advanced augmentation (e.g., rotation, color jitter)
- No test set evaluation metrics reported yet

---

## Possible Improvements

- Increase input resolution (e.g., 224×224)
- Fine-tune deeper layers of the backbone
- Add more augmentation
- Evaluate with confusion matrix
- Deploy as an API or web app

---

## Installation

```bash
pip install torch torchvision timm matplotlib numpy pandas tqdm pillow
```

---

## Running the Project

Run the notebook or script to:

1. Load dataset
2. Train model
3. Evaluate performance
4. Visualize predictions

---

## Author

Ali Luqmanu
Biomedical Engineering Student — KNUST

---

## License

MIT License

---
