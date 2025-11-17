# IE4483 Project — Dogs vs Cats & CIFAR-10 Classification
![Python](https://img.shields.io/badge/python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red)
![License](https://img.shields.io/badge/license-Educational-green)

This repository contains the complete implementation for the **IE4483 Artificial Intelligence and Data Mining** project 2.

---

## Quick Start

For a one-command setup:
```bash
bash setup.sh
```
This script will:
1. Create a Python virtual environment  
2. Install all dependencies  
3. Optionally download and extract the Dogs vs Cats dataset automatically  
4. Prepare everything for training

---

## Requirements
- Python 3.8 or above  
- GPU with CUDA (recommended but optional)  
- Git installed  
- Internet connection (for downloading CIFAR-10 or datasets)

---

## Project Structure
```
ie4483-dogs-vs-cats/
│
├── data/                  # Dataset directory (not tracked in Git)
│   └── datasets/          # train/, val/, test/ for Dogs vs Cats
│
├── models/                # Saved model weights (.pth)
│
├── src/
│   ├── datasets.py        # Data loaders + augmentations
│   ├── models.py          # ResNet-18 model definition
│   ├── train_dogs_cats.py # Train on Dogs vs Cats
│   ├── predict_test.py    # Predict test set → submission.csv
│   ├── train_cifar10.py   # Train on CIFAR-10 (balanced)
│   └── train_cifar10_imbalanced.py # CIFAR-10 with imbalance handling
│
├── submission.csv         # Output for Dogs vs Cats test set
├── requirements.txt
├── setup.sh
└── README.md
```

---

## Setup Instructions

### 1️ Clone the repository
```bash
git clone https://github.com/Nixon080304/IE4483-Project.git
cd IE4483-Project
```

### 2 Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

### 3️ Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dogs vs Cats Classification

### Step 1: Prepare the dataset
Download the dataset from [Google Drive](https://drive.google.com/file/d/1q0r6yeHQMS17R3wz-s2FIbMR5DAGZK5v/view)  
and extract it under `data/datasets/` so the structure is:

```
data/datasets/
├── train/
│   ├── cat/
│   └── dog/
├── val/
│   ├── cat/
│   └── dog/
└── test/
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

### Step 2: Train the model
```bash
python src/train_dogs_cats.py --data-root data/datasets --epochs 5
```
Best model will be saved to:
```
models/best_dogs_cats_resnet18.pth
```

Example output:
```
Epoch 1/5
Train loss: 0.2874, acc: 0.9054 - Val acc: 0.9568
New best model saved with val acc: 0.9568
```

### Step 3: Generate predictions for test set
```bash
python src/predict_test.py \
  --data-root data/datasets \
  --model-path models/best_dogs_cats_resnet18.pth \
  --output submission.csv
```
Output format:
```
id,label
1,0
2,1
3,1
...
```

---

## CIFAR-10 Classification

### Train baseline model
```bash
python src/train_cifar10.py
```
This automatically downloads the CIFAR-10 dataset and trains for 5 epochs.  
Best model → `models/best_cifar10_resnet18.pth`.

### Train imbalanced version (Part h)
```bash
python src/train_cifar10_imbalanced.py
```
This simulates imbalance (reduces classes 0, 1, 2 → 10 % of data) and trains using:
- Class-weighted loss  
- WeightedRandomSampler for oversampling  

Resulting weights → `models/best_cifar10_resnet18_imbalanced.pth`.

---

## Key Results

| Task                        | Dataset Size | Best Validation Accuracy |
|-----------------------------|--------------|--------------------------|
| Dogs vs Cats (balanced)     | 20 000 train / 5 000 val | **97.08 %** |
| CIFAR-10 (balanced)         | 50 000 train / 10 000 val | **81.71 %** |
| CIFAR-10 (imbalanced + fix) | 36 500 train / 10 000 val | **73.63 %** |

---

## Notes
- All models use **ResNet-18** backbone pretrained on ImageNet.
- Training/validation are GPU-accelerated if `cuda` is available.
- Datasets are ignored by Git (.gitignore) due to large file size.
- Scripts are modular and reusable for custom datasets.
---

## License
Educational use only — NTU IE4483 Project 2025.
