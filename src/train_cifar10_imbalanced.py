import os
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm


def create_imbalanced_subset(dataset, reduction_classes=(0, 1, 2), keep_ratio=0.1):
    """
    Simulate class imbalance by keeping only a fraction of samples
    for some classes (e.g., classes 0,1,2 keep only 10%).
    """
    targets = dataset.targets  # list of labels
    indices_to_keep = []

    # First collect indices by class
    class_to_indices = {}
    for idx, label in enumerate(targets):
        class_to_indices.setdefault(label, []).append(idx)

    for cls, idx_list in class_to_indices.items():
        if cls in reduction_classes:
            # keep only keep_ratio of samples
            k = max(1, int(len(idx_list) * keep_ratio))
            indices_to_keep.extend(idx_list[:k])
        else:
            indices_to_keep.extend(idx_list)

    print(f"Original size: {len(dataset)}, Imbalanced size: {len(indices_to_keep)}")
    return Subset(dataset, indices_to_keep)


def compute_class_weights_from_subset(subset, num_classes=10):
    """
    Compute class weights for CrossEntropyLoss based on subset labels.
    """
    # subset.indices maps into the original dataset
    dataset = subset.dataset
    indices = subset.indices

    labels = [dataset.targets[i] for i in indices]
    counts = Counter(labels)
    print("Class counts in imbalanced subset:", counts)

    total = sum(counts.values())
    # Inverse frequency weighting
    weights = []
    for cls in range(num_classes):
        cls_count = counts.get(cls, 1)
        weights.append(total / (num_classes * cls_count))

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    print("Class weights:", weights_tensor)
    return weights_tensor


def create_weighted_sampler(subset, num_classes=10):
    """
    Create a WeightedRandomSampler so that rare classes are oversampled.
    """
    dataset = subset.dataset
    indices = subset.indices

    labels = [dataset.targets[i] for i in indices]
    counts = Counter(labels)

    # weight per class = 1 / count
    class_weight = {cls: 1.0 / counts[cls] for cls in counts}

    # weight per sample
    sample_weights = [class_weight[label] for label in labels]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # 2. Base datasets
    train_dataset_full = datasets.CIFAR10(root="data", train=True,
                                          transform=transform_train, download=True)
    val_dataset = datasets.CIFAR10(root="data", train=False,
                                   transform=transform_test, download=True)

    # 3. Create imbalanced training subset (simulate few samples in classes 0,1,2)
    imbalanced_train_subset = create_imbalanced_subset(
        train_dataset_full,
        reduction_classes=(0, 1, 2),  # e.g. airplane, automobile, bird
        keep_ratio=0.1                 # keep only 10% of them
    )

    # 4a. Method 1: Class-weighted loss
    class_weights = compute_class_weights_from_subset(
        imbalanced_train_subset, num_classes=10
    )
    class_weights = class_weights.to(device)

    # 4b. Method 2: WeightedRandomSampler
    sampler = create_weighted_sampler(imbalanced_train_subset, num_classes=10)

    # DataLoader: use sampler instead of shuffle=True
    train_loader = DataLoader(
        imbalanced_train_subset,
        batch_size=64,
        sampler=sampler,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )

    # 5. Model: same ResNet-18 as before
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    # Use class-weighted CrossEntropy
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0.0
    for epoch in range(5):
        # TRAIN
        model.train()
        running_loss, running_corrects = 0.0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/5"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects / len(train_loader.dataset)

        # VALIDATE
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        print(f"Epoch {epoch+1}: Train acc={train_acc:.4f}, Val acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/best_cifar10_resnet18_imbalanced.pth")
            print(f"✅ New best (imbalanced) model saved ({best_acc:.4f})")

    print(f"Training done on imbalanced data. Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
