import os
import argparse
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm

from datasets import get_transforms
from models import create_resnet18


class TestImageDataset(Dataset):
    """
    Simple dataset for unlabeled test images.
    Returns (image_tensor, filename).
    """
    def __init__(self, img_paths: List[str], transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        filename = os.path.basename(path)
        return img, filename


def numeric_sort_key(path: str):
    """
    Try to sort by numeric id in filename, e.g. '123.jpg' -> 123.
    If that fails, fall back to plain name.
    """
    name = os.path.basename(path)
    stem, _ = os.path.splitext(name)
    try:
        return int(stem)
    except ValueError:
        return stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/datasets")
    parser.add_argument("--test-dir", type=str, default="test")
    parser.add_argument("--model-path", type=str, default="models/best_dogs_cats_resnet18.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output", type=str, default="submission.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Collect test image paths
    test_dir = os.path.join(args.data_root, args.test_dir)
    img_paths = [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(img_paths) == 0:
        raise RuntimeError(f"No images found in test directory: {test_dir}")

    # Sort for consistent ordering (important for submission)
    img_paths = sorted(img_paths, key=numeric_sort_key)
    print(f"Found {len(img_paths)} test images.")

    # 2. Dataset & DataLoader
    test_transform = get_transforms(image_size=args.image_size, train=False)
    test_dataset = TestImageDataset(img_paths, transform=test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 3. Load model
    model = create_resnet18(num_classes=2, pretrained=False, freeze_backbone=False)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # 4. Inference
    all_ids = []
    all_preds = []

    with torch.no_grad():
        for inputs, filenames in tqdm(test_loader, desc="Predict"):
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)   # preds are 0 or 1

            for fname, pred in zip(filenames, preds.cpu().numpy()):
                # Remove file extension for ID, e.g. "123.jpg" -> "123"
                stem, _ = os.path.splitext(fname)
                all_ids.append(stem)
                all_preds.append(int(pred))  # 0 = cat, 1 = dog

    # 5. Build submission DataFrame
    df = pd.DataFrame({
        "id": all_ids,
        "label": all_preds
    })

    # Sort by id just in case
    try:
        df["id_int"] = df["id"].astype(int)
        df = df.sort_values("id_int")
        df = df.drop(columns=["id_int"])
    except ValueError:
        df = df.sort_values("id")

    df.to_csv(args.output, index=False)
    print(f"Saved submission to {args.output}")


if __name__ == "__main__":
    main()

