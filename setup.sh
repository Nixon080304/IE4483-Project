#!/bin/bash
# =====================================================
# IE4483 Project Setup Script
# Author: Nixon Edward Winata
# =====================================================

echo "IE4483 Project — Environment Setup Starting..."

# 1. Check Python version
if ! command -v python3 &> /dev/null
then
    echo "❌ Python3 not found. Please install Python 3.8+."
    exit
fi

# 2. Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv

# 3. Activate venv
echo "⚙️  Activating virtual environment..."
source .venv/bin/activate

# 4. Upgrade pip and install dependencies
echo "⬆️  Upgrading pip and installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Create folder structure (if not exists)
echo "Ensuring directory structure..."
mkdir -p data/datasets models

# 6. Optional: Download Dogs vs Cats dataset (requires gdown)
read -p "Do you want to download the Dogs vs Cats dataset automatically? (y/n): " choice
if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
    echo "⬇️  Downloading dataset from Google Drive..."
    gdown --id 1q0r6yeHQMS17R3wz-s2FIbMR5DAGZK5v -O data/dogs_vs_cats_dataset.zip
    unzip -q data/dogs_vs_cats_dataset.zip -d data/
    rm data/dogs_vs_cats_dataset.zip
    echo "✅ Dataset ready at data/datasets/"
else
    echo "⚠️  Skipping automatic dataset download."
fi

echo "✅ Setup complete! To start, activate your environment with:"
echo "source .venv/bin/activate"
