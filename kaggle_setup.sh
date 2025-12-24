#!/bin/bash
# Kaggle setup script for K-Conditioned Decomposition
# Run this in the first cell of your Kaggle notebook

set -e  # Exit on error

echo "=========================================="
echo "KCD Setup for Kaggle"
echo "=========================================="

# Check if running in Kaggle
if [ ! -d "/kaggle/input" ]; then
    echo "⚠ Warning: This doesn't look like a Kaggle environment"
fi

# Clone repository (or upload files manually)
echo ""
echo "Step 1: Getting code..."
# If you uploaded the repo as a Kaggle dataset:
# cp -r /kaggle/input/kcd-repo/* /kaggle/working/

# Or clone from GitHub:
cd /kaggle/working
git clone https://github.com/notGiGi/UNFOLD.git
mv UNFOLD/* .
rm -rf UNFOLD

echo "✓ Code copied to /kaggle/working"

# Install dependencies
echo ""
echo "Step 2: Installing dependencies..."
pip install -q PyYAML
pip install -q torchvision

echo "✓ Dependencies installed"

# Verify dataset
echo ""
echo "Step 3: Verifying dataset..."
if [ -d "/kaggle/input/coco-2017-dataset/coco2017/test2017" ]; then
    IMAGE_COUNT=$(ls /kaggle/input/coco-2017-dataset/coco2017/test2017/*.jpg 2>/dev/null | wc -l)
    echo "✓ COCO dataset found: $IMAGE_COUNT images"
else
    echo "✗ COCO dataset not found!"
    echo "  Add 'coco-2017-dataset' to your Kaggle notebook inputs"
fi

# Check GPU
echo ""
echo "Step 4: Checking GPU..."
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "Next steps:"
echo "  1. Run: python kaggle_sanity_check.py"
echo "  2. If all checks pass, run: python train_kaggle.py"
echo ""
