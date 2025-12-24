"""Setup script for Kaggle environment.

Run this FIRST in your Kaggle notebook to ensure everything is configured correctly.
"""

import sys
import os
from pathlib import Path

print("=" * 60)
print("KCD Kaggle Setup")
print("=" * 60)

# Step 1: Check current directory
print("\n[1/5] Checking directory structure...")
cwd = Path.cwd()
print(f"  Current directory: {cwd}")

# Expected structure: /kaggle/working/kcd/
if not (cwd / "src" / "kcd").exists():
    print("  ✗ ERROR: src/kcd/ not found!")
    print("  Make sure you cloned the repo to /kaggle/working/kcd/")
    print("  Run: !git clone https://github.com/notGiGi/UNFOLD.git /kaggle/working/kcd")
    sys.exit(1)

print("  ✓ Repository structure found")

# Step 2: Add to Python path
print("\n[2/5] Configuring Python path...")
kcd_path = str(cwd)
if kcd_path not in sys.path:
    sys.path.insert(0, kcd_path)
    print(f"  ✓ Added {kcd_path} to sys.path")
else:
    print(f"  ✓ {kcd_path} already in sys.path")

# Step 3: Verify imports
print("\n[3/5] Verifying imports...")
try:
    from src.kcd.models import KCDModel
    print("  ✓ src.kcd.models imported")
except ImportError as e:
    print(f"  ✗ ERROR importing models: {e}")
    sys.exit(1)

try:
    from src.kcd.data.datasets import ImageFolderDataset
    print("  ✓ src.kcd.data.datasets imported")
except ImportError as e:
    print(f"  ✗ ERROR importing datasets: {e}")
    sys.exit(1)

try:
    from src.kcd.train import train_from_config
    print("  ✓ src.kcd.train imported")
except ImportError as e:
    print(f"  ✗ ERROR importing train: {e}")
    sys.exit(1)

# Step 4: Check dependencies
print("\n[4/5] Checking dependencies...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
except ImportError:
    print("  ✗ PyTorch not found")
    sys.exit(1)

try:
    import yaml
    print("  ✓ PyYAML installed")
except ImportError:
    print("  ⚠ PyYAML not found, installing...")
    os.system("pip install -q PyYAML")
    import yaml
    print("  ✓ PyYAML installed")

try:
    from PIL import Image
    print("  ✓ Pillow installed")
except ImportError:
    print("  ✗ Pillow not found")
    sys.exit(1)

# Step 5: Check GPU
print("\n[5/5] Checking GPU...")
if torch.cuda.is_available():
    print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"  ✓ CUDA {torch.version.cuda}")
    print(f"  ✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  ⚠ No GPU available (will use CPU)")

# Step 6: Check dataset
print("\n[Bonus] Checking COCO dataset...")
data_path = Path("/kaggle/input/coco-2017-dataset/coco2017/test2017")
if data_path.exists():
    image_count = len(list(data_path.glob("*.jpg")))
    print(f"  ✓ COCO dataset found: {image_count:,} images")
else:
    print("  ⚠ COCO dataset not found")
    print("    Add 'coco-2017-dataset' to your notebook inputs")

# Summary
print("\n" + "=" * 60)
print("SETUP COMPLETE!")
print("=" * 60)
print("\nNext steps:")
print("  1. Run: !python kaggle_sanity_check.py")
print("  2. If checks pass, run: !python train_kaggle.py")
print("\nOr copy cells from kaggle_notebook.py for interactive training")
print("=" * 60)
