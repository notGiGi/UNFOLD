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
    print("  [ERROR] ERROR: src/kcd/ not found!")
    print("  Make sure you cloned the repo to /kaggle/working/kcd/")
    print("  Run: !git clone https://github.com/notGiGi/UNFOLD.git /kaggle/working/kcd")
    sys.exit(1)

print("  [OK] Repository structure found")

# Check for all required __init__.py files and their content
init_file_contents = {
    cwd / "src" / "__init__.py": '"""KCD (K-Conditioned Decomposition) package."""\n',
    cwd / "src" / "kcd" / "__init__.py": '"""K-Conditioned Decomposition implementation."""\n',
    cwd / "src" / "kcd" / "data" / "__init__.py": '''"""Data loading and preprocessing modules."""

from .datasets import (
    ImageFolderDataset,
    build_dataloader,
    get_dataset,
    get_dataloader,
    DatasetConfig,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

__all__ = [
    "ImageFolderDataset",
    "build_dataloader",
    "get_dataset",
    "get_dataloader",
    "DatasetConfig",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
''',
    cwd / "src" / "kcd" / "models" / "__init__.py": '''"""Neural network models for K-Conditioned Decomposition."""

from .encoder import ConvTokenEncoder, Encoder, PretrainedVisionEncoder
from .slot_attention import SlotAttention
from .decoder import LayerDecoder, Decoder
from .kcd_model import KCDModel

__all__ = [
    "ConvTokenEncoder",
    "Encoder",
    "PretrainedVisionEncoder",
    "SlotAttention",
    "LayerDecoder",
    "Decoder",
    "KCDModel",
]
''',
}

missing_or_empty = []
for init_file, content in init_file_contents.items():
    if not init_file.exists() or init_file.stat().st_size < 10:
        missing_or_empty.append((init_file, content))
        if not init_file.exists():
            print(f"  [WARN] WARNING: Missing {init_file}")
        else:
            print(f"  [WARN] WARNING: Empty or corrupted {init_file}")

# Create or fix __init__.py files
if missing_or_empty:
    print("  --> Creating/fixing __init__.py files...")
    for init_file, content in missing_or_empty:
        init_file.parent.mkdir(parents=True, exist_ok=True)
        init_file.write_text(content)
        print(f"    [OK] Fixed {init_file}")
else:
    print("  [OK] All __init__.py files present and valid")

# Step 2: Add to Python path
print("\n[2/5] Configuring Python path...")
kcd_path = str(cwd)
if kcd_path not in sys.path:
    sys.path.insert(0, kcd_path)
    print(f"  [OK] Added {kcd_path} to sys.path")
else:
    print(f"  [OK] {kcd_path} already in sys.path")

# Step 3: Verify imports
print("\n[3/5] Verifying imports...")
try:
    from src.kcd.models import KCDModel
    print("  [OK] src.kcd.models imported")
except ImportError as e:
    print(f"  [ERROR] ERROR importing models: {e}")
    sys.exit(1)

try:
    from src.kcd.data.datasets import ImageFolderDataset
    print("  [OK] src.kcd.data.datasets imported")
except ImportError as e:
    print(f"  [ERROR] ERROR importing datasets: {e}")
    sys.exit(1)

try:
    from src.kcd.train import train_from_config
    print("  [OK] src.kcd.train imported")
except ImportError as e:
    print(f"  [ERROR] ERROR importing train: {e}")
    sys.exit(1)

# Step 4: Check dependencies
print("\n[4/5] Checking dependencies...")
try:
    import torch
    print(f"  [OK] PyTorch {torch.__version__}")
except ImportError:
    print("  [ERROR] PyTorch not found")
    sys.exit(1)

try:
    import yaml
    print("  [OK] PyYAML installed")
except ImportError:
    print("  [WARN] PyYAML not found, installing...")
    os.system("pip install -q PyYAML")
    import yaml
    print("  [OK] PyYAML installed")

try:
    from PIL import Image
    print("  [OK] Pillow installed")
except ImportError:
    print("  [ERROR] Pillow not found")
    sys.exit(1)

# Step 5: Check GPU
print("\n[5/5] Checking GPU...")
if torch.cuda.is_available():
    print(f"  [OK] GPU: {torch.cuda.get_device_name(0)}")
    print(f"  [OK] CUDA {torch.version.cuda}")
    print(f"  [OK] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  [WARN] No GPU available (will use CPU)")

# Step 6: Check dataset
print("\n[Bonus] Checking COCO dataset...")
data_path = Path("/kaggle/input/coco-2017-dataset/coco2017/test2017")
if data_path.exists():
    image_count = len(list(data_path.glob("*.jpg")))
    print(f"  [OK] COCO dataset found: {image_count:,} images")
else:
    print("  [WARN] COCO dataset not found")
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
