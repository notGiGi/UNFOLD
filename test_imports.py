"""Test all imports work correctly.

Run this to verify all package imports are working.
"""

import sys
from pathlib import Path

# Add current directory to path (simulate Kaggle environment)
sys.path.insert(0, str(Path(__file__).parent))

print("Testing imports...")
print("=" * 60)

# Test 1: Models
try:
    from src.kcd.models import KCDModel, ConvTokenEncoder, PretrainedVisionEncoder
    from src.kcd.models import SlotAttention, LayerDecoder
    print("[OK] src.kcd.models")
except ImportError as e:
    print(f"[FAIL] src.kcd.models - FAILED: {e}")
    sys.exit(1)

# Test 2: Data
try:
    from src.kcd.data.datasets import ImageFolderDataset, build_dataloader
    print("[OK] src.kcd.data.datasets - OK")
except ImportError as e:
    print(f"[FAIL] src.kcd.data.datasets - FAILED: {e}")
    sys.exit(1)

# Test 3: Training
try:
    from src.kcd.train import train_from_config, TrainingConfig, Trainer
    print("[OK] src.kcd.train - OK")
except ImportError as e:
    print(f"[FAIL] src.kcd.train - FAILED: {e}")
    sys.exit(1)

# Test 4: Losses
try:
    from src.kcd.losses import KCDLoss
    print("[OK] src.kcd.losses")
except ImportError as e:
    print(f"[FAIL] src.kcd.losses - FAILED: {e}")
    sys.exit(1)

# Test 5: Utils
try:
    from src.kcd.utils import set_seed
    print("[OK] src.kcd.utils - OK")
except ImportError as e:
    print(f"[FAIL] src.kcd.utils - FAILED: {e}")
    sys.exit(1)

print("=" * 60)
print("All imports successful!")
print("\nYou can now run:")
print("  - kaggle_sanity_check.py")
print("  - train_kaggle.py")
