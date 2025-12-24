# Kaggle Quick Start - KCD Training

⏱️ **5 minutes to start training**

---

## Step-by-Step Setup

### 1. Create Kaggle Notebook

1. Go to https://kaggle.com/code
2. Click **"New Notebook"**
3. Settings (right sidebar):
   - **Accelerator**: GPU T4 x2
   - **Internet**: ON
4. Add Data → Search **"COCO 2017"** → Add `coco-2017-dataset`

---

### 2. Clone Repository (Cell 1)

```python
# Clone repo to correct location
!git clone https://github.com/notGiGi/UNFOLD.git /kaggle/working/kcd

# Navigate to directory
%cd /kaggle/working/kcd

# Install dependencies
!pip install -q PyYAML

print("✓ Repository cloned!")
```

**Expected output:**
```
Cloning into '/kaggle/working/kcd'...
✓ Repository cloned!
```

---

### 3. Setup Environment (Cell 2)

```python
# Run setup script
!python kaggle_setup.py
```

**Expected output:**
```
============================================================
KCD Kaggle Setup
============================================================

[1/5] Checking directory structure...
  ✓ Repository structure found

[2/5] Configuring Python path...
  ✓ Added /kaggle/working/kcd to sys.path

[3/5] Verifying imports...
  ✓ src.kcd.models imported
  ✓ src.kcd.data.datasets imported
  ✓ src.kcd.train imported

[4/5] Checking dependencies...
  ✓ PyTorch 2.x.x
  ✓ PyYAML installed
  ✓ Pillow installed

[5/5] Checking GPU...
  ✓ GPU: Tesla T4
  ✓ CUDA 11.8
  ✓ Memory: 15.0 GB

[Bonus] Checking COCO dataset...
  ✓ COCO dataset found: 40,670 images

============================================================
SETUP COMPLETE!
============================================================
```

✅ **If you see "SETUP COMPLETE!", proceed to next step**

❌ **If errors occur:**
- **"src/kcd/ not found"**: Make sure you cloned to `/kaggle/working/kcd/`
- **"COCO dataset not found"**: Add `coco-2017-dataset` in Add Data
- **Import errors**: Restart kernel and re-run cells

---

### 4. Run Sanity Check (Cell 3)

```python
# Verify everything works before training
!python kaggle_sanity_check.py
```

**Expected output (takes ~2-3 minutes):**
```
============================================================
GPU Check
============================================================
✓ GPU available: Tesla T4
  Memory: 15.00 GB
  CUDA version: 11.8

============================================================
Dataset Check
============================================================
✓ Dataset path exists
✓ Found 40670 images
✓ Dataset created: 40670 images
✓ Sample image shape: torch.Size([3, 128, 128])

============================================================
Model Forward Pass Check
============================================================
✓ Model built and moved to cuda
  Total params: 1,234,567
  Trainable params: 1,234,567

  Testing K=3...
    ✓ K=3 forward pass successful
  Testing K=5...
    ✓ K=5 forward pass successful
  Testing K=7...
    ✓ K=7 forward pass successful

✓ All forward passes successful

============================================================
SANITY CHECK COMPLETE
============================================================
✓ All checks passed!
```

✅ **If all checks pass, you're ready to train!**

---

### 5. Start Training (Cell 4)

**Option A: Default Training (Recommended)**
```python
!python train_kaggle.py
```

**Option B: Custom Configuration**
```python
import sys
sys.path.insert(0, '/kaggle/working/kcd')

from src.kcd.data.datasets import ImageFolderDataset
from src.kcd.train import train_from_config
from torch.utils.data import DataLoader

# Create dataset
dataset = ImageFolderDataset(
    root_dir="/kaggle/input/coco-2017-dataset/coco2017/test2017",
    image_size=(128, 128),
    normalize=True,
)

# Create loader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
)

# Train
trainer = train_from_config(
    model_config_path="configs/model.yaml",
    train_config_path="configs/training.yaml",
    train_loader=train_loader,
    val_loader=None,
    device=None,
)

print(f"✓ Training complete!")
print(f"Checkpoint: {trainer.checkpoint_dir / 'checkpoint_final.pt'}")
```

**Training output:**
```
Epoch 1/50 | Batch 100/1271 | Loss: 0.0234 | K=5 | LR: 0.0001
Epoch 1/50 | Batch 200/1271 | Loss: 0.0198 | K=4 | LR: 0.0001
...
```

---

## Training Progress

**What to expect:**

| Time     | Epoch | Loss    | Status                          |
|----------|-------|---------|---------------------------------|
| 0-10 min | 1     | ~0.05   | Initial learning                |
| 1 hour   | 10    | ~0.02   | Visible decomposition emerging  |
| 4 hours  | 50    | ~0.008  | Good quality decomposition      |

**Files saved:**
- `checkpoints/checkpoint_epoch_10.pt` (every 10 epochs)
- `checkpoints/checkpoint_final.pt` (final model)
- `logs/training_log.jsonl` (detailed logs)

---

## Monitor Training (Optional)

**In a new cell while training runs:**

```python
# Monitor latest log entries
!tail -20 logs/training_log.jsonl

# Check GPU memory
!nvidia-smi

# List saved checkpoints
!ls -lh checkpoints/
```

---

## After Training

### Visualize Results (Cell 5)

```python
import sys
sys.path.insert(0, '/kaggle/working/kcd')

import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
from src.kcd.models import KCDModel
from src.kcd.data.datasets import ImageFolderDataset
from torch.utils.data import DataLoader

# Load model
checkpoint = torch.load("checkpoints/checkpoint_final.pt")

with open("configs/model.yaml") as f:
    config = yaml.safe_load(f)

model = KCDModel.from_config(config)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.cuda().eval()

# Get test images
dataset = ImageFolderDataset(
    root_dir="/kaggle/input/coco-2017-dataset/coco2017/test2017",
    image_size=(128, 128),
    normalize=True,
)
loader = DataLoader(dataset, batch_size=4, shuffle=True)
batch = next(iter(loader)).cuda()

# Decompose with K=5
with torch.no_grad():
    outputs = model(batch, num_slots=5)

# Denormalize for visualization
def denormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
    return img * std + mean

# Plot first image
K = 5
fig, axes = plt.subplots(2, K + 1, figsize=(15, 6))

# Original
img = denormalize(batch[0]).cpu().permute(1, 2, 0).numpy()
axes[0, 0].imshow(np.clip(img, 0, 1))
axes[0, 0].set_title("Original")
axes[0, 0].axis('off')

# Reconstruction
recon = denormalize(outputs['recons'][0]).cpu().permute(1, 2, 0).numpy()
axes[1, 0].imshow(np.clip(recon, 0, 1))
axes[1, 0].set_title("Reconstruction")
axes[1, 0].axis('off')

# Layers
for k in range(K):
    # RGB
    rgb = denormalize(outputs['layer_rgbs'][0, k]).cpu().permute(1, 2, 0).numpy()
    axes[0, k+1].imshow(np.clip(rgb, 0, 1))
    axes[0, k+1].set_title(f"Layer {k+1}")
    axes[0, k+1].axis('off')

    # Alpha
    alpha = outputs['layer_alphas'][0, k, 0].cpu().numpy()
    axes[1, k+1].imshow(alpha, cmap='gray', vmin=0, vmax=1)
    axes[1, k+1].set_title(f"Mask {k+1}")
    axes[1, k+1].axis('off')

plt.tight_layout()
plt.show()
```

---

## Download Results

1. Click **"Save Version"** (top right)
2. After notebook finishes, click **"Output"** tab
3. Download:
   - `checkpoint_final.pt`
   - `training_log.jsonl`

---

## Common Issues

### ❌ ModuleNotFoundError: No module named 'src.kcd'

**Fix:**
```python
import sys
sys.path.insert(0, '/kaggle/working/kcd')
```

### ❌ Dataset not found

**Fix:**
1. Add Data → Search "COCO 2017"
2. Add `coco-2017-dataset`
3. Restart kernel

### ❌ CUDA out of memory

**Fix in `train_kaggle.py`:**
```python
batch_size = 16  # Reduce from 32
```

---

## Configuration Files

Located in `configs/`:

- **`model.yaml`**: Custom encoder (train from scratch)
- **`model_pretrained.yaml`**: ViT encoder (faster, recommended)
- **`training.yaml`**: Training hyperparameters

To use pretrained encoder:
```python
# In train_kaggle.py, change:
model_config = "configs/model_pretrained.yaml"
```

---

## Complete Cell-by-Cell Example

See **`kaggle_notebook.py`** for full interactive notebook with 12 cells including:
- Setup
- Sanity checks
- Training
- Visualization
- Log analysis

---

## Need Help?

- **Issues**: https://github.com/notGiGi/UNFOLD/issues
- **Documentation**: See `KAGGLE_INSTRUCTIONS.md` for detailed guide

---

**Ready to train? Start with Cell 1! 🚀**
