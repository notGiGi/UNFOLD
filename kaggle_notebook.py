"""Complete Kaggle Notebook Code for KCD Training.

Copy this entire file into a Kaggle notebook cell-by-cell.
Each section is marked with # CELL X comments.
"""

# ============================================================
# CELL 1: Setup and Installation
# ============================================================

import sys
import os
from pathlib import Path

# Install dependencies
print("Installing dependencies...")
!pip install -q PyYAML

# Clone repository
print("\nCloning KCD repository...")
!git clone https://github.com/notGiGi/UNFOLD.git /kaggle/working/kcd
%cd /kaggle/working/kcd

# Add to path
sys.path.insert(0, '/kaggle/working/kcd')

print("✓ Setup complete!")


# ============================================================
# CELL 2: Verify Environment
# ============================================================

import torch
import yaml

print("Environment Check:")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Check dataset
data_path = Path("/kaggle/input/coco-2017-dataset/coco2017/test2017")
if data_path.exists():
    image_count = len(list(data_path.glob("*.jpg")))
    print(f"\n✓ COCO dataset found: {image_count:,} images")
else:
    print("\n✗ COCO dataset not found!")
    print("  Add 'coco-2017-dataset' to your notebook inputs")


# ============================================================
# CELL 3: Run Sanity Check
# ============================================================

print("Running sanity checks...")
print("This will verify model building and forward passes\n")

!python kaggle_sanity_check.py


# ============================================================
# CELL 4: Quick Test - Single Batch Forward Pass
# ============================================================

from src.kcd.models import KCDModel
from src.kcd.data.datasets import ImageFolderDataset
from torch.utils.data import DataLoader

# Create small dataset
print("Testing on real COCO images...")
dataset = ImageFolderDataset(
    root_dir="/kaggle/input/coco-2017-dataset/coco2017/test2017",
    image_size=(128, 128),
    use_normalization=True,
)

loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

# Load model
with open("configs/model.yaml") as f:
    config = yaml.safe_load(f)

model = KCDModel.from_config(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Forward pass
batch = next(iter(loader)).to(device)
print(f"\nInput batch: {batch.shape}")

with torch.no_grad():
    outputs = model(batch, num_slots=5)

print(f"✓ Forward pass successful!")
print(f"  Reconstruction: {outputs['recons'].shape}")
print(f"  Layer RGBs: {outputs['layer_rgbs'].shape}")
print(f"  Layer Alphas: {outputs['layer_alphas'].shape}")
print(f"  Slots: {outputs['slots'].shape}")

# Verify reconstruction quality
recon_loss = torch.nn.functional.mse_loss(outputs['recons'], batch)
print(f"\nRandom init MSE loss: {recon_loss.item():.4f}")
print("(Should decrease significantly during training)")


# ============================================================
# CELL 5: Visualize Initial Decomposition (Before Training)
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

def denormalize(img):
    """Denormalize image for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img.device)
    return img * std + mean

def visualize_decomposition(image, outputs, idx=0):
    """Visualize decomposition for one image."""
    K = outputs['layer_rgbs'].shape[1]

    fig, axes = plt.subplots(2, K + 1, figsize=(3 * (K + 1), 6))

    # Original image
    img = denormalize(image[idx]).cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis('off')

    # Reconstruction
    recon = denormalize(outputs['recons'][idx]).cpu().permute(1, 2, 0).numpy()
    recon = np.clip(recon, 0, 1)
    axes[1, 0].imshow(recon)
    axes[1, 0].set_title("Reconstruction")
    axes[1, 0].axis('off')

    # Layers
    for k in range(K):
        # RGB layer
        rgb = denormalize(outputs['layer_rgbs'][idx, k]).cpu().permute(1, 2, 0).numpy()
        rgb = np.clip(rgb, 0, 1)
        axes[0, k + 1].imshow(rgb)
        axes[0, k + 1].set_title(f"Layer {k + 1} RGB")
        axes[0, k + 1].axis('off')

        # Alpha mask
        alpha = outputs['layer_alphas'][idx, k, 0].cpu().numpy()
        axes[1, k + 1].imshow(alpha, cmap='gray', vmin=0, vmax=1)
        axes[1, k + 1].set_title(f"Layer {k + 1} Alpha")
        axes[1, k + 1].axis('off')

    plt.tight_layout()
    plt.show()

# Visualize random initialization
print("Decomposition before training (random init):")
visualize_decomposition(batch, outputs, idx=0)


# ============================================================
# CELL 6: Configure Training
# ============================================================

# Training configuration
TRAINING_CONFIG = {
    # Model
    "model_config": "configs/model.yaml",  # or "configs/model_pretrained.yaml"

    # Data
    "batch_size": 32,
    "num_workers": 2,
    "image_size": (128, 128),

    # Training
    "num_epochs": 50,
    "save_every": 10,  # Save checkpoint every N epochs

    # Resume
    "resume_from": None,  # or "checkpoints/checkpoint_epoch_10.pt"
}

print("Training Configuration:")
print("=" * 60)
for key, value in TRAINING_CONFIG.items():
    print(f"  {key}: {value}")


# ============================================================
# CELL 7: Start Training
# ============================================================

from src.kcd.train import train_from_config
from torch.utils.data import DataLoader

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

# Create dataset and loader
dataset = ImageFolderDataset(
    root_dir="/kaggle/input/coco-2017-dataset/coco2017/test2017",
    image_size=TRAINING_CONFIG["image_size"],
    use_normalization=True,
)

train_loader = DataLoader(
    dataset,
    batch_size=TRAINING_CONFIG["batch_size"],
    shuffle=True,
    num_workers=TRAINING_CONFIG["num_workers"],
    pin_memory=True,
    drop_last=True,
)

print(f"\nDataset: {len(dataset)} images")
print(f"Batches per epoch: {len(train_loader)}")

# Train
trainer = train_from_config(
    model_config_path=TRAINING_CONFIG["model_config"],
    train_config_path="configs/training.yaml",
    train_loader=train_loader,
    val_loader=None,
    device=None,  # Auto-detect
    resume_from=TRAINING_CONFIG["resume_from"],
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)


# ============================================================
# CELL 8: Load Best Checkpoint and Evaluate
# ============================================================

# Load final checkpoint
checkpoint_path = "checkpoints/checkpoint_final.pt"
print(f"Loading checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path)

# Rebuild model
with open(TRAINING_CONFIG["model_config"]) as f:
    config = yaml.safe_load(f)

model = KCDModel.from_config(config)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
print(f"  Training loss: {checkpoint.get('train_loss', 'N/A'):.4f}")


# ============================================================
# CELL 9: Visualize Results After Training
# ============================================================

# Get fresh batch
test_loader = DataLoader(dataset, batch_size=8, shuffle=True)
test_batch = next(iter(test_loader)).to(device)

# Test different K values
for K in [3, 5, 7]:
    print(f"\nTesting K={K}")

    with torch.no_grad():
        outputs = model(test_batch, num_slots=K)

    # Compute loss
    recon_loss = torch.nn.functional.mse_loss(outputs['recons'], test_batch)
    print(f"  Reconstruction MSE: {recon_loss.item():.4f}")

    # Visualize
    print(f"\n  Visualizing decomposition with K={K}:")
    visualize_decomposition(test_batch, outputs, idx=0)


# ============================================================
# CELL 10: Analyze Training Logs
# ============================================================

import json
import matplotlib.pyplot as plt

# Load training logs
log_file = "logs/training_log.jsonl"
print(f"Analyzing logs from: {log_file}\n")

logs = []
with open(log_file) as f:
    for line in f:
        logs.append(json.loads(line))

print(f"Total training steps: {len(logs)}")

# Extract metrics
epochs = [log['epoch'] for log in logs]
total_losses = [log['total_loss'] for log in logs]
recon_losses = [log['recon_loss'] for log in logs]

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Total loss
axes[0].plot(epochs, total_losses, alpha=0.3, label='Per-batch')
# Smooth with rolling average
window = 100
if len(total_losses) > window:
    smoothed = np.convolve(total_losses, np.ones(window)/window, mode='valid')
    axes[0].plot(range(window-1, len(total_losses)), smoothed, 'r-', linewidth=2, label='Smoothed')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Total Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Reconstruction loss
axes[1].plot(epochs, recon_losses, alpha=0.3, label='Per-batch')
if len(recon_losses) > window:
    smoothed = np.convolve(recon_losses, np.ones(window)/window, mode='valid')
    axes[1].plot(range(window-1, len(recon_losses)), smoothed, 'r-', linewidth=2, label='Smoothed')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Reconstruction Loss')
axes[1].set_title('Reconstruction Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nFinal metrics:")
print(f"  Initial total loss: {total_losses[0]:.4f}")
print(f"  Final total loss: {total_losses[-1]:.4f}")
print(f"  Improvement: {(total_losses[0] - total_losses[-1]) / total_losses[0] * 100:.1f}%")


# ============================================================
# CELL 11: Save Sample Visualizations
# ============================================================

print("Generating sample visualizations...")

# Create output directory
output_dir = Path("/kaggle/working/visualizations")
output_dir.mkdir(exist_ok=True)

# Generate visualizations for different images and K values
test_loader = DataLoader(dataset, batch_size=16, shuffle=True)
test_batch = next(iter(test_loader)).to(device)

for K in [3, 5, 7]:
    print(f"\nGenerating K={K} visualizations...")

    with torch.no_grad():
        outputs = model(test_batch, num_slots=K)

    # Save first 4 images
    for idx in range(4):
        fig, axes = plt.subplots(2, K + 1, figsize=(3 * (K + 1), 6))

        # Original
        img = denormalize(test_batch[idx]).cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("Original")
        axes[0, 0].axis('off')

        # Reconstruction
        recon = denormalize(outputs['recons'][idx]).cpu().permute(1, 2, 0).numpy()
        recon = np.clip(recon, 0, 1)
        axes[1, 0].imshow(recon)
        axes[1, 0].set_title("Reconstruction")
        axes[1, 0].axis('off')

        # Layers
        for k in range(K):
            rgb = denormalize(outputs['layer_rgbs'][idx, k]).cpu().permute(1, 2, 0).numpy()
            rgb = np.clip(rgb, 0, 1)
            axes[0, k + 1].imshow(rgb)
            axes[0, k + 1].set_title(f"Layer {k + 1}")
            axes[0, k + 1].axis('off')

            alpha = outputs['layer_alphas'][idx, k, 0].cpu().numpy()
            axes[1, k + 1].imshow(alpha, cmap='gray', vmin=0, vmax=1)
            axes[1, k + 1].set_title(f"Mask {k + 1}")
            axes[1, k + 1].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / f"decomposition_K{K}_sample{idx}.png", dpi=150, bbox_inches='tight')
        plt.close()

print(f"\n✓ Visualizations saved to {output_dir}")
print(f"  Download them from the Kaggle output tab")


# ============================================================
# CELL 12: Download Results
# ============================================================

print("Files ready for download:")
print("=" * 60)
print("\nCheckpoints:")
!ls -lh checkpoints/

print("\nLogs:")
!ls -lh logs/

print("\nVisualizations:")
!ls -lh visualizations/

print("\n" + "=" * 60)
print("Training complete! Download files from:")
print("  • /kaggle/working/checkpoints/")
print("  • /kaggle/working/logs/")
print("  • /kaggle/working/visualizations/")
print("=" * 60)
