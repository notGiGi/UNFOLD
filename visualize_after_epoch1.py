"""Visualization script to run after epoch 1.

This script generates the required visualizations:
- Input image
- Reconstruction
- Alpha masks
- RGB layers
- Comparison K=2 vs K=6 on same image
"""

import sys
sys.path.insert(0, '/kaggle/working/kcd')

import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.kcd.models import KCDModel
from src.kcd.data.datasets import ImageFolderDataset
from torch.utils.data import DataLoader


def denormalize(img):
    """Denormalize image for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img.device)
    return img * std + mean


def visualize_single_decomposition(image, outputs, title="Decomposition", save_path=None):
    """Visualize one decomposition (input, recon, alphas, layers)."""
    K = outputs['layer_rgbs'].shape[1]

    # Create figure: 3 rows x (K+1) columns
    # Row 1: Original + RGB layers
    # Row 2: Reconstruction + Alpha masks
    # Row 3: Empty + Composited layers
    fig, axes = plt.subplots(3, K + 1, figsize=(3 * (K + 1), 9))

    # Denormalize
    img = denormalize(image[0]).cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)

    recon = denormalize(outputs['recons'][0]).cpu().permute(1, 2, 0).numpy()
    recon = np.clip(recon, 0, 1)

    # Row 1, Col 0: Original
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Row 2, Col 0: Reconstruction
    axes[1, 0].imshow(recon)
    axes[1, 0].set_title("Reconstruction", fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # Row 3, Col 0: MSE
    mse = torch.nn.functional.mse_loss(outputs['recons'][0], image[0]).item()
    axes[2, 0].text(0.5, 0.5, f"MSE: {mse:.4f}",
                    ha='center', va='center', fontsize=16, fontweight='bold')
    axes[2, 0].axis('off')

    # RGB layers and alphas
    for k in range(K):
        # Row 1: RGB layer
        rgb = denormalize(outputs['layer_rgbs'][0, k]).cpu().permute(1, 2, 0).numpy()
        rgb = np.clip(rgb, 0, 1)
        axes[0, k + 1].imshow(rgb)
        axes[0, k + 1].set_title(f"Layer {k + 1} RGB", fontsize=12)
        axes[0, k + 1].axis('off')

        # Row 2: Alpha mask
        alpha = outputs['layer_alphas'][0, k, 0].cpu().numpy()
        im = axes[1, k + 1].imshow(alpha, cmap='gray', vmin=0, vmax=1)
        axes[1, k + 1].set_title(f"Layer {k + 1} Alpha", fontsize=12)
        axes[1, k + 1].axis('off')

        # Add colorbar for alpha
        plt.colorbar(im, ax=axes[1, k + 1], fraction=0.046, pad=0.04)

        # Row 3: Composited (RGB * Alpha)
        composited = rgb * alpha[:, :, np.newaxis]
        axes[2, k + 1].imshow(composited)
        axes[2, k + 1].set_title(f"Layer {k + 1} Composite", fontsize=12)
        axes[2, k + 1].axis('off')

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.show()


def visualize_k_comparison(image, model, device, save_path=None):
    """Compare K=2 vs K=6 on the same image."""
    fig, axes = plt.subplots(2, 9, figsize=(27, 6))

    img = denormalize(image[0]).cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)

    for row, K in enumerate([2, 6]):
        with torch.no_grad():
            outputs = model(image, num_slots=K)

        recon = denormalize(outputs['recons'][0]).cpu().permute(1, 2, 0).numpy()
        recon = np.clip(recon, 0, 1)

        mse = torch.nn.functional.mse_loss(outputs['recons'][0], image[0]).item()

        # Column 0: Original
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"K={K}: Original", fontsize=12, fontweight='bold')
        axes[row, 0].axis('off')

        # Column 1: Reconstruction
        axes[row, 1].imshow(recon)
        axes[row, 1].set_title(f"Recon (MSE={mse:.4f})", fontsize=12)
        axes[row, 1].axis('off')

        # Columns 2-8: Layers (up to 7)
        for k in range(min(K, 7)):
            if k < K:
                # RGB * Alpha composite
                rgb = denormalize(outputs['layer_rgbs'][0, k]).cpu().permute(1, 2, 0).numpy()
                rgb = np.clip(rgb, 0, 1)
                alpha = outputs['layer_alphas'][0, k, 0].cpu().numpy()
                composite = rgb * alpha[:, :, np.newaxis]

                axes[row, k + 2].imshow(composite)
                axes[row, k + 2].set_title(f"Layer {k + 1}", fontsize=10)
            else:
                axes[row, k + 2].axis('off')
            axes[row, k + 2].axis('off')

    plt.suptitle("K-Conditioning Comparison: K=2 vs K=6 (Same Image)",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.show()


def main():
    """Main visualization script."""
    print("=" * 60)
    print("VISUALIZATION AFTER EPOCH 1")
    print("=" * 60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load checkpoint
    checkpoint_path = Path("checkpoints/checkpoint_epoch_1.pt")
    if not checkpoint_path.exists():
        print(f"\nERROR: Checkpoint not found: {checkpoint_path}")
        print("Run training first to generate checkpoint after epoch 1")
        return

    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model
    with open("configs/model_pretrained.yaml") as f:
        config = yaml.safe_load(f)

    model = KCDModel.from_config(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"  Loaded model from epoch {checkpoint['epoch']}")
    if 'train_loss' in checkpoint:
        print(f"  Training loss: {checkpoint['train_loss']:.4f}")

    # Load dataset
    print("\nLoading test images...")
    dataset = ImageFolderDataset(
        root_dir="/kaggle/input/coco-2017-dataset/coco2017/train2017",
        image_size=(128, 128),
        use_normalization=True,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Create output directory
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    print(f"  Output directory: {output_dir}")

    # Get test image
    batch = next(iter(loader)).to(device)
    print(f"\nTest image shape: {batch.shape}")

    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # 1. Decomposition with K=5
    print("\n1. Single decomposition (K=5)...")
    with torch.no_grad():
        outputs_k5 = model(batch, num_slots=5)

    visualize_single_decomposition(
        batch,
        outputs_k5,
        title="Decomposition After Epoch 1 (K=5)",
        save_path=output_dir / "decomposition_epoch1_k5.png"
    )

    # 2. K=2 vs K=6 comparison
    print("\n2. K-conditioning comparison (K=2 vs K=6)...")
    visualize_k_comparison(
        batch,
        model,
        device,
        save_path=output_dir / "k_comparison_epoch1.png"
    )

    # 3. Additional samples with different K values
    print("\n3. Additional samples...")
    for idx, K in enumerate([2, 3, 4, 5, 6]):
        batch = next(iter(loader)).to(device)

        with torch.no_grad():
            outputs = model(batch, num_slots=K)

        visualize_single_decomposition(
            batch,
            outputs,
            title=f"Sample {idx + 1} (K={K})",
            save_path=output_dir / f"sample{idx + 1}_k{K}.png"
        )

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
