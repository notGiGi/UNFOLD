"""Training script for Kaggle with COCO dataset.

This script is optimized for Kaggle's environment and COCO 2017 test set.
"""

import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, '/kaggle/working/kcd')

from src.kcd.data.datasets import ImageFolderDataset
from src.kcd.train import train_from_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training script for Kaggle."""

    logger.info("=" * 60)
    logger.info("K-CONDITIONED DECOMPOSITION - KAGGLE TRAINING")
    logger.info("=" * 60)

    # ============================================================
    # CONFIGURATION
    # ============================================================

    # Dataset path (COCO 2017 test set)
    data_path = "/kaggle/input/coco-2017-dataset/coco2017/test2017"

    # Model configuration (choose one)
    # Option 1: Custom encoder (train from scratch)
    model_config = "configs/model.yaml"

    # Option 2: Pretrained encoder (recommended for COCO)
    # model_config = "configs/model_pretrained.yaml"

    # Training configuration
    train_config = "configs/training.yaml"

    # Training settings
    batch_size = 32  # Adjust based on GPU memory
    num_workers = 2  # Kaggle has 2 CPU cores
    num_epochs = 50  # Override config if needed

    # Resume from checkpoint (set to None for fresh start)
    resume_from = None
    # resume_from = "checkpoints/checkpoint_epoch_10.pt"

    # ============================================================
    # DATASET SETUP
    # ============================================================

    logger.info("\nSetting up dataset...")
    logger.info(f"  Data path: {data_path}")

    # Create dataset
    dataset = ImageFolderDataset(
        root_dir=data_path,
        image_size=(128, 128),  # Match decoder output_size in config
        normalize=True,
    )

    logger.info(f"  Dataset size: {len(dataset)} images")

    # Create dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        drop_last=True,  # Drop incomplete batches for stability
    )

    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Batches per epoch: {len(train_loader)}")
    logger.info(f"  Total images per epoch: {len(train_loader) * batch_size}")

    # ============================================================
    # TRAINING
    # ============================================================

    logger.info("\nStarting training...")
    logger.info(f"  Model config: {model_config}")
    logger.info(f"  Training config: {train_config}")

    # GPU info
    if torch.cuda.is_available():
        logger.info(f"  Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("  Device: CPU")

    # Train model
    trainer = train_from_config(
        model_config_path=model_config,
        train_config_path=train_config,
        train_loader=train_loader,
        val_loader=None,  # No validation set for COCO test
        device=None,  # Auto-detect GPU/CPU
        resume_from=resume_from,
    )

    # ============================================================
    # COMPLETION
    # ============================================================

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Final checkpoint: {trainer.checkpoint_dir / 'checkpoint_final.pt'}")
    logger.info(f"Training logs: {trainer.log_file}")
    logger.info("\nNext steps:")
    logger.info("  1. Download checkpoints from /kaggle/working/checkpoints/")
    logger.info("  2. Visualize results with visualize_decomposition.py")
    logger.info("  3. Analyze training logs")


if __name__ == "__main__":
    main()
