"""Fast training script for Kaggle - SMALL SUBSET for debugging.

This uses only 1000 images to verify everything works quickly.
"""

import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

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
    """Main training script for Kaggle - SMALL SUBSET."""

    logger.info("=" * 60)
    logger.info("K-CONDITIONED DECOMPOSITION - SMALL SUBSET TEST")
    logger.info("=" * 60)

    # Dataset path (COCO 2017 train set)
    data_path = "/kaggle/input/coco-2017-dataset/coco2017/train2017"

    # Model configuration
    model_config = "configs/model_pretrained.yaml"  # ResNet50, frozen, 128x128

    # Training configuration
    train_config = "configs/training.yaml"

    # Training settings
    batch_size = 8
    num_workers = 2

    # SUBSET SIZE - adjust based on needs
    # 1000 = 1 min, 10000 = 10 min, 50000 = 50 min per 3 epochs
    subset_size = 10000

    logger.info("\nSetting up dataset...")
    logger.info(f"  Data path: {data_path}")
    logger.info(f"  USING SMALL SUBSET: {subset_size} images")

    # Create full dataset
    full_dataset = ImageFolderDataset(
        root_dir=data_path,
        image_size=(128, 128),
        use_normalization=True,
    )

    logger.info(f"  Full dataset: {len(full_dataset)} images")

    # Create small subset
    import random
    random.seed(42)
    indices = random.sample(range(len(full_dataset)), subset_size)
    dataset = Subset(full_dataset, indices)

    logger.info(f"  Subset: {len(dataset)} images")

    # Create dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Batches per epoch: {len(train_loader)}")
    logger.info(f"  Total images per epoch: {len(train_loader) * batch_size}")

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
        val_loader=None,
        device=None,
        resume_from=None,
    )

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Final checkpoint: {trainer.checkpoint_dir / 'checkpoint_final.pt'}")
    logger.info(f"Training logs: {trainer.log_dir}")


if __name__ == "__main__":
    main()
