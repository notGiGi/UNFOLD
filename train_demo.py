"""Production Demo Training - Publication Quality Results.

This trains a model optimized for:
- Visual quality (ViT-B/16 @ 224x224)
- Robustness (50K diverse images)
- Impressive demos (K=2-8 range)

Expected: 12-15 hours on Kaggle T4
Result: Publication-ready decompositions
"""

import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
import random

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
    """Production demo training."""

    logger.info("=" * 60)
    logger.info("K-CONDITIONED DECOMPOSITION - PRODUCTION DEMO TRAINING")
    logger.info("=" * 60)

    # ============================================================
    # CONFIGURATION
    # ============================================================

    # Dataset path
    data_path = "/kaggle/input/coco-2017-dataset/coco2017/train2017"

    # Model configuration - PRODUCTION QUALITY
    model_config = "configs/model_demo.yaml"  # ViT-B/16, 224x224, 128-dim slots

    # Training configuration - PRODUCTION SETTINGS
    train_config = "configs/training_demo.yaml"  # 50 epochs, K=2-8

    # Training settings
    batch_size = 16  # Adjust if OOM
    num_workers = 2

    # PRODUCTION DATASET SIZE
    # 50,000 images = good diversity without excessive training time
    subset_size = 50000

    logger.info("\n" + "=" * 60)
    logger.info("PRODUCTION DEMO CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"  Dataset: {subset_size:,} images from COCO")
    logger.info(f"  Encoder: ViT-B/16 (frozen) @ 224x224")
    logger.info(f"  Slot dim: 128 (high quality)")
    logger.info(f"  K range: 2-8 (flexible)")
    logger.info(f"  Epochs: 50 (full convergence)")
    logger.info(f"  Batch size: {batch_size}")
    logger.info("")
    logger.info("  Expected time: ~12-15 hours")
    logger.info("  Result: Publication-quality decompositions")
    logger.info("=" * 60)

    logger.info("\nSetting up dataset...")
    logger.info(f"  Data path: {data_path}")

    # Create full dataset
    full_dataset = ImageFolderDataset(
        root_dir=data_path,
        image_size=(224, 224),  # FULL RESOLUTION
        use_normalization=True,
    )

    logger.info(f"  Full dataset: {len(full_dataset)} images")

    # Create production subset
    random.seed(42)
    indices = random.sample(range(len(full_dataset)), min(subset_size, len(full_dataset)))
    dataset = Subset(full_dataset, indices)

    logger.info(f"  Production subset: {len(dataset):,} images")

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
    logger.info(f"  Batches per epoch: {len(train_loader):,}")

    logger.info("\nStarting production training...")
    logger.info(f"  Model config: {model_config}")
    logger.info(f"  Training config: {train_config}")

    # GPU info
    if torch.cuda.is_available():
        logger.info(f"  Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("  Device: CPU (NOT RECOMMENDED)")

    logger.info("\n" + "=" * 60)
    logger.info("This will take 12-15 hours. Results will be worth it!")
    logger.info("=" * 60)

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
    logger.info("PRODUCTION TRAINING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Final checkpoint: {trainer.checkpoint_dir / 'checkpoint_final.pt'}")
    logger.info(f"Training logs: {trainer.log_dir}")
    logger.info("\nYour model is now ready for:")
    logger.info("  - Public demos")
    logger.info("  - Paper figures")
    logger.info("  - Any image decomposition")


if __name__ == "__main__":
    main()
