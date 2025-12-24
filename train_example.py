"""Example training script for K-Conditioned Decomposition.

This demonstrates how to use the training loop with your own dataset.
"""

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.kcd.data.datasets import ImageFolderDataset, build_dataloader
from src.kcd.train import train_from_config, TrainingConfig, set_seed

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Example training script."""

    # Paths
    data_root = "data/train"  # Replace with your dataset path
    model_config = "configs/model.yaml"
    train_config = "configs/training.yaml"

    # Create dataset
    logger.info(f"Loading dataset from {data_root}")
    dataset = ImageFolderDataset(
        root_dir=data_root,
        image_size=(128, 128),
        use_normalization=True,
    )

    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,  # Faster GPU transfer
    )

    logger.info(f"Dataset size: {len(dataset)} images")
    logger.info(f"Batches per epoch: {len(train_loader)}")

    # Train model
    logger.info("Starting training...")
    trainer = train_from_config(
        model_config_path=model_config,
        train_config_path=train_config,
        train_loader=train_loader,
        val_loader=None,  # Add validation loader if available
        device=None,  # Auto-detect GPU/CPU
        resume_from=None,  # Set to checkpoint path to resume
    )

    logger.info("Training completed!")
    logger.info(f"Final checkpoint saved to: {trainer.checkpoint_dir / 'checkpoint_final.pt'}")


if __name__ == "__main__":
    main()
