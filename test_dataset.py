"""Quick test script for ImageFolderDataset.

This script demonstrates the dataset functionality and can be used
to verify the implementation before full training.
"""

import logging
from pathlib import Path

import torch
from src.kcd.data import ImageFolderDataset, build_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_dataset() -> None:
    """Test basic dataset functionality."""
    logger.info("Testing ImageFolderDataset...")

    # You would replace this with your actual image directory
    test_dir = "path/to/test/images"

    if not Path(test_dir).exists():
        logger.warning(
            f"Test directory {test_dir} does not exist. "
            "Please update test_dir variable with actual image path."
        )
        return

    # Create dataset with different configurations
    configs = [
        {"name": "basic", "crop": None, "norm": False},
        {"name": "with_center_crop", "crop": "center", "norm": False},
        {"name": "with_normalization", "crop": None, "norm": True},
    ]

    for config in configs:
        logger.info(f"\n--- Testing: {config['name']} ---")

        dataset = ImageFolderDataset(
            root_dir=test_dir,
            image_size=(128, 128),
            crop_type=config["crop"],
            use_normalization=config["norm"],
        )

        logger.info(f"Dataset size: {len(dataset)}")

        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"Sample shape: {sample.shape}")
            logger.info(f"Sample dtype: {sample.dtype}")
            logger.info(f"Sample range: [{sample.min():.3f}, {sample.max():.3f}]")


def test_dataloader() -> None:
    """Test build_dataloader function."""
    logger.info("\n\nTesting build_dataloader...")

    test_dir = "path/to/test/images"

    if not Path(test_dir).exists():
        logger.warning(
            f"Test directory {test_dir} does not exist. "
            "Please update test_dir variable with actual image path."
        )
        return

    config = {
        "image_size": [128, 128],
        "batch_size": 4,
        "num_workers": 0,
        "crop_type": "center",
        "use_normalization": True,
    }

    loader = build_dataloader(test_dir, config, shuffle=True)

    logger.info(f"DataLoader created: {len(loader)} batches")

    # Test loading one batch
    for batch in loader:
        logger.info(f"Batch shape: {batch.shape}")
        logger.info(f"Batch dtype: {batch.dtype}")
        logger.info(f"Batch range: [{batch.min():.3f}, {batch.max():.3f}]")
        break


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("ImageFolderDataset Test Script")
    logger.info("=" * 60)

    test_basic_dataset()
    test_dataloader()

    logger.info("\n" + "=" * 60)
    logger.info("Tests completed!")
    logger.info("=" * 60)
