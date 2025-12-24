"""Kaggle Sanity Check for K-Conditioned Decomposition.

This script validates the KCD implementation on Kaggle with COCO dataset.
Run this first to ensure everything works before full training.
"""

import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.append('/kaggle/working')

from src.kcd.models import KCDModel
from src.kcd.data.datasets import ImageFolderDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_gpu():
    """Verify GPU availability."""
    logger.info("=" * 60)
    logger.info("GPU Check")
    logger.info("=" * 60)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"  CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        logger.warning("⚠ No GPU available, using CPU")

    return device


def check_dataset():
    """Verify COCO dataset loading."""
    logger.info("\n" + "=" * 60)
    logger.info("Dataset Check")
    logger.info("=" * 60)

    data_path = Path("/kaggle/input/coco-2017-dataset/coco2017/test2017")

    if not data_path.exists():
        logger.error(f"✗ Dataset not found at {data_path}")
        logger.info("  Make sure COCO dataset is added to Kaggle notebook")
        return None

    logger.info(f"✓ Dataset path exists: {data_path}")

    # Count images
    image_files = list(data_path.glob("*.jpg"))
    logger.info(f"✓ Found {len(image_files)} images")

    # Create dataset
    dataset = ImageFolderDataset(
        root_dir=str(data_path),
        image_size=(128, 128),
        normalize=True,
    )

    logger.info(f"✓ Dataset created: {len(dataset)} images")

    # Test loading one image
    image = dataset[0]
    logger.info(f"✓ Sample image shape: {image.shape}")
    logger.info(f"  Value range: [{image.min():.3f}, {image.max():.3f}]")

    return dataset


def check_model_configs():
    """Verify model configurations."""
    logger.info("\n" + "=" * 60)
    logger.info("Model Config Check")
    logger.info("=" * 60)

    configs_to_check = [
        "configs/model.yaml",
        "configs/model_pretrained.yaml",
        "configs/training.yaml",
    ]

    configs = {}
    for config_path in configs_to_check:
        path = Path(config_path)
        if not path.exists():
            logger.error(f"✗ Config not found: {config_path}")
            return None

        with open(path) as f:
            config = yaml.safe_load(f)

        configs[config_path] = config
        logger.info(f"✓ Loaded: {config_path}")

    return configs


def check_model_forward(device):
    """Test model forward pass."""
    logger.info("\n" + "=" * 60)
    logger.info("Model Forward Pass Check")
    logger.info("=" * 60)

    # Load model config
    with open("configs/model.yaml") as f:
        config = yaml.safe_load(f)

    # Build model
    model = KCDModel.from_config(config)
    model = model.to(device)
    model.eval()

    logger.info(f"✓ Model built and moved to {device}")
    logger.info(f"  Encoder: {model.encoder.__class__.__name__}")
    logger.info(f"  Slot Attention: {model.slot_attention.__class__.__name__}")
    logger.info(f"  Decoder: {model.decoder.__class__.__name__}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"  Total params: {total_params:,}")
    logger.info(f"  Trainable params: {trainable_params:,}")

    # Test forward pass with different K values
    batch_size = 4
    images = torch.randn(batch_size, 3, 128, 128, device=device)

    for K in [3, 5, 7]:
        logger.info(f"\n  Testing K={K}...")

        with torch.no_grad():
            outputs = model(images, num_slots=K)

        # Verify shapes
        assert outputs['recons'].shape == (batch_size, 3, 128, 128)
        assert outputs['layer_rgbs'].shape == (batch_size, K, 3, 128, 128)
        assert outputs['layer_alphas'].shape == (batch_size, K, 1, 128, 128)
        assert outputs['slots'].shape == (batch_size, K, 64)

        # Verify alpha normalization
        alpha_sum = outputs['layer_alphas'].sum(dim=1)
        assert torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-5)

        logger.info(f"    ✓ K={K} forward pass successful")
        logger.info(f"      Reconstruction: {outputs['recons'].shape}")
        logger.info(f"      Layer RGBs: {outputs['layer_rgbs'].shape}")
        logger.info(f"      Layer Alphas: {outputs['layer_alphas'].shape}")

    logger.info("\n✓ All forward passes successful")
    return model


def check_pretrained_model(device):
    """Test pretrained encoder model."""
    logger.info("\n" + "=" * 60)
    logger.info("Pretrained Model Check")
    logger.info("=" * 60)

    # Load pretrained config
    with open("configs/model_pretrained.yaml") as f:
        config = yaml.safe_load(f)

    logger.info("Building pretrained model (this downloads ViT weights)...")
    model = KCDModel.from_config(config)
    model = model.to(device)
    model.eval()

    logger.info(f"✓ Pretrained model built")
    logger.info(f"  Encoder: {model.encoder.__class__.__name__}")
    logger.info(f"  Backbone: {config['encoder']['backbone']}")
    logger.info(f"  Frozen: {config['encoder']['freeze']}")

    # Count trainable params
    encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"  Encoder trainable params: {encoder_trainable:,}")
    logger.info(f"  Total trainable params: {total_trainable:,}")

    # Test forward pass (224x224 for ViT)
    images = torch.randn(2, 3, 224, 224, device=device)

    with torch.no_grad():
        outputs = model(images, num_slots=5)

    logger.info(f"✓ Pretrained forward pass successful")
    logger.info(f"  Input: {images.shape}")
    logger.info(f"  Reconstruction: {outputs['recons'].shape}")

    return model


def check_dataloader(dataset, device):
    """Test dataloader integration."""
    logger.info("\n" + "=" * 60)
    logger.info("DataLoader Check")
    logger.info("=" * 60)

    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    logger.info(f"✓ DataLoader created")
    logger.info(f"  Batch size: 8")
    logger.info(f"  Num workers: 2")
    logger.info(f"  Total batches: {len(loader)}")

    # Load model
    with open("configs/model.yaml") as f:
        config = yaml.safe_load(f)
    model = KCDModel.from_config(config)
    model = model.to(device)
    model.eval()

    # Test one batch
    logger.info("\nTesting batch loading and forward pass...")
    batch = next(iter(loader))
    batch = batch.to(device)

    logger.info(f"✓ Batch loaded: {batch.shape}")

    with torch.no_grad():
        outputs = model(batch, num_slots=5)

    logger.info(f"✓ Forward pass on real data successful")
    logger.info(f"  Reconstruction: {outputs['recons'].shape}")

    return loader


def check_memory(device):
    """Check GPU memory usage."""
    if device.type != "cuda":
        return

    logger.info("\n" + "=" * 60)
    logger.info("Memory Check")
    logger.info("=" * 60)

    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9

    logger.info(f"GPU Memory:")
    logger.info(f"  Allocated: {allocated:.2f} GB")
    logger.info(f"  Reserved: {reserved:.2f} GB")
    logger.info(f"  Total: {total:.2f} GB")
    logger.info(f"  Free: {total - reserved:.2f} GB")


def main():
    """Run all sanity checks."""
    logger.info("KAGGLE SANITY CHECK FOR K-CONDITIONED DECOMPOSITION")
    logger.info("=" * 60)

    try:
        # 1. GPU check
        device = check_gpu()

        # 2. Dataset check
        dataset = check_dataset()
        if dataset is None:
            logger.error("Dataset check failed. Stopping.")
            return

        # 3. Config check
        configs = check_model_configs()
        if configs is None:
            logger.error("Config check failed. Stopping.")
            return

        # 4. Model forward pass check
        model = check_model_forward(device)

        # 5. Pretrained model check
        pretrained_model = check_pretrained_model(device)

        # 6. DataLoader check
        loader = check_dataloader(dataset, device)

        # 7. Memory check
        check_memory(device)

        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("SANITY CHECK COMPLETE")
        logger.info("=" * 60)
        logger.info("✓ All checks passed!")
        logger.info("\nYou are ready to:")
        logger.info("  1. Run full training with train_kaggle.py")
        logger.info("  2. Experiment with different K values")
        logger.info("  3. Try both custom and pretrained encoders")
        logger.info("  4. Visualize decomposition results")

    except Exception as e:
        logger.error("\n" + "=" * 60)
        logger.error("SANITY CHECK FAILED")
        logger.error("=" * 60)
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
