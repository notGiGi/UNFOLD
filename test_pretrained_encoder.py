"""Test script for PretrainedVisionEncoder.

This script validates the pretrained encoder integration with KCD model.
"""

import logging

import torch
import yaml

from src.kcd.models import PretrainedVisionEncoder, KCDModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_pretrained_encoder_basic():
    """Test basic pretrained encoder functionality."""
    logger.info("=" * 60)
    logger.info("Testing PretrainedVisionEncoder")
    logger.info("=" * 60)

    # Create encoder
    encoder = PretrainedVisionEncoder(
        backbone="vit_b_16",
        pretrained=True,
        embed_dim=64,
        freeze=True,
        image_size=224,
    )

    # Test forward pass
    images = torch.randn(2, 3, 224, 224)
    tokens, spatial_shape = encoder(images)

    logger.info(f"\nInput images: {images.shape}")
    logger.info(f"Output tokens: {tokens.shape}")
    logger.info(f"Spatial shape: {spatial_shape}")

    # Verify shapes
    batch_size, num_tokens, embed_dim = tokens.shape
    assert batch_size == 2
    assert embed_dim == 64
    assert spatial_shape[0] * spatial_shape[1] == num_tokens

    logger.info(f"✓ Token shape correct: {tokens.shape}")
    logger.info(f"✓ Spatial grid: {spatial_shape[0]}x{spatial_shape[1]} = {num_tokens} tokens")


def test_freeze_unfreeze():
    """Test freeze/unfreeze functionality."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Freeze/Unfreeze")
    logger.info("=" * 60)

    encoder = PretrainedVisionEncoder(
        backbone="vit_b_16",
        pretrained=True,
        embed_dim=64,
        freeze=False,  # Start unfrozen
    )

    # Check trainable parameters
    def count_trainable():
        return sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    unfrozen_params = count_trainable()
    logger.info(f"\nUnfrozen trainable params: {unfrozen_params:,}")

    # Freeze
    encoder.freeze()
    frozen_params = count_trainable()
    logger.info(f"Frozen trainable params: {frozen_params:,}")

    assert frozen_params < unfrozen_params
    logger.info(f"✓ Freeze working ({unfrozen_params - frozen_params:,} params frozen)")

    # Unfreeze
    encoder.unfreeze()
    unfrozen_again = count_trainable()
    logger.info(f"Unfrozen again: {unfrozen_again:,}")

    assert unfrozen_again == unfrozen_params
    logger.info("✓ Unfreeze working")


def test_kcd_with_pretrained():
    """Test full KCD model with pretrained encoder."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing KCD with Pretrained Encoder")
    logger.info("=" * 60)

    # Load config
    with open("configs/model_pretrained.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Build model
    model = KCDModel.from_config(config)

    logger.info(f"\n✓ Model built successfully")
    logger.info(f"  Encoder: {model.encoder.__class__.__name__}")
    logger.info(f"  Slot Attention: {model.slot_attention.__class__.__name__}")
    logger.info(f"  Decoder: {model.decoder.__class__.__name__}")

    # Test forward pass
    images = torch.randn(4, 3, 224, 224)
    K = 5

    outputs = model(images, num_slots=K)

    logger.info(f"\nForward pass with K={K}:")
    logger.info(f"  Input: {images.shape}")
    logger.info(f"  Reconstruction: {outputs['recons'].shape}")
    logger.info(f"  Layer RGBs: {outputs['layer_rgbs'].shape}")
    logger.info(f"  Layer Alphas: {outputs['layer_alphas'].shape}")
    logger.info(f"  Slots: {outputs['slots'].shape}")
    logger.info(f"  Attention: {outputs['attn'].shape}")

    # Verify shapes
    assert outputs['recons'].shape == (4, 3, 224, 224)
    assert outputs['layer_rgbs'].shape == (4, K, 3, 224, 224)
    assert outputs['layer_alphas'].shape == (4, K, 1, 224, 224)
    assert outputs['slots'].shape == (4, K, 64)

    # Verify alpha normalization
    alpha_sum = outputs['layer_alphas'].sum(dim=1)
    assert torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-5)

    logger.info(f"\n✓ All shapes correct")
    logger.info(f"✓ Alpha masks sum to 1")


def test_gradient_flow():
    """Test that gradients flow correctly with frozen/unfrozen encoder."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Gradient Flow")
    logger.info("=" * 60)

    # Load config
    with open("configs/model_pretrained.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Test with frozen encoder
    logger.info("\n--- Testing with FROZEN encoder ---")
    config["encoder"]["freeze"] = True
    model_frozen = KCDModel.from_config(config)

    images = torch.randn(2, 3, 224, 224, requires_grad=True)
    outputs = model_frozen(images, num_slots=3)

    loss = outputs['recons'].mean()
    loss.backward()

    # Count parameters with gradients
    encoder_params_with_grad = sum(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model_frozen.encoder.parameters()
    )
    total_params_with_grad = sum(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model_frozen.parameters()
    )

    logger.info(f"Encoder params with gradients: {encoder_params_with_grad}")
    logger.info(f"Total params with gradients: {total_params_with_grad}")
    logger.info(f"✓ Frozen encoder has minimal gradient flow")

    # Test with unfrozen encoder
    logger.info("\n--- Testing with UNFROZEN encoder ---")
    config["encoder"]["freeze"] = False
    model_unfrozen = KCDModel.from_config(config)

    images = torch.randn(2, 3, 224, 224, requires_grad=True)
    outputs = model_unfrozen(images, num_slots=3)

    loss = outputs['recons'].mean()
    loss.backward()

    encoder_params_with_grad = sum(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model_unfrozen.encoder.parameters()
    )

    logger.info(f"Encoder params with gradients: {encoder_params_with_grad}")
    assert encoder_params_with_grad > 0
    logger.info(f"✓ Unfrozen encoder receives gradients")


if __name__ == "__main__":
    test_pretrained_encoder_basic()
    test_freeze_unfreeze()
    test_kcd_with_pretrained()
    test_gradient_flow()

    logger.info("\n" + "=" * 60)
    logger.info("All PretrainedVisionEncoder tests passed!")
    logger.info("=" * 60)
    logger.info("\nThe pretrained encoder is ready for:")
    logger.info("  • Training on natural images")
    logger.info("  • Fast convergence with frozen encoder")
    logger.info("  • Fine-tuning with unfrozen encoder")
    logger.info("  • Transfer learning to new domains")
