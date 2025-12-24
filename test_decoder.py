"""Test script for LayerDecoder module.

This script validates the layer-wise decoder with alpha compositing
and shared weight architecture.
"""

import logging

import torch
import matplotlib.pyplot as plt
from src.kcd.models.decoder import LayerDecoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_decoder() -> None:
    """Test basic LayerDecoder functionality."""
    logger.info("=" * 60)
    logger.info("Testing LayerDecoder")
    logger.info("=" * 60)

    decoder = LayerDecoder(
        slot_dim=64,
        hidden_dims=[64, 64, 64, 64],
        output_size=(128, 128),
        kernel_size=5,
        activation="relu",
    )

    # Create dummy slots
    batch_size = 4
    num_slots = 5
    slot_dim = 64

    slots = torch.randn(batch_size, num_slots, slot_dim)

    logger.info(f"\nInput slots: {slots.shape}")

    # Forward pass
    recons, layer_rgbs, layer_alphas = decoder(slots)

    logger.info(f"Output reconstruction: {recons.shape}")
    logger.info(f"Layer RGBs: {layer_rgbs.shape}")
    logger.info(f"Layer alphas: {layer_alphas.shape}")

    # Verify shapes
    assert recons.shape == (batch_size, 3, 128, 128)
    assert layer_rgbs.shape == (batch_size, num_slots, 3, 128, 128)
    assert layer_alphas.shape == (batch_size, num_slots, 1, 128, 128)

    logger.info("\n✓ Shape tests passed!")


def test_alpha_normalization() -> None:
    """Test that alpha masks sum to 1."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Alpha Normalization")
    logger.info("=" * 60)

    decoder = LayerDecoder(slot_dim=32, output_size=(64, 64))

    slots = torch.randn(2, 4, 32)

    _, _, layer_alphas = decoder(slots)

    # Sum across slots (dim=1)
    alpha_sum = layer_alphas.sum(dim=1)

    logger.info(f"\nAlpha masks shape: {layer_alphas.shape}")
    logger.info(f"Alpha sum shape: {alpha_sum.shape}")
    logger.info(f"Alpha sum mean: {alpha_sum.mean():.6f} (should be ~1.0)")
    logger.info(f"Alpha sum std: {alpha_sum.std():.6f} (should be ~0.0)")

    # Check normalization
    assert torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-5)

    logger.info("\n✓ Alpha masks correctly normalized!")


def test_shared_weights() -> None:
    """Test that decoder uses shared weights across slots."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Shared Weight Architecture")
    logger.info("=" * 60)

    decoder = LayerDecoder(slot_dim=64, output_size=(64, 64))

    # Count parameters
    total_params = sum(p.numel() for p in decoder.parameters())

    logger.info(f"\nTotal decoder parameters: {total_params:,}")

    # With shared weights, param count doesn't depend on K
    # Test with different K values
    for K in [2, 5, 10, 20]:
        slots = torch.randn(1, K, 64)
        recons, rgbs, alphas = decoder(slots)

        logger.info(
            f"K={K:2d}: recons={recons.shape}, rgbs={rgbs.shape}, "
            f"alphas={alphas.shape}"
        )

        assert rgbs.shape[1] == K
        assert alphas.shape[1] == K

    logger.info(
        f"\n✓ Decoder handles variable K with constant {total_params:,} parameters!"
    )


def test_alpha_compositing() -> None:
    """Test that reconstruction is correct alpha composite."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Alpha Compositing")
    logger.info("=" * 60)

    decoder = LayerDecoder(slot_dim=32, output_size=(32, 32))

    slots = torch.randn(2, 3, 32)

    recons, layer_rgbs, layer_alphas = decoder(slots)

    # Manual compositing: Σ_k (alpha_k * rgb_k)
    manual_recons = torch.sum(layer_rgbs * layer_alphas, dim=1)

    logger.info(f"\nReconstructed image: {recons.shape}")
    logger.info(f"Manual composite: {manual_recons.shape}")

    # Check they match
    diff = (recons - manual_recons).abs().max()
    logger.info(f"Max difference: {diff.item():.8f}")

    assert torch.allclose(recons, manual_recons, atol=1e-6)

    logger.info("\n✓ Alpha compositing is correct!")


def test_gradient_flow() -> None:
    """Test that gradients flow through decoder."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Gradient Flow")
    logger.info("=" * 60)

    decoder = LayerDecoder(slot_dim=64, output_size=(64, 64))

    slots = torch.randn(2, 4, 64, requires_grad=True)

    recons, layer_rgbs, layer_alphas = decoder(slots)

    # Compute dummy loss
    loss = recons.sum() + layer_rgbs.sum() + layer_alphas.sum()
    loss.backward()

    # Check that gradients exist
    assert slots.grad is not None
    logger.info(f"\nSlots gradient: shape={slots.grad.shape}")
    logger.info(f"Gradient magnitude: {slots.grad.abs().mean():.6f}")

    # Check decoder parameters have gradients
    for name, param in decoder.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            logger.info(
                f"  {name}: grad magnitude = {param.grad.abs().mean():.6f}"
            )

    logger.info("\n✓ Gradients flow correctly through decoder!")


def test_position_encoding() -> None:
    """Test that position encoding provides spatial information."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Position Encoding")
    logger.info("=" * 60)

    decoder = LayerDecoder(slot_dim=64, output_size=(64, 64))

    # Position encoding should be fixed
    pos_enc = decoder.position_encoding

    logger.info(f"\nPosition encoding shape: {pos_enc.shape}")
    logger.info(f"Position encoding range: [{pos_enc.min():.2f}, {pos_enc.max():.2f}]")

    # Should be in [-1, 1] range
    assert pos_enc.min() >= -1.0 and pos_enc.max() <= 1.0

    # Should be 2 channels (x, y)
    assert pos_enc.shape[0] == 2

    logger.info("\n✓ Position encoding is correct!")


def test_different_output_sizes() -> None:
    """Test decoder with different output resolutions."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Different Output Sizes")
    logger.info("=" * 60)

    sizes = [(64, 64), (128, 128), (256, 256), (128, 256)]

    for size in sizes:
        decoder = LayerDecoder(slot_dim=64, output_size=size)
        slots = torch.randn(2, 3, 64)

        recons, _, _ = decoder(slots)

        logger.info(
            f"Output size {size[0]:3d}x{size[1]:3d}: recons shape = {recons.shape}"
        )

        assert recons.shape == (2, 3, size[0], size[1])

    logger.info("\n✓ Decoder works with different resolutions!")


if __name__ == "__main__":
    test_basic_decoder()
    test_alpha_normalization()
    test_shared_weights()
    test_alpha_compositing()
    test_gradient_flow()
    test_position_encoding()
    test_different_output_sizes()

    logger.info("\n" + "=" * 60)
    logger.info("All LayerDecoder tests passed successfully!")
    logger.info("=" * 60)
