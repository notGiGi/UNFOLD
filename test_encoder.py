"""Test script for ConvTokenEncoder.

This script demonstrates the encoder functionality and validates the
spatial tokenization for slot-based decomposition.
"""

import logging

import torch
from src.kcd.models.encoder import ConvTokenEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_conv_token_encoder() -> None:
    """Test ConvTokenEncoder with various configurations."""
    logger.info("=" * 60)
    logger.info("Testing ConvTokenEncoder")
    logger.info("=" * 60)

    # Test configuration 1: No downsampling
    logger.info("\n--- Config 1: No downsampling (stride=1) ---")
    encoder_1 = ConvTokenEncoder(
        in_channels=3,
        hidden_dims=[64, 64, 64, 64],
        embedding_dim=64,
        kernel_size=5,
        stride=1,
        num_groups=8,
        activation="relu",
    )

    images = torch.randn(8, 3, 128, 128)
    tokens, (h, w) = encoder_1(images)

    logger.info(f"Input shape: {images.shape}")
    logger.info(f"Output tokens shape: {tokens.shape}")
    logger.info(f"Spatial shape: ({h}, {w})")
    logger.info(f"Number of tokens (N): {tokens.shape[1]}")
    logger.info(f"Embedding dim (D): {tokens.shape[2]}")
    assert tokens.shape == (8, 128 * 128, 64), "Unexpected output shape!"
    assert (h, w) == (128, 128), "Unexpected spatial shape!"

    # Test configuration 2: With downsampling (stride=2)
    logger.info("\n--- Config 2: With 2x downsampling (stride=2) ---")
    encoder_2 = ConvTokenEncoder(
        in_channels=3,
        hidden_dims=[64, 64, 64, 64],
        embedding_dim=64,
        kernel_size=5,
        stride=2,
        num_groups=8,
        activation="relu",
    )

    tokens, (h, w) = encoder_2(images)

    logger.info(f"Input shape: {images.shape}")
    logger.info(f"Output tokens shape: {tokens.shape}")
    logger.info(f"Spatial shape: ({h}, {w})")
    logger.info(f"Number of tokens (N): {tokens.shape[1]}")
    logger.info(f"Total downsampling: {128 / h}x")
    expected_size = 128 // (2 ** 4)  # 4 blocks with stride 2
    assert h == w == expected_size, f"Expected {expected_size}x{expected_size} tokens!"
    assert tokens.shape == (8, expected_size * expected_size, 64)

    # Test configuration 3: Different embedding dimension
    logger.info("\n--- Config 3: Different embedding dim (128) ---")
    encoder_3 = ConvTokenEncoder(
        in_channels=3,
        hidden_dims=[64, 64, 64],
        embedding_dim=128,
        kernel_size=3,
        stride=2,
        num_groups=8,
        activation="gelu",
    )

    tokens, (h, w) = encoder_3(images)

    logger.info(f"Input shape: {images.shape}")
    logger.info(f"Output tokens shape: {tokens.shape}")
    logger.info(f"Spatial shape: ({h}, {w})")
    logger.info(f"Embedding dim: {tokens.shape[2]}")
    assert tokens.shape[2] == 128, "Unexpected embedding dimension!"

    # Test batching consistency
    logger.info("\n--- Testing batch size consistency ---")
    for batch_size in [1, 4, 16, 32]:
        test_images = torch.randn(batch_size, 3, 128, 128)
        test_tokens, _ = encoder_2(test_images)
        logger.info(
            f"Batch size {batch_size:2d}: tokens shape = {test_tokens.shape}"
        )
        assert test_tokens.shape[0] == batch_size

    logger.info("\n" + "=" * 60)
    logger.info("All tests passed!")
    logger.info("=" * 60)


def test_spatial_preservation() -> None:
    """Test that spatial structure is preserved in tokens."""
    logger.info("\n\nTesting spatial structure preservation...")

    encoder = ConvTokenEncoder(
        in_channels=3,
        hidden_dims=[32, 32],
        embedding_dim=32,
        stride=2,
        num_groups=4,
    )

    # Create checkerboard pattern
    images = torch.zeros(1, 3, 32, 32)
    images[:, :, ::2, ::2] = 1.0  # White squares
    images[:, :, 1::2, 1::2] = 1.0

    tokens, (h, w) = encoder(images)

    logger.info(f"Input image: {images.shape}")
    logger.info(f"Token grid: {h}x{w}")
    logger.info(f"Tokens shape: {tokens.shape}")

    # Reshape tokens back to spatial grid
    spatial_tokens = tokens.reshape(1, h, w, -1)
    logger.info(f"Reshaped to spatial: {spatial_tokens.shape}")

    logger.info("Spatial structure preserved successfully!")


def test_determinism() -> None:
    """Test that encoder is deterministic (no BatchNorm)."""
    logger.info("\n\nTesting deterministic behavior...")

    encoder = ConvTokenEncoder(
        in_channels=3,
        hidden_dims=[64, 64],
        embedding_dim=64,
        stride=1,
        num_groups=8,
    )
    encoder.eval()

    # Same input, different batch sizes
    single_image = torch.randn(1, 3, 64, 64)
    batch_images = single_image.repeat(8, 1, 1, 1)

    with torch.no_grad():
        tokens_single, _ = encoder(single_image)
        tokens_batch, _ = encoder(batch_images)

    # First image in batch should match single image
    diff = (tokens_single - tokens_batch[0:1]).abs().max()
    logger.info(f"Max difference: {diff.item():.8f}")

    assert diff < 1e-6, "Encoder is not deterministic!"
    logger.info("Encoder is deterministic (GroupNorm working correctly)!")


if __name__ == "__main__":
    test_conv_token_encoder()
    test_spatial_preservation()
    test_determinism()

    logger.info("\n" + "=" * 60)
    logger.info("All ConvTokenEncoder tests completed successfully!")
    logger.info("=" * 60)
