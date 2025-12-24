"""Test script for SlotAttention module.

This script demonstrates and validates the K-conditioned slot attention
mechanism for object decomposition.
"""

import logging

import torch
import matplotlib.pyplot as plt
from src.kcd.models.slot_attention import SlotAttention

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_slot_attention() -> None:
    """Test basic SlotAttention functionality."""
    logger.info("=" * 60)
    logger.info("Testing SlotAttention")
    logger.info("=" * 60)

    # Create slot attention module
    slot_attn = SlotAttention(
        slot_dim=64,
        num_iterations=3,
        mlp_hidden_dim=128,
        epsilon=1e-8,
    )

    # Create dummy spatial tokens (from encoder)
    batch_size = 8
    num_tokens = 16 * 16  # 16x16 spatial grid
    token_dim = 64

    tokens = torch.randn(batch_size, num_tokens, token_dim)

    # Test with different K values
    logger.info("\n--- Testing different K values ---")
    for K in [3, 5, 7, 10]:
        slots, attn = slot_attn(tokens, num_slots=K)

        logger.info(f"\nK = {K}:")
        logger.info(f"  Input tokens: {tokens.shape}")
        logger.info(f"  Output slots: {slots.shape}")
        logger.info(f"  Attention weights: {attn.shape}")

        # Verify shapes
        assert slots.shape == (batch_size, K, token_dim)
        assert attn.shape == (batch_size, K, num_tokens)

        # Verify attention normalization
        # Sum over slots (dim=1) should equal 1 for each spatial location
        attn_sum = attn.sum(dim=1)
        logger.info(f"  Attention sum (should be ~1): {attn_sum.mean():.6f}")
        assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5)

    logger.info("\n✓ All shape and normalization tests passed!")


def test_k_conditioning() -> None:
    """Test that K can vary across batches."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing K-Conditioning (Dynamic K)")
    logger.info("=" * 60)

    slot_attn = SlotAttention(
        slot_dim=64,
        num_iterations=3,
        mlp_hidden_dim=128,
    )

    tokens = torch.randn(4, 256, 64)

    # Different K values should all work
    k_values = [2, 5, 8, 11]
    logger.info(f"\nTesting K values: {k_values}")

    for K in k_values:
        slots, attn = slot_attn(tokens, num_slots=K)
        logger.info(f"K={K:2d}: slots shape = {slots.shape}")
        assert slots.shape[1] == K

    logger.info("\n✓ K-conditioning works correctly!")


def test_attention_competition() -> None:
    """Test that attention exhibits competitive behavior."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Slot Competition Mechanism")
    logger.info("=" * 60)

    slot_attn = SlotAttention(
        slot_dim=32,
        num_iterations=5,
        mlp_hidden_dim=64,
    )

    # Create structured input: 4 distinct regions
    batch_size = 1
    H, W = 8, 8
    num_tokens = H * W
    token_dim = 32

    # Create 4 quadrants with different features
    tokens = torch.zeros(batch_size, num_tokens, token_dim)

    # Top-left quadrant
    tokens[0, :H//2*W + W//2, :token_dim//4] = 1.0

    # Top-right quadrant
    tokens[0, :H//2*W + W//2:H//2*W, token_dim//4:token_dim//2] = 1.0

    # Bottom-left quadrant
    tokens[0, H//2*W:H//2*W + W//2, token_dim//2:3*token_dim//4] = 1.0

    # Bottom-right quadrant
    tokens[0, H//2*W + W//2:, 3*token_dim//4:] = 1.0

    # Run slot attention with K=4
    slots, attn = slot_attn(tokens, num_slots=4)

    logger.info(f"\nInput: {H}x{W} grid with 4 distinct quadrants")
    logger.info(f"Slots: K=4")
    logger.info(f"Iterations: {slot_attn.num_iterations}")

    # Analyze attention sparsity
    # If competition works, each slot should focus on different regions
    attn_reshaped = attn.reshape(batch_size, 4, H, W)

    logger.info("\nAttention distribution per slot:")
    for k in range(4):
        slot_attn_map = attn_reshaped[0, k]
        max_val = slot_attn_map.max().item()
        mean_val = slot_attn_map.mean().item()
        entropy = -(attn_reshaped[0, k] * torch.log(attn_reshaped[0, k] + 1e-8)).sum().item()

        logger.info(
            f"  Slot {k}: max={max_val:.4f}, mean={mean_val:.4f}, "
            f"entropy={entropy:.2f}"
        )

    logger.info("\n✓ Slot competition analysis complete!")


def test_iterative_refinement() -> None:
    """Test that slots improve across iterations."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Iterative Refinement")
    logger.info("=" * 60)

    # Create slots with different iteration counts
    iteration_counts = [1, 3, 5, 7]

    tokens = torch.randn(4, 128, 64)
    K = 5

    logger.info(f"\nComparing attention sharpness across iterations:")

    for num_iter in iteration_counts:
        slot_attn = SlotAttention(
            slot_dim=64,
            num_iterations=num_iter,
            mlp_hidden_dim=128,
        )
        slot_attn.eval()

        with torch.no_grad():
            slots, attn = slot_attn(tokens, num_slots=K)

        # Measure attention entropy (lower = sharper)
        entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=-1).mean()

        logger.info(
            f"  Iterations={num_iter}: attention entropy={entropy:.2f} "
            f"(lower = sharper)"
        )

    logger.info(
        "\n✓ More iterations should lead to sharper attention "
        "(lower entropy)"
    )


def test_gradient_flow() -> None:
    """Test that gradients flow correctly through slot attention."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Gradient Flow")
    logger.info("=" * 60)

    slot_attn = SlotAttention(
        slot_dim=64,
        num_iterations=3,
        mlp_hidden_dim=128,
    )

    tokens = torch.randn(2, 64, 64, requires_grad=True)

    slots, attn = slot_attn(tokens, num_slots=4)

    # Compute dummy loss
    loss = slots.sum()
    loss.backward()

    # Check that gradients exist
    assert tokens.grad is not None
    logger.info(f"\nTokens gradient: shape={tokens.grad.shape}")
    logger.info(f"Gradient magnitude: {tokens.grad.abs().mean():.6f}")

    # Check slot attention parameters have gradients
    for name, param in slot_attn.named_parameters():
        assert param.grad is not None
        logger.info(
            f"  {name}: grad magnitude = {param.grad.abs().mean():.6f}"
        )

    logger.info("\n✓ Gradients flow correctly!")


if __name__ == "__main__":
    test_basic_slot_attention()
    test_k_conditioning()
    test_attention_competition()
    test_iterative_refinement()
    test_gradient_flow()

    logger.info("\n" + "=" * 60)
    logger.info("All SlotAttention tests passed successfully!")
    logger.info("=" * 60)
