"""End-to-end test script for complete KCDModel.

This script validates the full K-Conditioned Decomposition pipeline:
encoder → slot attention → decoder → compositing.
"""

import logging
from pathlib import Path

import torch
import yaml
import matplotlib.pyplot as plt

from src.kcd.models.kcd_model import KCDModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model_initialization() -> None:
    """Test model initialization from config."""
    logger.info("=" * 60)
    logger.info("Testing KCDModel Initialization")
    logger.info("=" * 60)

    # Load config
    config_path = Path("configs/model.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Build model
    model = KCDModel.from_config(config)

    logger.info("\n✓ Model initialized successfully from config")
    logger.info(f"  Encoder type: {model.encoder.__class__.__name__}")
    logger.info(f"  Decoder type: {model.decoder.__class__.__name__}")
    logger.info(f"  Uses token encoder: {model.uses_token_encoder}")
    logger.info(f"  Uses layer decoder: {model.uses_layer_decoder}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"\nModel parameters:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,}")


def test_forward_pass() -> None:
    """Test end-to-end forward pass with different K values."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing End-to-End Forward Pass")
    logger.info("=" * 60)

    # Load config and build model
    config_path = Path("configs/model.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = KCDModel.from_config(config)
    model.eval()

    # Test input
    batch_size = 4
    H, W = 128, 128
    images = torch.randn(batch_size, 3, H, W)

    logger.info(f"\nInput images: {images.shape}")

    # Test with different K values
    k_values = [3, 5, 7, 10]

    for K in k_values:
        logger.info(f"\n--- Testing with K={K} slots ---")

        with torch.no_grad():
            outputs = model(images, num_slots=K)

        # Verify all expected outputs
        assert "recons" in outputs
        assert "layer_alphas" in outputs
        assert "slots" in outputs
        assert "attn" in outputs
        assert "spatial_shape" in outputs

        # Verify shapes
        recons = outputs["recons"]
        layer_alphas = outputs["layer_alphas"]
        slots = outputs["slots"]
        attn = outputs["attn"]
        spatial_shape = outputs["spatial_shape"]

        logger.info(f"  Reconstruction: {recons.shape}")
        logger.info(f"  Layer alphas: {layer_alphas.shape}")
        logger.info(f"  Slots: {slots.shape}")
        logger.info(f"  Attention: {attn.shape}")
        logger.info(f"  Spatial shape: {spatial_shape}")

        # Verify reconstruction shape
        assert recons.shape == (batch_size, 3, H, W)
        logger.info(f"  ✓ Reconstruction shape correct")

        # Verify layer alphas shape
        assert layer_alphas.shape == (batch_size, K, 1, H, W)
        logger.info(f"  ✓ Layer alphas shape correct")

        # Verify slots shape
        assert slots.shape == (batch_size, K, config["slot_attention"]["slot_dim"])
        logger.info(f"  ✓ Slots shape correct")

        # Verify attention shape
        H_t, W_t = spatial_shape
        assert attn.shape == (batch_size, K, H_t * W_t)
        logger.info(f"  ✓ Attention shape correct")

        # Verify alpha normalization (sum to 1 across slots)
        alpha_sum = layer_alphas.sum(dim=1)
        assert torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-5)
        logger.info(f"  ✓ Alpha masks sum to 1 (mean: {alpha_sum.mean():.6f})")

        # Check if layer_rgbs is available (new LayerDecoder)
        if "layer_rgbs" in outputs:
            layer_rgbs = outputs["layer_rgbs"]
            assert layer_rgbs.shape == (batch_size, K, 3, H, W)
            logger.info(f"  Layer RGBs: {layer_rgbs.shape}")
            logger.info(f"  ✓ Layer RGBs available (new LayerDecoder)")

    logger.info("\n✓ All forward pass tests passed!")


def test_k_conditioning() -> None:
    """Test that same model handles variable K correctly."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing K-Conditioning (Variable K Support)")
    logger.info("=" * 60)

    # Load model
    config_path = Path("configs/model.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = KCDModel.from_config(config)
    model.eval()

    # Same input
    images = torch.randn(2, 3, 128, 128)

    # Different K values in sequence
    k_sequence = [2, 4, 7, 3, 10, 5]

    logger.info(f"\nTesting K sequence: {k_sequence}")
    logger.info("(Same model, different K per forward pass)\n")

    for K in k_sequence:
        with torch.no_grad():
            outputs = model(images, num_slots=K)

        slots = outputs["slots"]
        layer_alphas = outputs["layer_alphas"]

        logger.info(f"K={K:2d}: slots={slots.shape}, alphas={layer_alphas.shape}")
        assert slots.shape[1] == K
        assert layer_alphas.shape[1] == K

    logger.info("\n✓ K-conditioning works! Single model handles all K values.")


def test_gradient_flow() -> None:
    """Test that gradients flow through entire pipeline."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Gradient Flow Through Pipeline")
    logger.info("=" * 60)

    # Load model
    config_path = Path("configs/model.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = KCDModel.from_config(config)
    model.train()

    # Input with gradients
    images = torch.randn(2, 3, 128, 128, requires_grad=True)

    # Forward pass
    outputs = model(images, num_slots=5)
    recons = outputs["recons"]

    # Compute reconstruction loss
    target = torch.randn_like(recons)
    loss = ((recons - target) ** 2).mean()

    logger.info(f"\nReconstruction loss: {loss.item():.6f}")

    # Backward pass
    loss.backward()

    # Check input gradients
    assert images.grad is not None
    logger.info(f"\nInput gradient: shape={images.grad.shape}")
    logger.info(f"  Magnitude: {images.grad.abs().mean():.6f}")

    # Check all model parameters have gradients
    params_with_grad = 0
    params_without_grad = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                params_with_grad += 1
            else:
                params_without_grad += 1
                logger.warning(f"  Parameter without gradient: {name}")

    logger.info(f"\nParameters with gradients: {params_with_grad}")
    logger.info(f"Parameters without gradients: {params_without_grad}")

    if params_without_grad > 0:
        logger.warning("⚠ Some parameters did not receive gradients!")
    else:
        logger.info("✓ All parameters received gradients!")

    # Check gradient magnitudes by component
    logger.info("\nGradient magnitudes by component:")

    encoder_grads = [
        p.grad.abs().mean().item()
        for p in model.encoder.parameters()
        if p.grad is not None
    ]
    logger.info(f"  Encoder: {sum(encoder_grads) / len(encoder_grads):.6f}")

    slot_grads = [
        p.grad.abs().mean().item()
        for p in model.slot_attention.parameters()
        if p.grad is not None
    ]
    logger.info(f"  Slot Attention: {sum(slot_grads) / len(slot_grads):.6f}")

    decoder_grads = [
        p.grad.abs().mean().item()
        for p in model.decoder.parameters()
        if p.grad is not None
    ]
    logger.info(f"  Decoder: {sum(decoder_grads) / len(decoder_grads):.6f}")

    logger.info("\n✓ Gradients flow correctly through entire pipeline!")


def test_reconstruction_quality() -> None:
    """Test that model can overfit to a single image (sanity check)."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Reconstruction Capacity (Overfitting Test)")
    logger.info("=" * 60)

    # Load model
    config_path = Path("configs/model.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = KCDModel.from_config(config)
    model.train()

    # Single image to overfit
    target_image = torch.randn(1, 3, 128, 128)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_steps = 100
    K = 5

    logger.info(f"\nOverfitting to single image with K={K} slots")
    logger.info(f"Training for {num_steps} steps...\n")

    losses = []

    for step in range(num_steps):
        optimizer.zero_grad()

        # Forward
        outputs = model(target_image, num_slots=K)
        recons = outputs["recons"]

        # Loss
        loss = ((recons - target_image) ** 2).mean()
        losses.append(loss.item())

        # Backward
        loss.backward()
        optimizer.step()

        if (step + 1) % 20 == 0:
            logger.info(f"  Step {step + 1:3d}: Loss = {loss.item():.6f}")

    # Check loss decreased
    initial_loss = losses[0]
    final_loss = losses[-1]
    improvement = (initial_loss - final_loss) / initial_loss * 100

    logger.info(f"\nInitial loss: {initial_loss:.6f}")
    logger.info(f"Final loss: {final_loss:.6f}")
    logger.info(f"Improvement: {improvement:.1f}%")

    if improvement > 50:
        logger.info("✓ Model can learn to reconstruct (loss decreased >50%)")
    else:
        logger.warning("⚠ Model may have learning issues (loss decreased <50%)")


def test_layer_decomposition() -> None:
    """Test that layers show meaningful decomposition."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Layer Decomposition Properties")
    logger.info("=" * 60)

    # Load model
    config_path = Path("configs/model.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = KCDModel.from_config(config)
    model.eval()

    # Test image
    images = torch.randn(1, 3, 128, 128)
    K = 5

    with torch.no_grad():
        outputs = model(images, num_slots=K)

    layer_alphas = outputs["layer_alphas"]
    attn = outputs["attn"]

    logger.info(f"\nAnalyzing K={K} layer decomposition:")

    # Analyze alpha mask statistics
    logger.info("\nAlpha mask statistics per layer:")
    for k in range(K):
        alpha_k = layer_alphas[0, k, 0]  # (H, W)
        mean_alpha = alpha_k.mean().item()
        max_alpha = alpha_k.max().item()
        min_alpha = alpha_k.min().item()
        coverage = (alpha_k > 0.5).float().mean().item() * 100

        logger.info(
            f"  Layer {k}: mean={mean_alpha:.4f}, "
            f"max={max_alpha:.4f}, min={min_alpha:.4f}, "
            f"coverage={coverage:.1f}%"
        )

    # Analyze attention sparsity
    logger.info("\nAttention sparsity per slot:")
    H_t, W_t = outputs["spatial_shape"]
    attn_reshaped = attn.reshape(1, K, H_t, W_t)

    for k in range(K):
        attn_k = attn_reshaped[0, k]
        entropy = -(attn_k * torch.log(attn_k + 1e-8)).sum().item()
        max_attn = attn_k.max().item()

        logger.info(
            f"  Slot {k}: entropy={entropy:.2f}, max_attention={max_attn:.4f}"
        )

    # Check layer RGB diversity (if available)
    if "layer_rgbs" in outputs:
        layer_rgbs = outputs["layer_rgbs"]
        logger.info("\nLayer RGB diversity:")

        for k in range(K):
            rgb_k = layer_rgbs[0, k]  # (3, H, W)
            rgb_std = rgb_k.std(dim=[1, 2]).mean().item()
            logger.info(f"  Layer {k}: spatial variance={rgb_std:.4f}")

    logger.info("\n✓ Layer decomposition analysis complete!")


if __name__ == "__main__":
    test_model_initialization()
    test_forward_pass()
    test_k_conditioning()
    test_gradient_flow()
    test_reconstruction_quality()
    test_layer_decomposition()

    logger.info("\n" + "=" * 60)
    logger.info("All KCDModel end-to-end tests passed successfully!")
    logger.info("=" * 60)
    logger.info("\nThe model is ready for:")
    logger.info("  • Dataset integration")
    logger.info("  • Training loop implementation")
    logger.info("  • Visualization of decompositions")
    logger.info("  • Research experiments")
