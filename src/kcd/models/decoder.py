"""Decoder module for reconstructing images from slots.

This module implements spatial broadcast decoders that transform slot
representations into image layers with alpha compositing.
"""

import logging
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LayerDecoder(nn.Module):
    """Layer-wise decoder with alpha compositing for image reconstruction.

    This decoder transforms K slot representations into K image layers, each
    with RGB appearance and an alpha mask. The layers are then composed via
    alpha blending to produce the final reconstruction.

    Why Shared Decoding is Important:
    ---------------------------------
    The decoder uses the SAME network weights for ALL slots. This design is
    critical for several reasons:

    1. **Parameter Efficiency**: With K slots, using separate decoders would
       require K× the parameters. Shared weights keep the model size constant
       regardless of K.

    2. **K-Conditioned Training**: During training, K varies per batch. Shared
       weights allow the model to handle any K without architectural changes.

    3. **Slot Permutation Invariance**: Shared weights ensure all slots are
       treated equally. There's no "first slot" or "last slot" bias.

    4. **Generalization**: The decoder learns a universal "slot → layer" mapping
       rather than specialized decoders for specific object positions.

    5. **Object Discovery**: Forces slots to differentiate via their content
       (D-dimensional vectors), not via decoder specialization.

    Alpha Compositing Explanation:
    ------------------------------
    Each slot k produces:
    - Appearance A_k: RGB image (3, H, W) - what the layer looks like
    - Alpha α_k: Mask (1, H, W) - where the layer is visible

    The alpha masks are normalized across slots using softmax:
        α'_k = exp(α_k) / Σ_j exp(α_j)

    This ensures: Σ_k α'_k(x,y) = 1 at every pixel (x,y)

    Final reconstruction via weighted sum:
        I(x,y) = Σ_k α'_k(x,y) · A_k(x,y)

    Why This Works:
    - Each pixel in the output is a mixture of slot appearances
    - Alpha masks determine contribution of each slot
    - Softmax competition: slots compete for pixel ownership
    - Background can be represented as a slot with uniform appearance

    The combination of slot attention (spatial feature binding) + alpha
    compositing (layer blending) enables unsupervised decomposition.

    Spatial Broadcast Architecture:
    -------------------------------
    Slots are abstract D-dimensional vectors. To decode them spatially:

    1. **Broadcast**: Repeat slot vector to all spatial locations
       (K, D) → (K, D, H, W)

    2. **Position Encoding**: Add (x, y) coordinates at each location
       (K, D, H, W) + (2, H, W) → (K, D+2, H, W)

    3. **Convolutional Decoding**: CNN transforms position-aware slots
       (K, D+2, H, W) → (K, 4, H, W)  [RGB + alpha]

    This allows the decoder to produce spatially-varying outputs while
    maintaining translational equivariance.
    """

    def __init__(
        self,
        slot_dim: int = 64,
        hidden_dims: List[int] = [64, 64, 64, 64],
        output_size: Tuple[int, int] = (128, 128),
        kernel_size: int = 5,
        activation: str = "relu",
        use_layer_norm: bool = False,
    ) -> None:
        """Initialize layer-wise decoder with shared weights.

        Args:
            slot_dim: Dimension of slot representations (D)
            hidden_dims: List of hidden channel dimensions for conv layers
            output_size: Target output image size (H, W)
            kernel_size: Convolutional kernel size (default: 5)
            activation: Activation function ('relu', 'gelu', 'elu')
            use_layer_norm: If True, add LayerNorm after position encoding

        Note:
            - Same decoder weights are applied to ALL K slots
            - Output is always 4 channels: RGB (3) + alpha (1)
            - Alpha is processed with softmax across slots
        """
        super().__init__()

        self.slot_dim = slot_dim
        self.hidden_dims = hidden_dims
        self.output_size = output_size
        self.kernel_size = kernel_size

        # Build 2D position encoding (learnable or fixed)
        self.position_encoding = self._build_position_encoding(output_size)

        # Optional normalization of position-encoded input
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.input_norm = nn.LayerNorm(slot_dim + 2)
        else:
            self.input_norm = nn.Identity()

        # Build shared convolutional decoder
        # Input: slot (D) + position (2) = D+2 channels
        # Output: RGB (3) + alpha_logit (1) = 4 channels
        input_channels = slot_dim + 2

        layers: List[nn.Module] = []
        current_channels = input_channels

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(
                    current_channels,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                self._get_activation(activation),
            ])
            current_channels = hidden_dim

        # Final 1x1 conv to RGB + alpha
        layers.append(
            nn.Conv2d(current_channels, 4, kernel_size=1)
        )

        self.decoder_network = nn.Sequential(*layers)

        logger.info(
            f"LayerDecoder initialized:\n"
            f"  Slot dim: {slot_dim}\n"
            f"  Hidden dims: {hidden_dims}\n"
            f"  Output size: {output_size}\n"
            f"  Output channels: 4 (RGB + alpha)\n"
            f"  Shared weights: Yes (all {slot_dim}-dim slots use same decoder)"
        )

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "elu": nn.ELU(inplace=True),
        }
        return activations.get(name.lower(), nn.ReLU(inplace=True))

    def _build_position_encoding(
        self,
        size: Tuple[int, int],
    ) -> nn.Parameter:
        """Build 2D position encoding grid.

        Creates a fixed grid of (x, y) coordinates in [-1, 1] range.
        This provides spatial information to the decoder.

        Args:
            size: Output spatial size (H, W)

        Returns:
            Position encoding of shape (2, H, W)
        """
        h, w = size

        # Create normalized coordinate grids in [-1, 1]
        y_grid = torch.linspace(-1, 1, h).view(h, 1).expand(h, w)
        x_grid = torch.linspace(-1, 1, w).view(1, w).expand(h, w)

        # Stack to (2, H, W)
        position_encoding = torch.stack([x_grid, y_grid], dim=0)

        # Register as buffer (not trainable, but moves with model)
        return nn.Parameter(position_encoding, requires_grad=False)

    def forward(
        self, slots: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode slots to layered reconstruction.

        This is the core decoding pipeline:
        1. Broadcast each slot to spatial grid
        2. Add position encoding
        3. Apply shared CNN decoder
        4. Extract RGB and alpha per slot
        5. Normalize alphas with softmax
        6. Composite layers with alpha blending

        Args:
            slots: Slot representations of shape (B, K, D)

        Returns:
            recons: Reconstructed image of shape (B, 3, H, W)
                   Computed as Σ_k (alpha_k * rgb_k)
            layer_rgbs: Per-slot RGB appearances of shape (B, K, 3, H, W)
                       What each layer looks like
            layer_alphas: Per-slot alpha masks of shape (B, K, 1, H, W)
                         Where each layer is visible (sums to 1 across K)

        Example:
            >>> decoder = LayerDecoder(slot_dim=64, output_size=(128, 128))
            >>> slots = torch.randn(4, 5, 64)  # 4 images, 5 slots
            >>> recons, rgbs, alphas = decoder(slots)
            >>> recons.shape
            torch.Size([4, 3, 128, 128])
            >>> rgbs.shape
            torch.Size([4, 5, 3, 128, 128])
            >>> alphas.shape
            torch.Size([4, 5, 1, 128, 128])
            >>> alphas.sum(dim=1)  # Should be all ones
            torch.Size([4, 1, 128, 128])
        """
        batch_size, num_slots, slot_dim = slots.shape
        h, w = self.output_size

        # --- STEP 1: Spatial Broadcast ---
        # Expand each slot from (D,) to (D, H, W) by repeating
        # This creates a constant "slot field" across all locations
        slots_broadcast = slots.reshape(batch_size * num_slots, slot_dim, 1, 1)
        slots_broadcast = slots_broadcast.expand(-1, -1, h, w)
        # Shape: (B*K, D, H, W)

        # --- STEP 2: Add Position Encoding ---
        # Concatenate (x, y) coordinates to give spatial awareness
        position_encoding = self.position_encoding.to(slots.device)
        position_encoding = position_encoding.unsqueeze(0).expand(
            batch_size * num_slots, -1, -1, -1
        )
        # Shape: (B*K, 2, H, W)

        # Concatenate slot + position
        decoder_input = torch.cat([slots_broadcast, position_encoding], dim=1)
        # Shape: (B*K, D+2, H, W)

        # Optional normalization
        if self.use_layer_norm:
            # Normalize over channel dimension
            decoder_input = decoder_input.permute(0, 2, 3, 1)  # (B*K, H, W, D+2)
            decoder_input = self.input_norm(decoder_input)
            decoder_input = decoder_input.permute(0, 3, 1, 2)  # (B*K, D+2, H, W)

        # --- STEP 3: Shared CNN Decoding ---
        # Same weights applied to ALL slots
        # Each slot is decoded independently but with identical architecture
        decoder_output = self.decoder_network(decoder_input)
        # Shape: (B*K, 4, H, W) where 4 = RGB (3) + alpha_logit (1)

        # Reshape to separate batch and slots
        decoder_output = decoder_output.reshape(batch_size, num_slots, 4, h, w)
        # Shape: (B, K, 4, H, W)

        # --- STEP 4: Extract RGB and Alpha ---
        # Split channels into appearance and alpha
        layer_rgbs = decoder_output[:, :, :3, :, :]  # (B, K, 3, H, W)
        alpha_logits = decoder_output[:, :, 3:4, :, :]  # (B, K, 1, H, W)

        # --- STEP 5: Competitive Alpha Normalization ---
        # Apply softmax across slots (dim=1)
        # This ensures Σ_k alpha_k(x, y) = 1 at every pixel
        # Slots compete for pixel ownership via alpha masks
        layer_alphas = F.softmax(alpha_logits, dim=1)  # (B, K, 1, H, W)

        # After softmax:
        # - layer_alphas.sum(dim=1) = all ones
        # - Each pixel is "owned" by a distribution over slots
        # - Sharp alphas = clear segmentation, soft alphas = blending

        # --- STEP 6: Alpha Compositing ---
        # Weighted sum of layer appearances according to alpha masks
        # This is the differentiable rendering step
        recons = torch.sum(layer_rgbs * layer_alphas, dim=1)
        # Shape: (B, 3, H, W)

        # Reconstruction formula: I(x,y) = Σ_k α_k(x,y) · RGB_k(x,y)
        # This is a differentiable mixture model
        # Gradients flow back through alphas AND rgbs to slots

        return recons, layer_rgbs, layer_alphas


# Legacy decoder for backward compatibility
class Decoder(nn.Module):
    """Spatial broadcast decoder for slot-based reconstruction.

    Note: For new code, prefer LayerDecoder which provides explicit
    layer separation and more detailed outputs (per-layer RGB + alpha).
    """

    def __init__(
        self,
        slot_dim: int = 64,
        hidden_dims: List[int] = [64, 64, 64, 64],
        output_channels: int = 4,
        spatial_broadcast_size: Tuple[int, int] = (128, 128),
        activation: str = "relu",
    ) -> None:
        """Initialize decoder.

        Args:
            slot_dim: Dimension of slot representations
            hidden_dims: List of hidden dimensions
            output_channels: Number of output channels (RGB + alpha)
            spatial_broadcast_size: Size to broadcast slots to (H, W)
            activation: Activation function name
        """
        super().__init__()

        self.slot_dim = slot_dim
        self.output_channels = output_channels
        self.spatial_broadcast_size = spatial_broadcast_size

        self.position_encoding = self._build_position_encoding(
            spatial_broadcast_size
        )

        input_dim = slot_dim + 2

        layers: List[nn.Module] = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(
                    current_dim,
                    hidden_dim,
                    kernel_size=5,
                    padding=2,
                ),
                self._get_activation(activation),
            ])
            current_dim = hidden_dim

        layers.append(
            nn.Conv2d(current_dim, output_channels, kernel_size=1)
        )

        self.network = nn.Sequential(*layers)

        logger.info(
            f"Decoder initialized: {slot_dim} -> {hidden_dims} -> {output_channels}"
        )

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "elu": nn.ELU(inplace=True),
        }
        return activations.get(name.lower(), nn.ReLU(inplace=True))

    def _build_position_encoding(
        self,
        size: Tuple[int, int],
    ) -> torch.Tensor:
        """Build 2D position encoding.

        Args:
            size: Spatial size (H, W)

        Returns:
            Position encoding of shape (2, H, W)
        """
        h, w = size
        y_grid = torch.linspace(-1, 1, h).view(h, 1).expand(h, w)
        x_grid = torch.linspace(-1, 1, w).view(1, w).expand(h, w)
        position_encoding = torch.stack([x_grid, y_grid], dim=0)
        return position_encoding

    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            slots: Slot representations of shape (B, K, D)

        Returns:
            recons: Reconstructed images of shape (B, 3, H, W)
            masks: Alpha masks of shape (B, K, 1, H, W)
        """
        batch_size, num_slots, _ = slots.shape
        h, w = self.spatial_broadcast_size

        position_encoding = self.position_encoding.to(slots.device)
        position_encoding = position_encoding.unsqueeze(0).expand(
            batch_size * num_slots, -1, -1, -1
        )

        slots = slots.reshape(batch_size * num_slots, self.slot_dim, 1, 1)
        slots = slots.expand(-1, -1, h, w)

        decoder_input = torch.cat([slots, position_encoding], dim=1)

        decoder_output = self.network(decoder_input)

        decoder_output = decoder_output.reshape(
            batch_size, num_slots, self.output_channels, h, w
        )

        rgb = decoder_output[:, :, :3, :, :]
        alpha_logits = decoder_output[:, :, 3:4, :, :]

        masks = torch.softmax(alpha_logits, dim=1)

        recons = torch.sum(rgb * masks, dim=1)

        return recons, masks
