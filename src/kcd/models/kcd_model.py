"""K-Conditioned Decomposition model.

This module implements the complete KCD architecture for unsupervised
image decomposition into K semantic layers.
"""

import logging
from typing import Dict, Tuple, Optional, Any

import torch
import torch.nn as nn

from .encoder import ConvTokenEncoder, Encoder, PretrainedVisionEncoder
from .slot_attention import SlotAttention
from .decoder import LayerDecoder, Decoder

logger = logging.getLogger(__name__)


class KCDModel(nn.Module):
    """K-Conditioned Decomposition for unsupervised image layer decomposition.

    KCD learns to decompose images into K semantic layers without supervision
    by combining three key components:
    1. Spatial tokenization (encoder)
    2. Competitive binding (slot attention)
    3. Layer-wise rendering (decoder with alpha compositing)

    Mathematical Formulation:
    ------------------------
    Given an input image I ∈ R^(3×H×W) and number of slots K:

    1. **Encoding**: Extract spatial feature tokens
       X = Encoder(I) ∈ R^(B×N×D)
       where N = H_t × W_t is the number of spatial tokens

    2. **Slot Attention**: Bind features to K object-centric slots
       S = SlotAttention(X, K) ∈ R^(B×K×D)

       This is an iterative process with T refinement steps:
       For t = 1, ..., T:
           Q = Linear_q(LayerNorm(S^t))
           K = Linear_k(LayerNorm(X))
           V = Linear_v(LayerNorm(X))

           A_logits = (Q @ K^T) / sqrt(D)
           A = softmax(A_logits, dim=-1)
           A = A / sum(A, dim=slots)  # Competitive normalization

           S^(t+1) = GRU(A @ V, S^t) + MLP(S^(t+1))

    3. **Decoding**: Render each slot into RGB appearance + alpha mask
       For each slot k:
           s_k = S[:, k, :]  # Shape: (B, D)

           # Spatial broadcast
           s_k_spatial = broadcast(s_k, (H, W))  # (B, D, H, W)

           # Add position encoding
           input = concat([s_k_spatial, pos_enc])  # (B, D+2, H, W)

           # Decode with shared weights
           [RGB_k, α_k] = Decoder(input)  # (B, 3, H, W), (B, 1, H, W)

       # Alpha normalization across slots
       α'_k = softmax(α_k, dim=slots)

    4. **Compositing**: Blend layers via alpha compositing
       Î = Σ_{k=1}^K α'_k ⊙ RGB_k

    Training Objective:
    ------------------
    L = ||I - Î||² + λ * R(S)

    where:
    - First term: MSE reconstruction loss
    - R(S): Optional regularization on slots
    - K is sampled uniformly from [K_min, K_max] during training

    K-Conditioning Benefits:
    -----------------------
    1. **Generalization**: Model learns decomposition at multiple granularities
    2. **Flexibility**: Single model handles variable object counts
    3. **Regularization**: Prevents overfitting to specific K values
    4. **Efficiency**: No need for separate models per K

    The key insight is that competitive normalization in both slot attention
    and alpha compositing creates pressure for slots to specialize on
    distinct, semantically meaningful image regions.
    """

    def __init__(
        self,
        encoder: nn.Module,
        slot_attention: SlotAttention,
        decoder: nn.Module,
    ) -> None:
        """Initialize K-Conditioned Decomposition model.

        Args:
            encoder: Spatial feature encoder (ConvTokenEncoder or legacy Encoder)
            slot_attention: Slot attention module for competitive binding
            decoder: Layer decoder (LayerDecoder or legacy Decoder)

        Note:
            Recommended configuration:
            - encoder: ConvTokenEncoder with GroupNorm
            - slot_attention: SlotAttention with 3-7 iterations
            - decoder: LayerDecoder with shared weights
        """
        super().__init__()

        self.encoder = encoder
        self.slot_attention = slot_attention
        self.decoder = decoder

        # Determine encoder type for appropriate handling
        self.uses_token_encoder = isinstance(encoder, (ConvTokenEncoder, PretrainedVisionEncoder))
        self.uses_layer_decoder = isinstance(decoder, LayerDecoder)

        logger.info(
            f"KCDModel initialized:\n"
            f"  Encoder: {encoder.__class__.__name__}\n"
            f"  Slot Attention: {slot_attention.__class__.__name__}\n"
            f"  Decoder: {decoder.__class__.__name__}\n"
            f"  Token encoder: {self.uses_token_encoder}\n"
            f"  Layer decoder: {self.uses_layer_decoder}"
        )

    def forward(
        self,
        images: torch.Tensor,
        num_slots: int,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: Image → Tokens → Slots → Layers → Reconstruction.

        This implements the complete KCD pipeline:
        1. Encode image to spatial tokens
        2. Apply slot attention with K slots
        3. Decode slots to layers (RGB + alpha)
        4. Composite layers via alpha blending

        Args:
            images: Input RGB images of shape (B, 3, H, W)
            num_slots: Number of slots K to decompose into (dynamic per batch)

        Returns:
            Dictionary with keys:
                - recons: Final reconstructed image (B, 3, H, W)
                         Computed as Σ_k (α_k ⊙ RGB_k)
                - layer_rgbs: Per-slot RGB appearances (B, K, 3, H, W)
                            What each discovered layer looks like
                - layer_alphas: Per-slot alpha masks (B, K, 1, H, W)
                              Where each layer is visible (sum to 1)
                - slots: Slot representations (B, K, D)
                        Abstract object-centric features
                - attn: Slot attention weights (B, K, N)
                       Which spatial tokens each slot attended to
                - spatial_shape: (H_t, W_t) token grid shape (if token encoder)

        Example:
            >>> model = KCDModel(encoder, slot_attn, decoder)
            >>> images = torch.randn(8, 3, 128, 128)
            >>>
            >>> # Variable K during training
            >>> outputs = model(images, num_slots=5)
            >>> recons = outputs['recons']  # (8, 3, 128, 128)
            >>> layers = outputs['layer_rgbs']  # (8, 5, 3, 128, 128)
            >>> masks = outputs['layer_alphas']  # (8, 5, 1, 128, 128)
            >>>
            >>> # Different K for same model
            >>> outputs = model(images, num_slots=7)  # Works!
        """
        batch_size = images.shape[0]

        # --- STEP 1: Encoding ---
        # Transform image to spatial feature tokens
        if self.uses_token_encoder:
            # ConvTokenEncoder returns (tokens, spatial_shape)
            tokens, spatial_shape = self.encoder(images)
            # tokens: (B, N, D) where N = H_t * W_t
        else:
            # Legacy Encoder returns feature maps
            features = self.encoder(images)  # (B, D, H', W')

            # Reshape to token format
            batch_feat, channels, height, width = features.shape
            tokens = features.permute(0, 2, 3, 1).reshape(
                batch_feat, height * width, channels
            )  # (B, N, D)
            spatial_shape = (height, width)

        # --- STEP 2: Slot Attention ---
        # Competitively bind tokens to K slots
        slots, attn = self.slot_attention(tokens, num_slots)
        # slots: (B, K, D)
        # attn: (B, K, N) - attention weights showing binding

        # --- STEP 3: Decoding + Compositing ---
        # Decode slots to image layers and composite
        if self.uses_layer_decoder:
            # LayerDecoder returns (recons, layer_rgbs, layer_alphas)
            recons, layer_rgbs, layer_alphas = self.decoder(slots)
        else:
            # Legacy Decoder returns (recons, masks)
            recons, layer_alphas = self.decoder(slots)
            # Legacy decoder doesn't return separate RGB layers
            layer_rgbs = None  # Not available in legacy mode

        # --- Build output dictionary ---
        outputs = {
            "recons": recons,  # (B, 3, H, W) - final reconstruction
            "layer_alphas": layer_alphas,  # (B, K, 1, H, W) - where layers are
            "slots": slots,  # (B, K, D) - abstract representations
            "attn": attn,  # (B, K, N) - spatial attention weights
            "spatial_shape": spatial_shape,  # (H_t, W_t) - token grid
        }

        # Add layer RGBs if available (new LayerDecoder)
        if layer_rgbs is not None:
            outputs["layer_rgbs"] = layer_rgbs  # (B, K, 3, H, W)

        return outputs

    @staticmethod
    def from_config(config: Dict[str, Any]) -> "KCDModel":
        """Create KCD model from configuration dictionary.

        Supports both new and legacy encoder/decoder types based on config.

        Args:
            config: Configuration dictionary with keys:
                   - encoder: dict with 'type' and parameters
                   - slot_attention: dict with slot attention params
                   - decoder: dict with 'type' and parameters

        Returns:
            Fully initialized KCDModel

        Example:
            >>> config = {
            ...     'encoder': {'type': 'conv_token', 'embedding_dim': 64, ...},
            ...     'slot_attention': {'slot_dim': 64, 'num_iterations': 3, ...},
            ...     'decoder': {'type': 'layer', 'slot_dim': 64, ...}
            ... }
            >>> model = KCDModel.from_config(config)
        """
        # --- Build Encoder ---
        encoder_config = config["encoder"]
        encoder_type = encoder_config.get("type", "cnn")  # Default to legacy

        if encoder_type == "pretrained":
            # Pretrained vision encoder (ViT, ResNet, etc.)
            encoder = PretrainedVisionEncoder(
                backbone=encoder_config.get("backbone", "vit_b_16"),
                pretrained=encoder_config.get("pretrained", True),
                embed_dim=encoder_config.get("embed_dim", 64),
                freeze=encoder_config.get("freeze", True),
                image_size=encoder_config.get("image_size", 224),
            )
        elif encoder_type == "conv_token":
            # New ConvTokenEncoder
            encoder = ConvTokenEncoder(
                in_channels=encoder_config["in_channels"],
                hidden_dims=encoder_config["hidden_dims"],
                embedding_dim=encoder_config["embedding_dim"],
                kernel_size=encoder_config["kernel_size"],
                stride=encoder_config.get("stride", 1),
                num_groups=encoder_config.get("num_groups", 8),
                activation=encoder_config["activation"],
            )
        else:
            # Legacy Encoder
            encoder = Encoder(
                in_channels=encoder_config["in_channels"],
                hidden_dims=encoder_config["hidden_dims"],
                kernel_size=encoder_config["kernel_size"],
                output_dim=encoder_config.get(
                    "output_dim", encoder_config.get("embedding_dim", 64)
                ),
                activation=encoder_config["activation"],
            )

        # --- Build Slot Attention ---
        slot_config = config["slot_attention"]
        slot_attention = SlotAttention(
            slot_dim=slot_config["slot_dim"],
            num_iterations=slot_config["num_iterations"],
            mlp_hidden_dim=slot_config["mlp_hidden_dim"],
            epsilon=slot_config["epsilon"],
            use_implicit_diff=slot_config.get("use_implicit_diff", False),
        )

        # --- Build Decoder ---
        decoder_config = config["decoder"]
        decoder_type = decoder_config.get("type", "broadcast")  # Default to legacy

        if decoder_type == "layer":
            # New LayerDecoder
            decoder = LayerDecoder(
                slot_dim=decoder_config["slot_dim"],
                hidden_dims=decoder_config["hidden_dims"],
                output_size=tuple(decoder_config["output_size"]),
                kernel_size=decoder_config.get("kernel_size", 5),
                activation=decoder_config["activation"],
                use_layer_norm=decoder_config.get("use_layer_norm", False),
            )
        else:
            # Legacy Decoder
            decoder = Decoder(
                slot_dim=decoder_config["slot_dim"],
                hidden_dims=decoder_config["hidden_dims"],
                output_channels=decoder_config.get("output_channels", 4),
                spatial_broadcast_size=tuple(
                    decoder_config.get(
                        "spatial_broadcast_size", decoder_config.get("output_size", [128, 128])
                    )
                ),
                activation=decoder_config["activation"],
            )

        return KCDModel(encoder, slot_attention, decoder)
