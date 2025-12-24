"""Encoder module for feature extraction and spatial tokenization.

This module provides CNN-based encoders that transform images into spatial
token representations suitable for slot-based object decomposition.
"""

import logging
from typing import List, Tuple, Dict, Optional
import math

import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)


class ConvTokenEncoder(nn.Module):
    """Convolutional encoder that produces spatial tokens for slot attention.

    This encoder transforms images into a grid of spatial tokens, where each
    token represents local visual features at a specific spatial location.

    Design Rationale for Slot-Based Factorization:
    ----------------------------------------------
    1. **Spatial Tokenization**: Images are encoded into a grid of N tokens,
       where each token corresponds to a spatial region. This creates a
       structured representation that slot attention can bind to objects.

    2. **Local Features**: Each token captures local visual features (edges,
       textures, parts) within its receptive field, allowing slots to compose
       objects from parts.

    3. **Spatial Correspondence**: The (H_t, W_t) grid structure preserves
       spatial relationships, enabling the decoder to spatially broadcast
       slot representations back to image space with correct positioning.

    4. **Dimensionality Reduction**: Downsampling reduces the number of tokens
       N compared to pixels, making slot attention computationally tractable
       while retaining semantic information.

    5. **Hierarchical Features**: Stacked convolutions build hierarchical
       features, from low-level edges to mid-level object parts, providing
       rich descriptors for object discovery.

    6. **Deterministic Architecture**: GroupNorm (not BatchNorm) ensures
       deterministic behavior regardless of batch size, critical for K-
       conditioned training where K varies per batch.

    The output (B, N, D) format is optimized for transformer-style attention
    mechanisms, where N tokens compete for K slots.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims: List[int] = [64, 64, 64, 64],
        embedding_dim: int = 64,
        kernel_size: int = 5,
        stride: int = 1,
        num_groups: int = 8,
        activation: str = "relu",
    ) -> None:
        """Initialize convolutional token encoder.

        Args:
            in_channels: Number of input channels (3 for RGB images)
            hidden_dims: List of hidden channel dimensions for each conv block.
                        Length determines network depth.
            embedding_dim: Final token embedding dimension (D)
            kernel_size: Convolutional kernel size (default: 5)
            stride: Stride for downsampling (1=no downsample, 2=2x downsample).
                   Higher stride reduces token count N.
            num_groups: Number of groups for GroupNorm. Must divide all
                       hidden_dims and embedding_dim.
            activation: Activation function ('relu', 'gelu', 'elu')

        Example:
            >>> encoder = ConvTokenEncoder(
            ...     in_channels=3,
            ...     hidden_dims=[64, 64, 64, 64],
            ...     embedding_dim=64,
            ...     stride=2,  # 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8
            ... )
            >>> images = torch.randn(32, 3, 128, 128)
            >>> tokens, spatial_shape = encoder(images)
            >>> # tokens: (32, 64, 64) where 64 = 8*8 spatial tokens
            >>> # spatial_shape: (8, 8)
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.embedding_dim = embedding_dim
        self.stride = stride
        self.num_groups = num_groups

        # Build convolutional blocks
        blocks: List[nn.Module] = []
        current_channels = in_channels

        for i, hidden_channels in enumerate(hidden_dims):
            # Validate GroupNorm compatibility
            if hidden_channels % num_groups != 0:
                raise ValueError(
                    f"hidden_dims[{i}]={hidden_channels} must be divisible "
                    f"by num_groups={num_groups}"
                )

            block = self._make_conv_block(
                in_channels=current_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                num_groups=num_groups,
                activation=activation,
            )
            blocks.append(block)
            current_channels = hidden_channels

        self.conv_blocks = nn.Sequential(*blocks)

        # Final 1x1 projection to embedding dimension
        if embedding_dim % num_groups != 0:
            raise ValueError(
                f"embedding_dim={embedding_dim} must be divisible "
                f"by num_groups={num_groups}"
            )

        self.final_proj = nn.Sequential(
            nn.Conv2d(current_channels, embedding_dim, kernel_size=1),
            nn.GroupNorm(num_groups, embedding_dim),
            self._get_activation(activation),
        )

        self._log_architecture()

    def _make_conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        num_groups: int,
        activation: str,
    ) -> nn.Sequential:
        """Create a convolutional block with GroupNorm.

        Block structure: Conv -> GroupNorm -> Activation

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            num_groups: Number of groups for GroupNorm
            activation: Activation function name

        Returns:
            Sequential block
        """
        padding = kernel_size // 2

        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.GroupNorm(num_groups, out_channels),
            self._get_activation(activation),
        )

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name.

        Args:
            name: Activation function name

        Returns:
            Activation module
        """
        activations = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "elu": nn.ELU(inplace=True),
        }
        return activations.get(name.lower(), nn.ReLU(inplace=True))

    def _log_architecture(self) -> None:
        """Log encoder architecture details."""
        num_blocks = len(self.hidden_dims)
        total_stride = self.stride ** num_blocks

        logger.info(
            f"ConvTokenEncoder initialized:\n"
            f"  Input channels: {self.in_channels}\n"
            f"  Hidden dims: {self.hidden_dims}\n"
            f"  Embedding dim: {self.embedding_dim}\n"
            f"  Depth: {num_blocks} blocks\n"
            f"  Stride per block: {self.stride}\n"
            f"  Total downsampling: {total_stride}x\n"
            f"  GroupNorm groups: {self.num_groups}"
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Encode images to spatial tokens.

        Args:
            x: Input images of shape (B, C, H, W)

        Returns:
            tokens: Spatial tokens of shape (B, N, D) where:
                   - B: batch size
                   - N: number of spatial tokens (H_t * W_t)
                   - D: embedding dimension
            spatial_shape: Tuple (H_t, W_t) indicating spatial grid dimensions

        Example:
            >>> encoder = ConvTokenEncoder(embedding_dim=64, stride=2)
            >>> images = torch.randn(8, 3, 128, 128)
            >>> tokens, (h, w) = encoder(images)
            >>> print(tokens.shape)  # (8, 1024, 64) if h=w=32
            >>> print(h, w)  # (32, 32)
        """
        # Pass through convolutional blocks
        features = self.conv_blocks(x)  # (B, C, H', W')

        # Final projection to embedding dimension
        features = self.final_proj(features)  # (B, D, H_t, W_t)

        batch_size, embedding_dim, height, width = features.shape

        # Reshape to token sequence: (B, D, H_t, W_t) -> (B, N, D)
        # where N = H_t * W_t
        tokens = features.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, embedding_dim
        )

        spatial_shape = (height, width)

        return tokens, spatial_shape


class PretrainedVisionEncoder(nn.Module):
    """Pretrained vision encoder for robust visual feature extraction.

    This encoder leverages pretrained vision transformers (ViT) or CNNs to
    extract spatial visual features. Pretrained encoders shift the learning
    burden from "what is an object" (already learned from large-scale data)
    to "how to factorize the scene into layers" (learned via slot attention).

    Why Pretrained Encoders Improve Layer Decomposition:
    ----------------------------------------------------
    1. **Rich Visual Features**: Pretrained on ImageNet/large datasets, these
       encoders already understand edges, textures, object parts, and semantic
       concepts. This provides a strong initialization for slot attention.

    2. **Transfer Learning**: Features learned on millions of images generalize
       better to new domains than randomly initialized CNNs. This is especially
       important for natural images beyond simple synthetic datasets.

    3. **Reduced Learning Burden**: Instead of learning both "what are visual
       features" AND "how to group them", the model only needs to learn the
       grouping (slot assignment). This makes training faster and more stable.

    4. **Better Spatial Awareness**: Vision Transformers (ViT) naturally produce
       spatial tokens with global context, which helps slot attention discover
       semantically meaningful groupings rather than arbitrary regions.

    5. **Generalizes to Arbitrary Images**: Pretrained encoders work on natural
       images, complex scenes, and diverse datasets without dataset-specific
       tuning of the encoder architecture.

    How Spatial Tokens Relate to Image Regions:
    ------------------------------------------
    For Vision Transformers (ViT):
    - Image is divided into patches (e.g., 16x16 pixels each)
    - Each patch becomes one token
    - Token grid: (H/patch_size, W/patch_size)
    - Example: 224x224 image with patch_size=16 → 14x14 = 196 tokens

    Each token represents:
    - Local patch appearance (colors, textures, edges)
    - Global context (via self-attention in ViT)
    - Semantic information (objects, parts, backgrounds)

    Slot attention then binds these tokens to K slots, effectively asking:
    "Which tokens belong to which object/layer?"

    Freeze/Unfreeze Strategy:
    -------------------------
    Stage 1 (Frozen Encoder):
    - freeze(): Set encoder weights to non-trainable
    - Only train slot attention + decoder
    - Fast convergence, stable gradients
    - Good for initial experiments

    Stage 2 (Unfrozen Encoder):
    - unfreeze(): Allow encoder to adapt
    - Fine-tune entire model end-to-end
    - Better performance on specific domains
    - Requires more careful learning rate tuning

    Typical workflow:
    1. Train 50 epochs with frozen encoder (fast)
    2. Unfreeze and train 20 more epochs with lower LR (refinement)
    """

    def __init__(
        self,
        backbone: str = "vit_b_16",
        pretrained: bool = True,
        embed_dim: int = 768,
        freeze: bool = True,
        image_size: int = 224,
    ):
        """Initialize pretrained vision encoder.

        Args:
            backbone: Name of pretrained model from torchvision.models
                     Recommended: 'vit_b_16', 'vit_b_32', 'vit_l_16'
                     Also supports: 'resnet50', 'convnext_base'
            pretrained: If True, load pretrained weights from ImageNet
            embed_dim: Target embedding dimension for slot attention
                      If backbone output != embed_dim, add projection
            freeze: If True, freeze encoder weights (non-trainable)
            image_size: Expected input image size (for ViT position encoding)

        Note:
            - ViT models require specific image sizes (224, 384, etc.)
            - If freeze=True, encoder will be frozen immediately
            - CLS token is discarded, only spatial tokens are used
        """
        super().__init__()

        self.backbone_name = backbone
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.is_vit = "vit" in backbone.lower()

        # Load pretrained backbone
        logger.info(f"Loading pretrained backbone: {backbone}")
        if pretrained:
            weights = "IMAGENET1K_V1"  # Default pretrained weights
        else:
            weights = None

        # Get backbone model
        if self.is_vit:
            # Vision Transformer
            if backbone == "vit_b_16":
                model = models.vit_b_16(weights=weights)
                backbone_dim = 768
                self.patch_size = 16
            elif backbone == "vit_b_32":
                model = models.vit_b_32(weights=weights)
                backbone_dim = 768
                self.patch_size = 32
            elif backbone == "vit_l_16":
                model = models.vit_l_16(weights=weights)
                backbone_dim = 1024
                self.patch_size = 16
            else:
                raise ValueError(f"Unsupported ViT backbone: {backbone}")

            # Remove classification head, keep only feature extraction
            self.backbone = model
            # We'll manually extract features in forward()

        else:
            # CNN-based backbones (ResNet, ConvNeXt, etc.)
            if backbone == "resnet50":
                model = models.resnet50(weights=weights)
                backbone_dim = 2048
                # Remove final FC and avgpool
                self.backbone = nn.Sequential(*list(model.children())[:-2])
                self.is_vit = False
            elif backbone == "convnext_base":
                model = models.convnext_base(weights=weights)
                backbone_dim = 1024
                # Remove classification head
                self.backbone = model.features
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone_dim = backbone_dim

        # Projection layer if backbone_dim != embed_dim
        if backbone_dim != embed_dim:
            self.projection = nn.Linear(backbone_dim, embed_dim)
            logger.info(
                f"Adding projection: {backbone_dim} → {embed_dim}"
            )
        else:
            self.projection = nn.Identity()

        # Freeze if requested
        if freeze:
            self.freeze()

        logger.info(
            f"PretrainedVisionEncoder initialized:\n"
            f"  Backbone: {backbone}\n"
            f"  Pretrained: {pretrained}\n"
            f"  Backbone dim: {backbone_dim}\n"
            f"  Target embed dim: {embed_dim}\n"
            f"  Frozen: {freeze}\n"
            f"  ViT: {self.is_vit}"
        )

    def freeze(self) -> None:
        """Freeze encoder weights (set requires_grad=False).

        Use this for:
        - Initial training phase
        - When you want to train only slot attention + decoder
        - Faster convergence with pretrained features

        After freezing, only projection layer (if any) remains trainable.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Projection can remain trainable
        logger.info("Encoder frozen (backbone weights non-trainable)")

    def unfreeze(self) -> None:
        """Unfreeze encoder weights (set requires_grad=True).

        Use this for:
        - Fine-tuning phase after initial training
        - End-to-end training with lower learning rate
        - Domain adaptation

        Recommendation: Use lower LR (1e-5) when unfrozen.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

        logger.info("Encoder unfrozen (backbone weights trainable)")

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Extract spatial visual tokens from images.

        Args:
            x: Input images of shape (B, 3, H, W)

        Returns:
            tokens: Spatial tokens of shape (B, N, D) where:
                   - B: batch size
                   - N: number of spatial tokens
                   - D: embedding dimension
            spatial_shape: Tuple (H_t, W_t) indicating token grid dimensions

        Example:
            >>> encoder = PretrainedVisionEncoder(backbone='vit_b_16')
            >>> images = torch.randn(4, 3, 224, 224)
            >>> tokens, (h, w) = encoder(images)
            >>> print(tokens.shape)  # (4, 196, 768) for 224x224 input
            >>> print(h, w)  # (14, 14) = 196 tokens
        """
        batch_size = x.shape[0]

        if self.is_vit:
            # Vision Transformer path
            # ViT outputs: (B, num_tokens, dim) where first token is CLS

            # Pass through ViT encoder
            # We need to access internal layers to get patch embeddings
            # Different approach: use the encoder directly

            # Get patch embeddings
            x = self.backbone._process_input(x)  # (B, C, H, W) → patches
            n = x.shape[0]

            # Expand class token
            batch_class_token = self.backbone.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            # Add positional encoding
            x = x + self.backbone.encoder.pos_embedding

            # Pass through transformer blocks
            x = self.backbone.encoder.dropout(x)
            x = self.backbone.encoder.layers(x)
            x = self.backbone.encoder.ln(x)

            # Remove CLS token (first token)
            tokens = x[:, 1:, :]  # (B, N, backbone_dim)

            # Compute spatial grid dimensions
            # For ViT: H_t = W_t = sqrt(N)
            num_tokens = tokens.shape[1]
            spatial_size = int(math.sqrt(num_tokens))
            spatial_shape = (spatial_size, spatial_size)

        else:
            # CNN path (ResNet, ConvNeXt, etc.)
            # Outputs: (B, C, H', W')

            features = self.backbone(x)  # (B, backbone_dim, H', W')
            batch_size, channels, height, width = features.shape

            # Reshape to token format: (B, C, H', W') → (B, N, C)
            tokens = features.permute(0, 2, 3, 1).reshape(
                batch_size, height * width, channels
            )  # (B, H'*W', backbone_dim)

            spatial_shape = (height, width)

        # Project to target embedding dimension
        tokens = self.projection(tokens)  # (B, N, embed_dim)

        return tokens, spatial_shape


# Legacy encoder for backward compatibility
class Encoder(nn.Module):
    """Legacy CNN encoder that outputs feature maps.

    Note: For new code, prefer ConvTokenEncoder which provides spatial
    tokenization and better integration with slot attention.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims: List[int] = [64, 64, 64, 64],
        kernel_size: int = 5,
        output_dim: int = 64,
        activation: str = "relu",
    ) -> None:
        """Initialize encoder.

        Args:
            in_channels: Number of input channels (3 for RGB)
            hidden_dims: List of hidden dimension sizes
            kernel_size: Convolutional kernel size
            output_dim: Output feature dimension
            activation: Activation function name
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        layers: List[nn.Module] = []
        current_dim = in_channels

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(
                    current_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                self._get_activation(activation),
            ])
            current_dim = hidden_dim

        layers.append(
            nn.Conv2d(current_dim, output_dim, kernel_size=1)
        )

        self.network = nn.Sequential(*layers)

        logger.info(
            f"Encoder initialized: {in_channels} -> {hidden_dims} -> {output_dim}"
        )

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "elu": nn.ELU(inplace=True),
        }
        return activations.get(name.lower(), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images of shape (B, C, H, W)

        Returns:
            Feature maps of shape (B, output_dim, H', W')
        """
        return self.network(x)
