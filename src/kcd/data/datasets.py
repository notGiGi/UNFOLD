"""Dataset implementations for K-Conditioned Decomposition.

This module provides research-grade image datasets optimized for unsupervised
image decomposition tasks. The datasets are designed to support object-centric
learning where models must discover and decompose visual scenes without
supervision.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)


# ImageNet normalization statistics for transfer learning
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class DatasetConfig:
    """Configuration for dataset loading.

    Attributes:
        dataset_name: Name identifier for the dataset
        image_size: Target image size as (height, width)
        batch_size: Number of images per batch
        num_workers: Number of parallel data loading workers
        train_split: Fraction of data to use for training (0.0-1.0)
        crop_type: Type of cropping ('center', 'random', or None)
        use_normalization: Whether to normalize with ImageNet stats
    """
    dataset_name: str
    image_size: Tuple[int, int]
    batch_size: int
    num_workers: int
    train_split: float
    crop_type: Optional[str] = None
    use_normalization: bool = False


class ImageFolderDataset(Dataset):
    """Image folder dataset for unsupervised image decomposition research.

    This dataset is specifically designed for object-centric representation
    learning tasks where the model must learn to decompose images into
    constituent objects or layers without supervision.

    Design Rationale for Unsupervised Decomposition:
    ------------------------------------------------
    1. **No Labels Required**: Returns only images, suitable for reconstruction-
       based learning where the model learns to decompose and reconstruct.

    2. **Flexible Preprocessing**: Supports various preprocessing strategies
       (resize, crop, normalize) to handle diverse image sources while
       maintaining consistent input dimensions required by slot-based models.

    3. **Deterministic Ordering**: Sorted file loading ensures reproducibility
       across experiments, critical for scientific research.

    4. **RGB Consistency**: All images converted to RGB, preventing channel
       mismatch issues with grayscale or RGBA images.

    5. **Normalized Color Space**: Optional ImageNet normalization provides
       better-conditioned gradients for CNN encoders, accelerating convergence.

    6. **Spatial Uniformity**: Fixed spatial dimensions enable spatial broadcast
       decoding and consistent position encodings across batches.

    Compatible with Kaggle and standard filesystem structures.
    """

    def __init__(
        self,
        root_dir: str,
        image_size: Tuple[int, int] = (128, 128),
        crop_type: Optional[str] = None,
        use_normalization: bool = False,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        """Initialize image folder dataset.

        Args:
            root_dir: Path to root directory containing images (jpg/png/jpeg).
                     Images can be in nested subdirectories.
            image_size: Target image size as (height, width). Images will be
                       resized to this dimension.
            crop_type: Cropping strategy - 'center', 'random', or None.
                      - 'center': Deterministic center crop (for evaluation)
                      - 'random': Random crop (for training augmentation)
                      - None: No cropping, only resize
            use_normalization: If True, normalize with ImageNet mean/std.
                             Recommended for models pretrained on ImageNet or
                             when using standard CNN architectures.
            transform: Optional custom transform pipeline. If provided, other
                      transform arguments are ignored.

        Raises:
            ValueError: If root_dir does not exist or contains no valid images.
            ValueError: If crop_type is not in ['center', 'random', None].
        """
        self.root_dir = Path(root_dir)

        if not self.root_dir.exists():
            raise ValueError(f"Root directory does not exist: {root_dir}")

        if crop_type is not None and crop_type not in ['center', 'random']:
            raise ValueError(
                f"crop_type must be 'center', 'random', or None, got: {crop_type}"
            )

        self.image_size = image_size
        self.crop_type = crop_type
        self.use_normalization = use_normalization

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._build_transform()

        self.image_paths = self._load_image_paths()

        if len(self.image_paths) == 0:
            raise ValueError(
                f"No valid images found in {root_dir}. "
                "Supported formats: jpg, jpeg, png"
            )

        logger.info(
            f"Loaded {len(self.image_paths)} images from {root_dir} "
            f"(size={image_size}, crop={crop_type}, norm={use_normalization})"
        )

    def _build_transform(self) -> transforms.Compose:
        """Build transform pipeline based on configuration.

        Transform pipeline order:
        1. Resize to larger dimension (preserves aspect ratio)
        2. Optional crop to exact size
        3. Convert to tensor [0, 1]
        4. Optional normalization

        Returns:
            Composed torchvision transform pipeline
        """
        transform_list: List[transforms.Transform] = []

        # Resize to target size (may distort aspect ratio if not square)
        # For research: ensures consistent spatial dimensions
        transform_list.append(transforms.Resize(self.image_size))

        # Optional cropping for exact dimension control
        if self.crop_type == 'center':
            transform_list.append(transforms.CenterCrop(self.image_size))
        elif self.crop_type == 'random':
            transform_list.append(transforms.RandomCrop(self.image_size))

        # Convert PIL Image to tensor in range [0, 1]
        transform_list.append(transforms.ToTensor())

        # Optional normalization with ImageNet statistics
        # Improves gradient flow for standard CNN architectures
        if self.use_normalization:
            transform_list.append(
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            )

        return transforms.Compose(transform_list)

    def _load_image_paths(self) -> List[Path]:
        """Recursively load all valid image paths from root directory.

        Searches for common image formats (jpg, jpeg, png) in all subdirectories.
        Results are sorted for deterministic ordering across runs.

        Returns:
            Sorted list of Path objects pointing to valid images
        """
        valid_extensions = {'.jpg', '.jpeg', '.png'}

        image_paths = [
            p for p in self.root_dir.rglob('*')
            if p.suffix.lower() in valid_extensions and p.is_file()
        ]

        # Sort for reproducibility
        return sorted(image_paths)

    def __len__(self) -> int:
        """Return total number of images in dataset.

        Returns:
            Dataset size
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and transform image at given index.

        Args:
            idx: Index of image to load (0 <= idx < len(dataset))

        Returns:
            Transformed image tensor of shape (3, H, W) with dtype float32.
            Values in [0, 1] if not normalized, or standardized if normalized.

        Raises:
            IOError: If image cannot be loaded
        """
        image_path = self.image_paths[idx]

        try:
            # Load image and ensure RGB (handles grayscale, RGBA, etc.)
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise IOError(f"Cannot load image: {image_path}") from e

        # Apply transform pipeline
        image_tensor = self.transform(image)

        return image_tensor


def build_dataloader(
    root_dir: str,
    config: Dict[str, Any],
    shuffle: bool = True,
) -> DataLoader:
    """Build DataLoader from configuration dictionary.

    This is the primary factory function for creating dataloaders from
    YAML configuration files. It handles dataset creation, transform
    configuration, and dataloader setup in one call.

    Args:
        root_dir: Root directory containing images
        config: Configuration dictionary with keys:
            - image_size: [height, width] as list
            - batch_size: int
            - num_workers: int
            - crop_type: str or null ('center', 'random', or null)
            - use_normalization: bool (optional, default False)
        shuffle: Whether to shuffle data (True for training, False for eval)

    Returns:
        Configured PyTorch DataLoader ready for training/evaluation

    Example:
        >>> config = {
        ...     'image_size': [128, 128],
        ...     'batch_size': 64,
        ...     'num_workers': 4,
        ...     'crop_type': 'center',
        ...     'use_normalization': True
        ... }
        >>> loader = build_dataloader('/data/images', config, shuffle=True)
        >>> for batch in loader:
        ...     # batch shape: (64, 3, 128, 128)
        ...     pass
    """
    image_size = tuple(config['image_size'])
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    crop_type = config.get('crop_type', None)
    use_normalization = config.get('use_normalization', False)

    logger.info(f"Building dataloader for {root_dir}")

    dataset = ImageFolderDataset(
        root_dir=root_dir,
        image_size=image_size,
        crop_type=crop_type,
        use_normalization=use_normalization,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Only pin if CUDA available
        drop_last=False,  # Keep all data for evaluation
        persistent_workers=num_workers > 0,  # Reuse workers if available
    )

    logger.info(
        f"Created dataloader: {len(dataset)} images, "
        f"batch_size={batch_size}, num_workers={num_workers}"
    )

    return dataloader


# Legacy compatibility functions
def get_dataset(config: DatasetConfig, root_dir: str) -> Dataset:
    """Legacy factory function to create dataset.

    Note: For new code, prefer using ImageFolderDataset directly or
    build_dataloader() for complete dataloader setup.

    Args:
        config: Dataset configuration
        root_dir: Root directory containing images

    Returns:
        Dataset instance
    """
    logger.info(f"Creating dataset: {config.dataset_name}")

    dataset = ImageFolderDataset(
        root_dir=root_dir,
        image_size=config.image_size,
        crop_type=config.crop_type,
        use_normalization=config.use_normalization,
    )

    return dataset


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
) -> DataLoader:
    """Create DataLoader from dataset.

    Note: For new code, prefer using build_dataloader() which handles
    both dataset creation and dataloader setup from config.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
