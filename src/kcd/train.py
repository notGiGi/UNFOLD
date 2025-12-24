"""Training loop for K-Conditioned Decomposition.

This module implements a research-grade training loop with:
- K-conditioned training (variable K per batch)
- Multiple loss components for unsupervised decomposition
- Mixed precision training for efficiency
- Comprehensive logging and checkpointing
- Deterministic training with seed control
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import yaml

from .models.kcd_model import KCDModel

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training run.

    This dataclass encapsulates all hyperparameters and settings for a
    K-conditioned training run.
    """
    # K-conditioning parameters
    k_min: int = 3  # Minimum number of slots
    k_max: int = 7  # Maximum number of slots

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_epochs: int = 100
    gradient_clip: float = 1.0

    # Loss weights
    recon_loss_type: str = "mse"  # "mse" or "l1"
    overlap_penalty_weight: float = 0.1
    tv_loss_weight: float = 0.01
    slot_usage_weight: float = 0.001

    # Training settings
    mixed_precision: bool = True
    deterministic: bool = True
    seed: int = 42

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_every: int = 5  # Save checkpoint every N epochs
    log_every: int = 10  # Log metrics every N steps

    # Dataset
    batch_size: int = 32
    num_workers: int = 4


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducible training.

    Ensures deterministic behavior across Python, NumPy, and PyTorch.
    Critical for research reproducibility and K-conditioned training
    where randomness affects K sampling.

    Args:
        seed: Random seed value
        deterministic: If True, use deterministic CUDA operations
                      (slower but reproducible)

    Note:
        Deterministic mode may reduce performance but ensures exact
        reproducibility across runs, which is important for ablations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Deterministic mode enabled with seed {seed}")
    else:
        torch.backends.cudnn.benchmark = True
        logger.info(f"Non-deterministic mode with seed {seed}")


def sample_k(k_min: int, k_max: int) -> int:
    """Sample number of slots K uniformly from [K_min, K_max].

    This is the core of K-conditioned training. Each batch uses a
    different K value, forcing the model to learn decomposition at
    multiple granularities simultaneously.

    Benefits:
    1. Single model handles variable object counts
    2. Prevents overfitting to specific K values
    3. Acts as regularization
    4. Improves generalization to unseen K at test time

    Args:
        k_min: Minimum number of slots (inclusive)
        k_max: Maximum number of slots (inclusive)

    Returns:
        Randomly sampled K value

    Example:
        >>> # Training loop
        >>> for batch in dataloader:
        ...     K = sample_k(3, 7)  # K ∈ {3, 4, 5, 6, 7}
        ...     outputs = model(batch, num_slots=K)
    """
    return random.randint(k_min, k_max)


class LossComputer:
    """Compute all loss components for unsupervised decomposition.

    This class implements the complete loss function:
        L_total = L_recon + λ_overlap·L_overlap + λ_tv·L_tv + λ_usage·L_usage

    Each component serves a specific purpose in unsupervised learning:

    1. **Reconstruction Loss**: Forces model to reconstruct input
       - MSE or L1 distance between input and reconstruction
       - Drives the entire learning process
       - Without other losses, tends to create blurry reconstructions

    2. **Overlap Penalty**: Encourages sharp alpha masks
       - Penalizes slots covering same pixels (α_i·α_j should be small)
       - Forces slots to specialize on distinct regions
       - Prevents "explaining away" where all slots claim same pixels

    3. **Total Variation Loss**: Smooths alpha masks spatially
       - Penalizes spatial discontinuities in alpha masks
       - Encourages coherent, blob-like regions
       - Prevents noisy, scattered alpha masks

    4. **Slot Usage Regularizer**: Prevents dead slots
       - Encourages all K slots to be used
       - Prevents collapse where few slots explain entire image
       - Uses entropy of slot usage to measure diversity

    The balance between these losses is crucial for successful decomposition.
    """

    def __init__(
        self,
        recon_loss_type: str = "mse",
        overlap_penalty_weight: float = 0.1,
        tv_loss_weight: float = 0.01,
        slot_usage_weight: float = 0.001,
    ):
        """Initialize loss computer with weights.

        Args:
            recon_loss_type: Type of reconstruction loss ("mse" or "l1")
            overlap_penalty_weight: Weight for overlap penalty (λ_overlap)
            tv_loss_weight: Weight for TV loss on alphas (λ_tv)
            slot_usage_weight: Weight for slot usage regularizer (λ_usage)
        """
        self.recon_loss_type = recon_loss_type
        self.overlap_penalty_weight = overlap_penalty_weight
        self.tv_loss_weight = tv_loss_weight
        self.slot_usage_weight = slot_usage_weight

    def compute_reconstruction_loss(
        self,
        images: torch.Tensor,
        recons: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reconstruction loss: ||I - Î||.

        Args:
            images: Ground truth images (B, 3, H, W)
            recons: Reconstructed images (B, 3, H, W)

        Returns:
            Scalar reconstruction loss
        """
        if self.recon_loss_type == "mse":
            # L2 loss: mean squared error
            loss = F.mse_loss(recons, images)
        elif self.recon_loss_type == "l1":
            # L1 loss: mean absolute error (more robust to outliers)
            loss = F.l1_loss(recons, images)
        else:
            raise ValueError(f"Unknown recon_loss_type: {self.recon_loss_type}")

        return loss

    def compute_overlap_penalty(
        self,
        layer_alphas: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize overlapping alpha masks between slots.

        The overlap penalty encourages slots to claim distinct pixels:
            L_overlap = Σ_{i≠j} Σ_{x,y} α_i(x,y) · α_j(x,y)

        High overlap means multiple slots claim the same pixel, which is
        wasteful. This loss pushes alphas toward one-hot assignments.

        Args:
            layer_alphas: Alpha masks of shape (B, K, 1, H, W)

        Returns:
            Scalar overlap penalty

        Note:
            Since alphas already sum to 1 via softmax, we compute pairwise
            products and sum. This is equivalent to measuring how "peaked"
            the alpha distribution is at each pixel.
        """
        batch_size, num_slots, _, height, width = layer_alphas.shape

        # Reshape to (B, K, H*W) for easier computation
        alphas = layer_alphas.squeeze(2).reshape(batch_size, num_slots, -1)
        # Shape: (B, K, N) where N = H*W

        # Compute pairwise overlap: α_i · α_j for all i ≠ j
        # Method: sum of squares minus sum of individual squares
        # Σ_{i≠j} α_i·α_j = (Σ_i α_i)^2 - Σ_i α_i^2
        alphas_sum_sq = alphas.sum(dim=1) ** 2  # (B, N)
        alphas_sq_sum = (alphas ** 2).sum(dim=1)  # (B, N)

        # Overlap at each pixel
        overlap = alphas_sum_sq - alphas_sq_sum  # (B, N)

        # Average over pixels and batch
        loss = overlap.mean()

        return loss

    def compute_tv_loss(
        self,
        layer_alphas: torch.Tensor,
    ) -> torch.Tensor:
        """Compute total variation loss on alpha masks.

        Total variation (TV) measures spatial smoothness:
            TV(α) = Σ_{x,y} |α(x+1,y) - α(x,y)| + |α(x,y+1) - α(x,y)|

        This encourages alpha masks to be spatially coherent (blob-like)
        rather than noisy or scattered. It acts as spatial regularization.

        Args:
            layer_alphas: Alpha masks of shape (B, K, 1, H, W)

        Returns:
            Scalar TV loss

        Note:
            We use L1 norm (absolute differences) rather than L2 to
            encourage sharp boundaries while maintaining smoothness.
        """
        # Compute differences along height and width
        # Height: difference between adjacent rows
        tv_h = torch.abs(
            layer_alphas[:, :, :, 1:, :] - layer_alphas[:, :, :, :-1, :]
        )

        # Width: difference between adjacent columns
        tv_w = torch.abs(
            layer_alphas[:, :, :, :, 1:] - layer_alphas[:, :, :, :, :-1]
        )

        # Sum TV along both dimensions and average
        loss = tv_h.mean() + tv_w.mean()

        return loss

    def compute_slot_usage_loss(
        self,
        layer_alphas: torch.Tensor,
    ) -> torch.Tensor:
        """Encourage all slots to be used (prevent dead slots).

        Measures how uniformly slots are used across the batch:
            Usage_k = mean_{b,x,y} α_k(b,x,y)
            L_usage = -H(Usage) where H is entropy

        High entropy means all slots used equally.
        Low entropy means few slots dominate (dead slots).

        Args:
            layer_alphas: Alpha masks of shape (B, K, 1, H, W)

        Returns:
            Scalar slot usage loss (negative entropy)

        Note:
            This prevents mode collapse where the model uses only a subset
            of available slots, wasting capacity.
        """
        # Average alpha value for each slot across batch and spatial dims
        slot_usage = layer_alphas.mean(dim=[0, 2, 3, 4])  # (K,)

        # Normalize to probability distribution
        slot_usage = slot_usage / (slot_usage.sum() + 1e-8)

        # Compute entropy: H = -Σ p·log(p)
        # Higher entropy = more uniform usage = better
        entropy = -(slot_usage * torch.log(slot_usage + 1e-8)).sum()

        # We want to maximize entropy, so minimize negative entropy
        loss = -entropy

        return loss

    def compute_total_loss(
        self,
        images: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss and breakdown.

        Combines all loss components with their weights:
            L = L_recon + λ_1·L_overlap + λ_2·L_tv + λ_3·L_usage

        Args:
            images: Ground truth images (B, 3, H, W)
            outputs: Model outputs dictionary with keys:
                    - recons: Reconstructed images (B, 3, H, W)
                    - layer_alphas: Alpha masks (B, K, 1, H, W)

        Returns:
            total_loss: Scalar loss for backpropagation
            loss_dict: Dictionary with breakdown of all components

        Example:
            >>> loss_computer = LossComputer()
            >>> outputs = model(images, num_slots=5)
            >>> loss, breakdown = loss_computer.compute_total_loss(images, outputs)
            >>> loss.backward()
            >>> print(breakdown)  # {'recon': 0.5, 'overlap': 0.02, ...}
        """
        recons = outputs["recons"]
        layer_alphas = outputs["layer_alphas"]

        # Compute individual loss components
        recon_loss = self.compute_reconstruction_loss(images, recons)
        overlap_loss = self.compute_overlap_penalty(layer_alphas)
        tv_loss = self.compute_tv_loss(layer_alphas)
        usage_loss = self.compute_slot_usage_loss(layer_alphas)

        # Weighted combination
        total_loss = (
            recon_loss
            + self.overlap_penalty_weight * overlap_loss
            + self.tv_loss_weight * tv_loss
            + self.slot_usage_weight * usage_loss
        )

        # Create breakdown for logging
        loss_dict = {
            "total": total_loss.item(),
            "recon": recon_loss.item(),
            "overlap": overlap_loss.item(),
            "tv": tv_loss.item(),
            "usage": usage_loss.item(),
        }

        return total_loss, loss_dict


class Trainer:
    """K-Conditioned training loop with comprehensive logging and checkpointing.

    This class implements the complete training pipeline:
    1. K-conditioned batch sampling
    2. Forward pass with variable K
    3. Multi-component loss computation
    4. Mixed precision training
    5. Gradient clipping
    6. Checkpointing and logging
    7. Resume from checkpoint

    Designed for Kaggle GPU environments with:
    - Efficient memory usage (mixed precision)
    - Robust checkpointing (resume after crashes)
    - Detailed logging (JSONL format for analysis)
    - Deterministic training (reproducibility)
    """

    def __init__(
        self,
        model: KCDModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: TrainingConfig,
        device: torch.device,
    ):
        """Initialize trainer.

        Args:
            model: KCDModel instance
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration
            device: Device to train on (cuda or cpu)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Set up directories
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.log_dir = Path(config.log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Loss computer
        self.loss_computer = LossComputer(
            recon_loss_type=config.recon_loss_type,
            overlap_penalty_weight=config.overlap_penalty_weight,
            tv_loss_weight=config.tv_loss_weight,
            slot_usage_weight=config.slot_usage_weight,
        )

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=config.mixed_precision)

        # Training state
        self.current_epoch = 0
        self.global_step = 0

        # Logging
        self.train_log_file = self.log_dir / "train.jsonl"
        self.val_log_file = self.log_dir / "val.jsonl"

        logger.info(
            f"Trainer initialized:\n"
            f"  Device: {device}\n"
            f"  K range: [{config.k_min}, {config.k_max}]\n"
            f"  Learning rate: {config.learning_rate}\n"
            f"  Mixed precision: {config.mixed_precision}\n"
            f"  Deterministic: {config.deterministic}\n"
            f"  Checkpoint dir: {self.checkpoint_dir}\n"
            f"  Log dir: {self.log_dir}"
        )

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary with average metrics for the epoch
        """
        self.model.train()

        epoch_metrics = {
            "total": 0.0,
            "recon": 0.0,
            "overlap": 0.0,
            "tv": 0.0,
            "usage": 0.0,
        }
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Get images (dataset returns dict with 'image' key)
            if isinstance(batch, dict):
                images = batch["image"].to(self.device)
            else:
                images = batch.to(self.device)

            # Sample K for this batch (K-conditioning!)
            K = sample_k(self.config.k_min, self.config.k_max)

            # Forward pass with mixed precision
            with autocast(enabled=self.config.mixed_precision):
                outputs = self.model(images, num_slots=K)
                loss, loss_dict = self.loss_computer.compute_total_loss(
                    images, outputs
                )

            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping (unscale first for accurate clipping)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip,
            )

            # Optimizer step with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate metrics
            for key in epoch_metrics:
                epoch_metrics[key] += loss_dict[key]
            num_batches += 1

            # Log every N steps
            if self.global_step % self.config.log_every == 0:
                self._log_step(loss_dict, K, "train")

            self.global_step += 1

        # Average metrics over epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        return epoch_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation.

        Returns:
            Dictionary with average validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()

        val_metrics = {
            "total": 0.0,
            "recon": 0.0,
            "overlap": 0.0,
            "tv": 0.0,
            "usage": 0.0,
        }
        num_batches = 0

        for batch in self.val_loader:
            # Get images
            if isinstance(batch, dict):
                images = batch["image"].to(self.device)
            else:
                images = batch.to(self.device)

            # Use fixed K for validation (middle of range)
            K = (self.config.k_min + self.config.k_max) // 2

            # Forward pass
            with autocast(enabled=self.config.mixed_precision):
                outputs = self.model(images, num_slots=K)
                _, loss_dict = self.loss_computer.compute_total_loss(
                    images, outputs
                )

            # Accumulate metrics
            for key in val_metrics:
                val_metrics[key] += loss_dict[key]
            num_batches += 1

        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= num_batches

        return val_metrics

    def train(self) -> None:
        """Run full training loop.

        Trains for config.max_epochs, saving checkpoints and logging metrics.
        """
        logger.info(f"Starting training for {self.config.max_epochs} epochs")
        logger.info(f"Training batches per epoch: {len(self.train_loader)}")

        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch

            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{self.config.max_epochs}")
            logger.info(f"{'='*60}")

            # Train epoch
            train_metrics = self.train_epoch()
            logger.info(
                f"Train - "
                f"Total: {train_metrics['total']:.6f}, "
                f"Recon: {train_metrics['recon']:.6f}, "
                f"Overlap: {train_metrics['overlap']:.6f}"
            )

            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate()
                logger.info(
                    f"Val   - "
                    f"Total: {val_metrics['total']:.6f}, "
                    f"Recon: {val_metrics['recon']:.6f}, "
                    f"Overlap: {val_metrics['overlap']:.6f}"
                )

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

        logger.info("\nTraining completed!")
        self.save_checkpoint("checkpoint_final.pt")

    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": asdict(self.config),
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint and resume training.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]

        logger.info(
            f"Loaded checkpoint from {checkpoint_path}\n"
            f"  Resuming from epoch {self.current_epoch}\n"
            f"  Global step: {self.global_step}"
        )

    def _log_step(
        self,
        loss_dict: Dict[str, float],
        K: int,
        split: str,
    ) -> None:
        """Log training step to JSONL file.

        Args:
            loss_dict: Dictionary of loss values
            K: Number of slots used this step
            split: "train" or "val"
        """
        log_file = self.train_log_file if split == "train" else self.val_log_file

        log_entry = {
            "step": self.global_step,
            "epoch": self.current_epoch,
            "K": K,
            **loss_dict,
        }

        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


def train_from_config(
    model_config_path: str,
    train_config_path: str,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    device: Optional[torch.device] = None,
    resume_from: Optional[str] = None,
) -> Trainer:
    """Train KCD model from configuration files.

    This is the main entry point for training. Loads configs, initializes
    model and trainer, and runs training loop.

    Args:
        model_config_path: Path to model config YAML
        train_config_path: Path to training config YAML
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        device: Device to train on (auto-detect if None)
        resume_from: Path to checkpoint to resume from (optional)

    Returns:
        Trained Trainer instance

    Example:
        >>> from torch.utils.data import DataLoader
        >>> from src.kcd.data.datasets import ImageFolderDataset
        >>>
        >>> # Create dataset and loader
        >>> dataset = ImageFolderDataset("data/train")
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>>
        >>> # Train
        >>> trainer = train_from_config(
        ...     "configs/model.yaml",
        ...     "configs/training.yaml",
        ...     loader,
        ... )
    """
    # Auto-detect device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configs
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    with open(train_config_path, "r") as f:
        train_config_dict = yaml.safe_load(f)

    # Create training config
    train_config = TrainingConfig(**train_config_dict)

    # Set seed for reproducibility
    set_seed(train_config.seed, train_config.deterministic)

    # Build model
    logger.info("Building model from config...")
    model = KCDModel.from_config(model_config)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device,
    )

    # Resume from checkpoint if specified
    if resume_from is not None:
        trainer.load_checkpoint(resume_from)

    # Run training
    trainer.train()

    return trainer
