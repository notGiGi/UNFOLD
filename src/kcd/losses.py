"""Loss functions for K-Conditioned Decomposition."""

import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class KCDLoss(nn.Module):
    """Loss function for K-Conditioned Decomposition."""

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        regularization_weight: float = 0.0,
    ) -> None:
        """Initialize loss function.

        Args:
            reconstruction_weight: Weight for reconstruction loss
            regularization_weight: Weight for regularization term
        """
        super().__init__()

        self.reconstruction_weight = reconstruction_weight
        self.regularization_weight = regularization_weight

        logger.info(
            f"KCDLoss initialized: recons_w={reconstruction_weight}, "
            f"reg_w={regularization_weight}"
        )

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss.

        Args:
            predictions: Dictionary with 'recons', 'masks', 'slots'
            targets: Ground truth images of shape (B, 3, H, W)

        Returns:
            Dictionary containing:
                - total_loss: Combined loss
                - reconstruction_loss: MSE reconstruction loss
                - regularization_loss: Optional regularization
        """
        recons = predictions["recons"]

        reconstruction_loss = F.mse_loss(recons, targets)

        regularization_loss = torch.tensor(0.0, device=targets.device)

        total_loss = (
            self.reconstruction_weight * reconstruction_loss +
            self.regularization_weight * regularization_loss
        )

        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "regularization_loss": regularization_loss,
        }
