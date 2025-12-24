"""K-Conditioned Decomposition (KCD) for unsupervised image layer decomposition."""

__version__ = "0.1.0"
__author__ = "KCD Research Team"

from .models.kcd_model import KCDModel
from .losses import KCDLoss

__all__ = ["KCDModel", "KCDLoss"]
