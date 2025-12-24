"""Evaluation script for K-Conditioned Decomposition."""

import logging
from pathlib import Path
from typing import Dict, Any
import argparse

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from kcd.models.kcd_model import KCDModel
from kcd.data.datasets import get_dataset, get_dataloader, DatasetConfig
from kcd.utils import (
    setup_logging,
    set_seed,
    load_config,
    load_checkpoint,
    get_device,
)

logger = logging.getLogger(__name__)


def visualize_decomposition(
    images: torch.Tensor,
    reconstructions: torch.Tensor,
    masks: torch.Tensor,
    save_path: str,
    num_samples: int = 4,
) -> None:
    """Visualize image decomposition results.

    Args:
        images: Original images (B, 3, H, W)
        reconstructions: Reconstructed images (B, 3, H, W)
        masks: Slot masks (B, K, 1, H, W)
        save_path: Path to save visualization
        num_samples: Number of samples to visualize
    """
    images = images[:num_samples].cpu()
    reconstructions = reconstructions[:num_samples].cpu()
    masks = masks[:num_samples].cpu()

    batch_size = images.shape[0]
    num_slots = masks.shape[1]

    fig, axes = plt.subplots(
        batch_size,
        num_slots + 2,
        figsize=(3 * (num_slots + 2), 3 * batch_size),
    )

    if batch_size == 1:
        axes = axes.reshape(1, -1)

    for i in range(batch_size):
        axes[i, 0].imshow(images[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(reconstructions[i].permute(1, 2, 0).numpy())
        axes[i, 1].set_title("Reconstruction")
        axes[i, 1].axis("off")

        for k in range(num_slots):
            mask = masks[i, k, 0].numpy()
            axes[i, k + 2].imshow(mask, cmap="viridis")
            axes[i, k + 2].set_title(f"Slot {k+1}")
            axes[i, k + 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Visualization saved to {save_path}")


def evaluate(
    model: KCDModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_slots: int,
    output_dir: str,
) -> Dict[str, float]:
    """Evaluate model on dataset.

    Args:
        model: KCD model
        dataloader: Data loader
        device: Device to run on
        num_slots: Number of slots to use
        output_dir: Directory to save outputs

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    reconstruction_errors = []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            batch = batch.to(device)

            predictions = model(batch, num_slots)

            recons = predictions["recons"]
            masks = predictions["masks"]

            mse = torch.mean((batch - recons) ** 2, dim=[1, 2, 3])
            reconstruction_errors.extend(mse.cpu().numpy().tolist())

            if batch_idx == 0:
                vis_path = output_path / f"decomposition_k{num_slots}.png"
                visualize_decomposition(batch, recons, masks, str(vis_path))

    mean_mse = np.mean(reconstruction_errors)
    std_mse = np.std(reconstruction_errors)

    logger.info(f"Evaluation results (K={num_slots}):")
    logger.info(f"  Mean MSE: {mean_mse:.6f}")
    logger.info(f"  Std MSE: {std_mse:.6f}")

    return {
        "mean_mse": mean_mse,
        "std_mse": std_mse,
        "num_slots": num_slots,
    }


def run_evaluation(
    model_config_path: str,
    train_config_path: str,
    checkpoint_path: str,
    data_root: str,
    num_slots: int,
    output_dir: str,
) -> None:
    """Main evaluation function.

    Args:
        model_config_path: Path to model config YAML
        train_config_path: Path to training config YAML
        checkpoint_path: Path to model checkpoint
        data_root: Root directory of dataset
        num_slots: Number of slots to use for evaluation
        output_dir: Directory to save outputs
    """
    model_config = load_config(model_config_path)
    train_config = load_config(train_config_path)

    setup_logging(train_config["logging"]["log_level"])
    set_seed(train_config["training"]["seed"])

    device = get_device()

    logger.info("Creating dataset and dataloader")
    dataset_config = DatasetConfig(
        dataset_name=train_config["data"]["dataset_name"],
        image_size=tuple(train_config["data"]["image_size"]),
        batch_size=train_config["data"]["batch_size"],
        num_workers=train_config["data"]["num_workers"],
        train_split=train_config["data"]["train_split"],
    )

    dataset = get_dataset(dataset_config, data_root)
    dataloader = get_dataloader(
        dataset,
        batch_size=dataset_config.batch_size,
        num_workers=dataset_config.num_workers,
        shuffle=False,
    )

    logger.info("Building model")
    model = KCDModel.from_config(model_config)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters())

    load_checkpoint(model, optimizer, checkpoint_path, device)

    logger.info(f"Starting evaluation with K={num_slots}")
    metrics = evaluate(model, dataloader, device, num_slots, output_dir)

    logger.info("Evaluation completed")


def main() -> None:
    """Entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate K-Conditioned Decomposition model"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default="configs/train.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory of dataset",
    )
    parser.add_argument(
        "--num-slots",
        type=int,
        default=5,
        help="Number of slots to use for evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/figures",
        help="Directory to save output figures",
    )

    args = parser.parse_args()

    run_evaluation(
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        num_slots=args.num_slots,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
