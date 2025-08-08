#!/usr/bin/env python3
"""
Learning Rate Finder Script for Favorita Grocery Sales Forecasting

This script helps find the optimal learning rate by training a model with
exponentially increasing learning rates and plotting the loss curve.

Usage:
    python script/lr_finder.py --dataloader_dir ./output/dataloaders --store_cluster 17 --item_cluster 15
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import (
    LightningWrapper,
    ModelType,
    model_factory,
    init_weights,
    compute_mav,
)
from src.model_utils import get_device
import lightning.pytorch as pl


class LRFinder:
    """Learning Rate Finder implementation"""

    def __init__(self, model: pl.LightningModule, train_loader, val_loader=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr_history = []
        self.loss_history = []

    def range_test(
        self,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_iter: int = 100,
        step_mode: str = "exp",
        smooth_f: float = 0.05,
        divergence_th: int = 5,
    ) -> Tuple[List[float], List[float]]:
        """
        Perform learning rate range test

        Args:
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations to test
            step_mode: How to step between lr values ('exp' or 'linear')
            smooth_f: Smoothing factor for loss
            divergence_th: Stop if loss increases by this factor

        Returns:
            Tuple of (learning_rates, losses)
        """
        # Setup
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=start_lr)

        if step_mode == "exp":
            lr_schedule = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iter)
        else:
            lr_schedule = np.linspace(start_lr, end_lr, num_iter)

        self.lr_history = []
        self.loss_history = []
        best_loss = float("inf")

        # Get data iterator
        data_iter = iter(self.train_loader)

        for i, lr in enumerate(lr_schedule):
            # Update learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            try:
                # Get next batch
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)

                # Forward pass
                optimizer.zero_grad()

                if len(batch) == 3:  # (x, y, weights)
                    x, y, weights = batch
                    x, y, weights = (
                        x.to(self.model.device),
                        y.to(self.model.device),
                        weights.to(self.model.device),
                    )
                    loss = self.model.training_step((x, y, weights), i)
                else:  # (x, y)
                    x, y = batch
                    x, y = x.to(self.model.device), y.to(self.model.device)
                    loss = self.model.training_step((x, y), i)

                # Extract loss value
                if isinstance(loss, dict):
                    loss_val = loss["loss"].item()
                else:
                    loss_val = loss.item()

                # Backward pass
                if isinstance(loss, dict):
                    loss["loss"].backward()
                else:
                    loss.backward()
                optimizer.step()

                # Record
                self.lr_history.append(lr)
                self.loss_history.append(loss_val)

                # Smooth loss for divergence check
                if i == 0:
                    smoothed_loss = loss_val
                    best_loss = loss_val
                else:
                    smoothed_loss = (
                        smooth_f * loss_val + (1 - smooth_f) * self.loss_history[-2]
                    )

                # Check for divergence
                if smoothed_loss > divergence_th * best_loss:
                    print(f"Stopping early at iteration {i}, loss diverged")
                    break

                if smoothed_loss < best_loss:
                    best_loss = smoothed_loss

                if i % 10 == 0:
                    print(
                        f"Iteration {i}/{num_iter}, LR: {lr:.2e}, Loss: {loss_val:.4f}"
                    )

            except Exception as e:
                print(f"Error at iteration {i}: {e}")
                break

        return self.lr_history, self.loss_history

    def plot(
        self,
        skip_start: int = 10,
        skip_end: int = 5,
        log_lr: bool = True,
        save_path: Optional[Path] = None,
        show_plot: bool = True,
    ):
        """Plot the learning rate finder results"""
        if len(self.lr_history) == 0:
            raise ValueError("No data to plot. Run range_test first.")

        # Skip some points at start and end
        lrs = (
            self.lr_history[skip_start:-skip_end]
            if skip_end > 0
            else self.lr_history[skip_start:]
        )
        losses = (
            self.loss_history[skip_start:-skip_end]
            if skip_end > 0
            else self.loss_history[skip_start:]
        )

        plt.figure(figsize=(10, 6))
        if log_lr:
            plt.semilogx(lrs, losses)
            plt.xlabel("Learning Rate (log scale)")
        else:
            plt.plot(lrs, losses)
            plt.xlabel("Learning Rate")

        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")
        plt.grid(True, alpha=0.3)

        # Find suggested LR (steepest descent)
        if len(losses) > 1:
            gradients = np.gradient(losses)
            min_gradient_idx = np.argmin(gradients)
            suggested_lr = lrs[min_gradient_idx]
            plt.axvline(
                x=suggested_lr,
                color="red",
                linestyle="--",
                label=f"Suggested LR: {suggested_lr:.2e}",
            )
            plt.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return suggested_lr if len(losses) > 1 else None

    def suggest_lr(self, skip_start: int = 10, skip_end: int = 5) -> float:
        """Suggest optimal learning rate based on steepest descent"""
        if len(self.loss_history) == 0:
            raise ValueError("No data available. Run range_test first.")

        losses = (
            self.loss_history[skip_start:-skip_end]
            if skip_end > 0
            else self.loss_history[skip_start:]
        )
        lrs = (
            self.lr_history[skip_start:-skip_end]
            if skip_end > 0
            else self.lr_history[skip_start:]
        )

        if len(losses) <= 1:
            return self.lr_history[0]

        gradients = np.gradient(losses)
        min_gradient_idx = np.argmin(gradients)
        return lrs[min_gradient_idx]


def load_data_for_cluster(
    dataloader_dir: Path, store_cluster: int, item_cluster: int
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Load train and validation dataloaders for specific cluster pair"""

    train_file = dataloader_dir / f"{store_cluster}_{item_cluster}_train_loader.pt"
    val_file = dataloader_dir / f"{store_cluster}_{item_cluster}_val_loader.pt"

    if not train_file.exists():
        raise FileNotFoundError(f"Training dataloader not found: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Validation dataloader not found: {val_file}")

    train_loader = torch.load(train_file)
    val_loader = torch.load(val_file)

    return train_loader, val_loader


def run_lr_finder(
    dataloader_dir: Path,
    store_cluster: int,
    item_cluster: int,
    model_type: ModelType = ModelType.SHALLOW_NN,
    label_cols: List[str] = None,
    y_to_log_features: List[str] = None,
    output_dir: Path = None,
    start_lr: float = 1e-7,
    end_lr: float = 1.0,
    num_iter: int = 100,
    log_level: str = "INFO",
) -> float:
    """
    Run learning rate finder for a specific model and cluster pair

    Returns:
        Suggested optimal learning rate
    """

    # Setup logging
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    logger = logging.getLogger(__name__)

    # Default values
    if label_cols is None:
        label_cols = ["unit_sales"]
    if y_to_log_features is None:
        y_to_log_features = ["unit_sales"]
    if output_dir is None:
        output_dir = Path("./output/lr_finder")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Running LR Finder for store_cluster={store_cluster}, item_cluster={item_cluster}"
    )
    logger.info(f"Model type: {model_type.value}")

    # Load data
    train_loader, val_loader = load_data_for_cluster(
        dataloader_dir, store_cluster, item_cluster
    )

    # Get input/output dimensions
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[1]
    output_dim = len(label_cols)

    logger.info(f"Input dim: {input_dim}, Output dim: {output_dim}")

    # Create model
    base_model = model_factory(model_type, input_dim, output_dim)
    base_model.apply(init_weights)

    # Compute MAV for metrics
    col_y_index_map = {col: idx for idx, col in enumerate(label_cols)}
    y_log_idx = [col_y_index_map[c] for c in y_to_log_features]
    train_mav = compute_mav(train_loader, y_log_idx, logger)
    val_mav = compute_mav(val_loader, y_log_idx, logger)

    # Create Lightning wrapper
    model_name = f"lr_finder_{store_cluster}_{item_cluster}_{model_type.value}"
    lightning_model = LightningWrapper(
        base_model,
        model_name=model_name,
        store=store_cluster,
        item=item_cluster,
        sales_idx=y_log_idx,
        train_mav=train_mav,
        val_mav=val_mav,
        lr=1e-3,  # Initial LR, will be overridden
        log_level=log_level,
    )

    # Move to device
    device = get_device()
    lightning_model = lightning_model.to(device)

    # Run LR finder
    lr_finder = LRFinder(lightning_model, train_loader, val_loader)

    logger.info(f"Starting LR range test from {start_lr} to {end_lr}")
    lrs, losses = lr_finder.range_test(
        start_lr=start_lr, end_lr=end_lr, num_iter=num_iter
    )

    # Plot and save results
    plot_path = (
        output_dir / f"lr_finder_{store_cluster}_{item_cluster}_{model_type.value}.png"
    )
    suggested_lr = lr_finder.plot(save_path=plot_path, show_plot=False)

    # Save data
    results_df = pd.DataFrame({"learning_rate": lrs, "loss": losses})
    csv_path = (
        output_dir / f"lr_finder_{store_cluster}_{item_cluster}_{model_type.value}.csv"
    )
    results_df.to_csv(csv_path, index=False)

    logger.info(f"Results saved to {csv_path}")
    logger.info(f"Plot saved to {plot_path}")
    logger.info(f"Suggested learning rate: {suggested_lr:.2e}")

    return suggested_lr


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Learning Rate Finder for Favorita Sales Forecasting"
    )

    parser.add_argument(
        "--dataloader_dir",
        type=str,
        required=True,
        help="Directory containing the dataloaders",
    )
    parser.add_argument(
        "--store_cluster", type=int, required=True, help="Store cluster ID"
    )
    parser.add_argument(
        "--item_cluster", type=int, required=True, help="Item cluster ID"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="ShallowNN",
        choices=["ShallowNN", "TwoLayerNN", "ResidualMLP"],
        help="Model type to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/lr_finder",
        help="Output directory for results",
    )
    parser.add_argument(
        "--start_lr", type=float, default=1e-7, help="Starting learning rate"
    )
    parser.add_argument(
        "--end_lr", type=float, default=1.0, help="Ending learning rate"
    )
    parser.add_argument(
        "--num_iter", type=int, default=100, help="Number of iterations"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Convert paths
    dataloader_dir = Path(args.dataloader_dir)
    output_dir = Path(args.output_dir)

    # Convert model type
    model_type = ModelType(args.model_type.upper())

    # Run LR finder
    suggested_lr = run_lr_finder(
        dataloader_dir=dataloader_dir,
        store_cluster=args.store_cluster,
        item_cluster=args.item_cluster,
        model_type=model_type,
        output_dir=output_dir,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        num_iter=args.num_iter,
        log_level=args.log_level,
    )

    print(f"\n{'='*50}")
    print(f"LEARNING RATE FINDER RESULTS")
    print(f"{'='*50}")
    print(f"Store Cluster: {args.store_cluster}")
    print(f"Item Cluster: {args.item_cluster}")
    print(f"Model Type: {args.model_type}")
    print(f"Suggested Learning Rate: {suggested_lr:.2e}")
    print(f"{'='*50}")
