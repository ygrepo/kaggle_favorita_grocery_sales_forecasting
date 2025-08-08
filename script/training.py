#!/usr/bin/env python3
"""
Training script for the Favorita Grocery Sales Forecasting model.

This script handles the complete training pipeline including:
- Data loading and preprocessing
- Model training
- Evaluation
- Model saving
- Logging
"""

import sys
import logging
import argparse
from pathlib import Path

import torch
import gc

# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import (
    MODEL_TYPE,
    FF_MODEL_TYPES,
)
from src.model_utils import train_all_models_for_cluster_pair
from src.data_utils import build_feature_and_label_cols
from src.utils import setup_logging, str2bool


def train(
    dataloader_dir: Path,
    model_dir: Path,
    model_logger_dir: Path,
    window_size: int,
    epochs: int,
    lr: float,
    seed: int,
    hidden_dim: int,
    h1: int,
    h2: int,
    depth: int,
    dropout: float,
    enable_progress_bar: bool,
    history_dir: Path,
    log_level: str,
):
    """
    Process each Parquet file in a directory, apply feature creation,
    and save the output with a prefix.

    Parameters
    ----------
    dataloader_dir : Path
        Directory to save dataloaders.
    model_dir : Path
        Directory to save trained models.
    model_logger_dir : Path
        Directory to save model logger.
    window_size : int
        Rolling window size for feature creation.
    history_dir : Path
        Directory to save training history.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    seed : int
        Random seed.
    enable_progress_bar : bool::
        Whether to enable progress bar.
    log_level : str
        Logging level (e.g., "INFO", "DEBUG").
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if dataloader_dir.is_file() and dataloader_dir.suffix == ".parquet":
        files = [dataloader_dir]
    else:
        files = list(dataloader_dir.glob("*_train_meta.parquet"))

    logger.info(
        f"Processing sales (store cluster, SKU cluster) {len(files)} Parquet files..."
    )

    (
        meta_cols,
        x_sales_features,
        x_cyclical_features,
        x_feature_cols,
        x_to_log_features,
        x_log_features,
        label_cols,
        y_log_features,
        y_to_log_features,
        all_features,
    ) = build_feature_and_label_cols(window_size)

    for file_path in files:
        logger.info(f"Processing {file_path.name}")
        parts = file_path.stem.split("_")
        store_cluster = int(parts[0])
        item_cluster = int(parts[1])
        logger.info(f"Store cluster: {store_cluster}")
        logger.info(f"Item cluster: {item_cluster}")
        train_all_models_for_cluster_pair(
            model_types=FF_MODEL_TYPES,
            model_dir=model_dir,
            model_logger_dir=model_logger_dir,
            dataloader_dir=dataloader_dir,
            label_cols=label_cols,
            y_to_log_features=y_to_log_features,
            store_cluster=store_cluster,
            item_cluster=item_cluster,
            history_dir=history_dir,
            lr=lr,
            epochs=epochs,
            seed=seed,
            hidden_dim=hidden_dim,
            h1=h1,
            h2=h2,
            depth=depth,
            dropout=dropout,
            enable_progress_bar=enable_progress_bar,
            log_level=log_level,
        )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Favorita Grocery Sales Forecasting model"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./output/models",
        help="Directory to save trained models (relative to project root)",
    )
    parser.add_argument(
        "--model_logger_dir",
        type=str,
        default="",
        help="Directory to save model logger (relative to project root)",
    )
    parser.add_argument(
        "--dataloader_dir",
        type=str,
        default="./output/dataloaders",
        help="Directory to save dataloaders (relative to project root)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=32,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--h1",
        type=int,
        default=64,
        help="Hidden dimension 1",
    )
    parser.add_argument(
        "--h2",
        type=int,
        default=32,
        help="Hidden dimension 2",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Depth",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Dropout rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed",
    )
    parser.add_argument(
        "--history_dir",
        type=str,
        default="",
        help="Directory to save training history (relative to project root)",
    )

    parser.add_argument(
        "--enable_progress_bar",
        type=str2bool,
        default=True,
        help="Whether to enable progress bar",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=1,
        help="Size of the lookback window",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="./output/logs",
        help="Directory to save training outputs (relative to project root)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()

    # Convert paths to absolute paths relative to project root
    project_root = Path(__file__).parent.parent
    dataloader_dir = (project_root / args.dataloader_dir).resolve()
    model_dir = (project_root / args.model_dir).resolve()
    model_logger_dir = (project_root / args.model_logger_dir).resolve()
    history_dir = (project_root / args.history_dir).resolve()
    log_dir = (project_root / args.log_dir).resolve()
    enable_progress_bar = args.enable_progress_bar

    # Set up logging
    logger = setup_logging(log_dir, args.log_level)

    try:
        # Log configuration
        logger.info("Starting training with configuration:")
        logger.info(f"  Project root: {project_root}")
        logger.info(f"  Dataloader directory: {dataloader_dir}")
        logger.info(f"  Model directory: {model_dir}")
        logger.info(f"  Model logger directory: {model_logger_dir}")
        logger.info(f"  History directory: {history_dir}")
        logger.info(f"  Window size: {args.window_size}")
        logger.info(f"  Epochs: {args.epochs}")
        logger.info(f"  Learning rate: {args.lr}")
        logger.info(f"  Hidden dimension: {args.hidden_dim}")
        logger.info(f"  Hidden dimension 1: {args.h1}")
        logger.info(f"  Hidden dimension 2: {args.h2}")
        logger.info(f"  Depth: {args.depth}")
        logger.info(f"  Dropout rate: {args.dropout}")
        logger.info(f"  Seed: {args.seed}")
        logger.info(f"  Enable progress bar: {enable_progress_bar}")
        torch.cuda.empty_cache()
        gc.collect()

        if not torch.cuda.is_available():
            logger.warning("⚠️ CUDA not available. Training will run on CPU.")
        else:
            logger.info("✅ CUDA is available. Proceeding with GPU training.")

        # mp.set_sharing_strategy("file_system")
        torch.multiprocessing.set_sharing_strategy("file_system")

        # Train model
        train(
            dataloader_dir=dataloader_dir,
            model_dir=model_dir,
            model_logger_dir=model_logger_dir,
            window_size=args.window_size,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            h1=args.h1,
            h2=args.h2,
            depth=args.depth,
            dropout=args.dropout,
            seed=args.seed,
            history_dir=history_dir,
            enable_progress_bar=enable_progress_bar,
            log_level=args.log_level,
        )
        logger.info("Training completed successfully!")

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
