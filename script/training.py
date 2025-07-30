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

import os

import sys
import logging
import argparse
from pathlib import Path

import torch
import gc
import torch.multiprocessing as mp

# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model_utils import (
    MODEL_TYPES,
    train_all_models_for_cluster_pair,
)
from src.data_utils import build_feature_and_label_cols
from src.utils import setup_logging, str2bool


def train(
    dataloader_dir: Path,
    model_dir: Path,
    window_size: int,
    epochs: int,
    num_workers: int,
    persistent_workers: bool,
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
    window_size : int
        Rolling window size for feature creation.
    history_dir : Path
        Directory to save training history.
    epochs : int
        Number of training epochs.
    num_workers : int
        Number of subprocesses to use for data loading.
    persistent_workers : bool
        Whether to use persistent workers for data loading.s
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
            model_types=[MODEL_TYPES.SHALLOW_NN],
            epochs=epochs,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            model_dir=model_dir,
            dataloader_dir=dataloader_dir,
            label_cols=label_cols,
            y_log_features=y_log_features,
            store_cluster=store_cluster,
            item_cluster=item_cluster,
            history_dir=history_dir,
            log_level=log_level,
        )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Favorita Grocery Sales Forecasting model"
    )
    parser.add_argument(
        "--dataloader_dir",
        type=str,
        default="./output/dataloaders",
        help="Directory to save dataloaders (relative to project root)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./output/models",
        help="Directory to save trained models (relative to project root)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading",
    )
    parser.add_argument(
        "--persistent_workers",
        type=str2bool,
        default=True,
        help="Whether to use persistent workers for data loading",
    )
    parser.add_argument(
        "--history_dir",
        type=str,
        default="",
        help="Directory to save training history (relative to project root)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=1,
        help="Size of the lookback window",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
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
    history_dir = (project_root / args.history_dir).resolve()
    log_dir = (project_root / args.log_dir).resolve()
    num_workers = args.num_workers
    persistent_workers = args.persistent_workers

    # Set up logging
    logger = setup_logging(log_dir, args.log_level)

    try:
        # Log configuration
        logger.info("Starting training with configuration:")
        logger.info(f"  Project root: {project_root}")
        logger.info(f"  Dataloader directory: {dataloader_dir}")
        logger.info(f"  Model directory: {model_dir}")
        logger.info(f"  History directory: {history_dir}")
        logger.info(f"  Window size: {args.window_size}")
        logger.info(f"  Epochs: {args.epochs}")
        logger.info(f"  Num workers: {num_workers}")
        logger.info(f"  Persistent workers: {persistent_workers}")
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
            window_size=args.window_size,
            epochs=args.epochs,
            history_dir=history_dir,
            log_level=args.log_level,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )
        logger.info("Training completed successfully!")

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
