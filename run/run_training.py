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
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import gc

# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model_utils import (
    MODEL_TYPES,
    train_all_models_for_cluster_pair,
)
from src.data_utils import build_feature_and_label_cols
from src.utils import setup_logging


def train(
    dataloader_dir: Path,
    model_dir: Path,
    window_size: int,
    epochs: int,
    history_fn: Path,
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
    history_fn : Path
        Path to save training history.
    epochs : int
        Number of training epochs.
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
    ) = build_feature_and_label_cols(window_size)

    for file_path in files:
        logger.info(f"Processing {file_path.name}")
        parts = file_path.stem.split("_")
        store_cluster = int(parts[0])
        item_cluster = int(parts[1])
        logger.info(f"Store cluster: {store_cluster}")
        logger.info(f"Item cluster: {item_cluster}")
        train_all_models_for_cluster_pair(
            model_types=MODEL_TYPES,
            epochs=epochs,
            model_dir=model_dir,
            dataloader_dir=dataloader_dir,
            label_cols=label_cols,
            y_log_features=y_log_features,
            store_cluster=store_cluster,
            item_cluster=item_cluster,
            history_fn=history_fn,
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
        "--history_fn",
        type=str,
        default="",
        help="Path to save training history (relative to project root)",
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
    history_fn = (project_root / args.history_fn).resolve()
    log_dir = (project_root / args.log_dir).resolve()

    # Set up logging
    logger = setup_logging(log_dir, args.log_level)

    try:
        # Log configuration
        logger.info("Starting training with configuration:")
        logger.info(f"  Project root: {project_root}")
        logger.info(f"  Dataloader directory: {dataloader_dir}")
        logger.info(f"  Model directory: {model_dir}")
        logger.info(f"  History fn: {history_fn}")
        logger.info(f"  Window size: {args.window_size}")
        logger.info(f"  Epochs: {args.epochs}")
        torch.cuda.empty_cache()
        gc.collect()

        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

        # Train model
        train(
            dataloader_dir=dataloader_dir,
            model_dir=model_dir,
            window_size=args.window_size,
            epochs=args.epochs,
            history_fn=history_fn,
            log_level=args.log_level,
        )
        logger.info("Training completed successfully!")

    except Exception as e:
        logger.exception("Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
