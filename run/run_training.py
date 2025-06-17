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
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import torch

# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model_utils import (
    train,
    ResidualMLP,
)
from src.utils import build_feature_and_label_cols


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration.

    Args:
        log_dir: Directory to save log files
        log_level: Logging level (e.g., 'INFO', 'DEBUG')

    Returns:
        Configured logger instance
    """
    # Create output directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set up log file path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return logger


def load_data(data_fn: Path) -> pd.DataFrame:
    """Load and preprocess training data.

    Args:
        data_fn: Path to the training data file

    Returns:
        Preprocessed DataFrame
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from {data_fn}")

    try:
        df = pd.read_excel(data_fn)
        logger.info(f"Loaded data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def load_weights(weights_fn: Path) -> pd.DataFrame:
    """Load and preprocess weights data.

    Args:
        weights_fn: Path to the weights data file

    Returns:
        Preprocessed DataFrame
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading weights from {weights_fn}")

    try:
        df = pd.read_excel(weights_fn)
        logger.info(f"Loaded weights with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading weights: {e}")
        raise


def train_model(
    df: pd.DataFrame,
    weights_df: pd.DataFrame,
    model_dir: Path,
    window_size: int = 16,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    epochs: int = 100,
    seed: int = 42,
) -> None:
    """Train the forecasting model.

    Args:
        df: Input DataFrame with training data
        output_dir: Directory to save outputs
        model_dir: Directory to save trained models
        window_size: Size of the lookback window
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        seed: Random seed for reproducibility
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model training")

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build feature and label columns
    (
        meta_cols,
        x_sales_features,
        x_cyclical_features,
        x_feature_cols,
        label_cols,
        y_sales_features,
        y_cyclical_features,
    ) = build_feature_and_label_cols(window_size)

    logger.info(
        f"Training with {len(x_feature_cols)} features and {len(label_cols)} labels"
    )
    # Ensure output directories exist
    model_dir.mkdir(parents=True, exist_ok=True)

    # Train the model
    train(
        df=df,
        weights_df=weights_df,
        x_feature_cols=x_feature_cols,
        x_sales_features=x_sales_features,
        x_cyclical_features=x_cyclical_features,
        label_cols=label_cols,
        y_sales_features=y_sales_features,
        y_cyclical_features=y_cyclical_features,
        item_col="item",
        train_frac=0.8,
        batch_size=batch_size,
        lr=learning_rate,
        epochs=epochs,
        seed=seed,
        model_cls=ResidualMLP,
        model_dir=str(model_dir),
    )

    logger.info("Training completed successfully")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Favorita Grocery Sales Forecasting model"
    )
    parser.add_argument(
        "--data-fn",
        type=str,
        default="./output/data/20250611_train_top_10_store_10_item_sales_cyclical_features_16_days_X_y.xlsx",
        help="Path to training data file (relative to project root)",
    )
    parser.add_argument(
        "--weights-fn",
        type=str,
        default="./output/data/top_10_item_weights.xlsx",
        help="Path to weights file (relative to project root)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./output/logs",
        help="Directory to save training outputs (relative to project root)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./output/models",
        help="Directory to save trained models (relative to project root)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=16,
        help="Size of the lookback window",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--log-level",
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
    data_fn = (project_root / args.data_fn).resolve()
    weights_fn = (project_root / args.weights_fn).resolve()
    model_dir = (project_root / args.model_dir).resolve()
    log_dir = (project_root / args.log_dir).resolve()

    # Set up logging
    logger = setup_logging(log_dir, args.log_level)

    try:
        # Log configuration
        logger.info("Starting training with configuration:")
        logger.info(f"  Project root: {project_root}")
        logger.info(f"  Data fn: {data_fn}")
        logger.info(f"  Weights fn: {weights_fn}")
        logger.info(f"  Model directory: {model_dir}")
        logger.info(f"  Window size: {args.window_size}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Learning rate: {args.learning_rate}")
        logger.info(f"  Epochs: {args.epochs}")
        logger.info(f"  Random seed: {args.seed}")

        # Load and preprocess data
        df = load_data(data_fn)

        # Load weights
        weights_df = load_weights(weights_fn)

        # Select top 10 store_items
        store_item = "44_364606"
        logger.info(f"Selected store_item: {store_item}")

        # Train model
        train_model(
            df=df.query("store_item == @store_item").reset_index(drop=True),
            weights_df=weights_df,
            model_dir=model_dir,
            window_size=args.window_size,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            seed=args.seed,
        )

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.exception("Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
