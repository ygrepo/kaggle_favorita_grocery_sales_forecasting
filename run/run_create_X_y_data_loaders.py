#!/usr/bin/env python3
"""
Training script for the Favorita Grocery Sales Forecasting model.

This script handles the complete training pipeline including:
- Data loading and preprocessing
- Logging
"""

import sys
import logging
import argparse
from pathlib import Path

# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model_utils import generate_loaders
from src.data_utils import load_X_y_data, load_X_y_data
from src.utils import setup_logging


def create_data_loaders(
    window_size: int,
    data_fn: Path,
    scalers_dir: Path,
    dataloader_dir: Path,
    log_level: str,
):
    """Create features for training the model."""
    logger = logging.getLogger(__name__)
    logger.info("Starting adding data")
    df = load_X_y_data(
        data_fn=data_fn,
        window_size=window_size,
        log_level=log_level,
    )
    (_, x_sales_features, x_cyclical_features, x_feature_cols, label_cols) = (
        build_feature_and_label_cols(window_size=window_size)
    )

    generate_loaders(
        df,
        x_feature_cols,
        x_sales_features,
        x_cyclical_features,
        window_size=window_size,
        log_level=log_level,
        scalers_dir=scalers_dir,
        dataloader_dir=dataloader_dir,
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create features for Favorita Grocery Sales Forecasting model"
    )
    parser.add_argument(
        "--data_fn",
        type=str,
        default="",
        help="Path to data file (relative to project root)",
    )
    parser.add_argument(
        "--scalers_dir",
        type=str,
        default="",
        help="Path to scalers directory (relative to project root)",
    )
    parser.add_argument(
        "--dataloader_dir",
        type=str,
        default="",
        help="Path to dataloader directory (relative to project root)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=16,
        help="Size of the lookback window",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="../output/logs",
        help="Directory to save script outputs (relative to project root)",
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
    data_fn = Path(args.data_fn).resolve()
    scalers_dir = Path(args.scalers_dir).resolve()
    dataloader_dir = Path(args.dataloader_dir).resolve()
    log_dir = Path(args.log_dir).resolve()
    window_size = args.window_size

    # Set up logging
    logger = setup_logging(log_dir, args.log_level)
    logger.info(f"Log dir: {log_dir}")

    try:
        # Log configuration
        logger.info("Starting creating data loaders with configuration:")
        logger.info(f"  Data fn: {data_fn}")
        logger.info(f"  Scalers dir: {scalers_dir}")
        logger.info(f"  Dataloader dir: {dataloader_dir}")
        logger.info(f"  Window size: {window_size}")
        logger.info(f"  Log dir: {log_dir}")
        logger.info(f"  Window size: {window_size}")

        create_data_loaders(
            window_size=window_size,
            data_fn=data_fn,
            scalers_dir=scalers_dir,
            dataloader_dir=dataloader_dir,
            log_level=args.log_level,
        )

    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        raise


if __name__ == "__main__":
    main()
