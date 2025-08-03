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
from src.data_utils import load_X_y_data, build_feature_and_label_cols
from src.utils import setup_logging


def create_data_loaders(
    data_dir: Path,
    dataloader_dir: Path,
    scalers_dir: Path,
    window_size: int,
    log_level: str,
):
    """Create features for training the model."""
    logger = logging.getLogger(__name__)
    logger.info("Starting adding data")

    if data_dir.is_file() and data_dir.suffix == ".parquet":
        files = [data_dir]
    else:
        files = list(data_dir.glob("*.parquet"))

    logger.info(
        f"Processing sales (store cluster, SKU cluster) {len(files)} Parquet files..."
    )

    for file_path in files:
        logger.info(f"Processing {file_path.name}")
        parts = file_path.stem.split("_")
        store_cluster = int(parts[-2])
        item_cluster = int(parts[-1])
        logger.info(f"Store cluster: {store_cluster}")
        logger.info(f"Item cluster: {item_cluster}")
        df = load_X_y_data(
            data_fn=file_path,
            window_size=window_size,
            log_level=log_level,
        )
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        (
            meta_cols,
            _,
            x_cyclical_features,
            x_feature_cols,
            x_to_log_features,
            x_log_features,
            label_cols,
            y_log_features,
            y_to_log_features,
            all_features,
        ) = build_feature_and_label_cols(window_size=window_size)
        generate_loaders(
            df,
            all_features=all_features,
            meta_cols=meta_cols,
            x_feature_cols=x_feature_cols,
            x_to_log_features=x_to_log_features,
            x_log_features=x_log_features,
            x_cyclical_features=x_cyclical_features,
            label_cols=label_cols,
            y_log_features=y_log_features,
            y_to_log_features=y_to_log_features,
            scalers_dir=scalers_dir,
            dataloader_dir=dataloader_dir,
            log_level=log_level,
        )
        logger.info("Data loaded")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create features for Favorita Grocery Sales Forecasting model"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Path to data directory (relative to project root)",
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
    data_dir = Path(args.data_dir).resolve()
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
        logger.info(f"  Data dir: {data_dir}")
        logger.info(f"  Dataloader dir: {dataloader_dir}")
        logger.info(f"  Scalers dir: {scalers_dir}")
        logger.info(f"  Window size: {window_size}")
        logger.info(f"  Log dir: {log_dir}")

        create_data_loaders(
            window_size=window_size,
            data_dir=data_dir,
            scalers_dir=scalers_dir,
            dataloader_dir=dataloader_dir,
            log_level=args.log_level,
        )

    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        raise


if __name__ == "__main__":
    main()
