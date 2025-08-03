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

from src.model_utils import generate_sequence_model_loaders
from src.data_utils import load_X_y_data, build_feature_and_label_cols
from src.utils import setup_logging


def create_data_loaders(
    data_dir: Path,
    dataloader_dir: Path,
    log_level: str,
    window_size: int = 1,
    max_encoder_length: int = 30,
    max_prediction_length: int = 1,
    val_horizon: int = 20,
):
    """
    Create data loaders for the Favorita Grocery Sales Forecasting model.

    Parameters
    ----------
    data_dir : Path
        Directory containing the input data files.
    dataloader_dir : Path
        Directory to save the generated data loaders.
    log_level : str
        Logging level for the script.
    max_encoder_length : int
        Size of the historical window for the model.
    max_prediction_length : int
        Size of the prediction window for the model.
    val_horizon : int
        Size of the validation horizon for the model.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting data loader creation")

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
            x_sales_features,
            x_cyclical_features,
            x_feature_cols,
            x_to_log_features,
            x_log_features,
            label_cols,
            y_log_features,
            y_to_log_features,
            all_features,
        ) = build_feature_and_label_cols(window_size=window_size)
        generate_sequence_model_loaders(
            df,
            meta_cols=meta_cols,
            x_sales_features=x_sales_features,
            x_cyclical_features=x_cyclical_features,
            label_cols=label_cols,
            dataloader_dir=dataloader_dir,
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            val_horizon=val_horizon,
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
        "max_encoder_length",
        type=int,
        default=30,
        help="Size of the historical window for the model",
    )
    parser.add_argument(
        "max_prediction_length",
        type=int,
        default=1,
        help="Size of the prediction window for the model",
    )
    parser.add_argument(
        "val_horizon",
        type=int,
        default=20,
        help="Size of the validation horizon for the model",
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
    dataloader_dir = Path(args.dataloader_dir).resolve()
    log_dir = Path(args.log_dir).resolve()

    # Set up logging
    logger = setup_logging(log_dir, args.log_level)
    logger.info(f"Log dir: {log_dir}")

    try:
        # Log configuration
        logger.info("Starting creating data loaders with configuration:")
        logger.info(f"  Data dir: {data_dir}")
        logger.info(f"  Dataloader dir: {dataloader_dir}")
        logger.info(f"  Scalers dir: {scalers_dir}")
        logger.info(f"  Log dir: {log_dir}")
        logger.info(f"  Max encoder length: {args.max_encoder_length}")
        logger.info(f"  Max prediction length: {args.max_prediction_length}")
        logger.info(f"  Validation horizon: {args.val_horizon}")
        logger.info(f"  Window size: {args.window_size}")

        create_data_loaders(
            data_dir=data_dir,
            dataloader_dir=dataloader_dir,
            log_level=args.log_level,
            window_size=args.window_size,
            max_encoder_length=args.max_encoder_length,
            max_prediction_length=args.max_prediction_length,
            val_horizon=args.val_horizon,
        )

    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        raise


if __name__ == "__main__":
    main()
