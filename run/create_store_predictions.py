#!/usr/bin/env python3
"""
Create store predictions script for the Favorita Grocery Sales Forecasting model.

This script handles the complete create store predictions pipeline including:
- Data loading and preprocessing
- Logging
"""

import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model_utils import (
    load_latest_models_from_checkpoints,
    batch_predict_all_store_items,
)
from src.utils import setup_logging


def create_store_predictions(
    input_dim, output_dim, checkpoints_dir=Path, 
    dataloader_dir=Path, window_size:int, log_level="INFO"
):
    logger = setup_logging("./output/logs", log_level)
    logger.info("Starting create store predictions")
    models = load_latest_models_from_checkpoints(input_dim, output_dim, log_level)
    
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
    batch_predict_all_store_items(models, checkpoints_dir, log_level)
    pass


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create store predictions for Sales Forecasting model"
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="",
        help="Path to checkpoints directory (relative to project root)",
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=12,
        help="Input dimension for model",
    )
    parser.add_argument(
        "--output_dim",
        type=int,
        default=3,
        help="Output dimension for model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Path to output directory (relative to project root)",
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
    checkpoints_dir = Path(args.checkpoints_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    log_dir = Path(args.log_dir).resolve()
    input_dim = args.input_dim
    output_dim = args.output_dim

    # Set up logging
    logger = setup_logging(log_dir, args.log_level)
    logger.info(f"Log dir: {log_dir}")

    try:
        # Log configuration
        logger.info("Starting creating store predictions with configuration:")
        logger.info(f"  Checkpoints dir: {checkpoints_dir}")
        logger.info(f"  Input dim: {input_dim}")
        logger.info(f"  Output dim: {output_dim}")
        logger.info(f"  Output dir: {output_dir}")

        create_sale_cyc_features(
            window_size=window_size,
            add_y_targets=add_y_targets,
            log_level=args.log_level,
            sales_dir=sales_dir,
            cyc_dir=cyc_dir,
            output_dir=output_dir,
        )

    except Exception as e:
        logger.error(f"Error creating training features: {e}")
        raise


if __name__ == "__main__":
    main()
