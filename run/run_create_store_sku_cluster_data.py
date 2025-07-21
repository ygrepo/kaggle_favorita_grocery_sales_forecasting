#!/usr/bin/env python3
"""
Create store SKU cluster data script for the Favorita Grocery Sales Forecasting model.

This script handles the complete training pipeline including:
- Data loading and preprocessing
- Logging
"""

import sys
import argparse
from pathlib import Path

# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_utils import load_raw_data, save_parquets_by_cluster_pairs
from src.utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create store SKU cluster data for Favorita Grocery Sales Forecasting model"
    )
    parser.add_argument(
        "--data_fn",
        type=str,
        default="",
        help="Path to training data file (relative to project root)",
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


if __name__ == "__main__":
    args = parse_args()
    data_fn = Path(args.data_fn).resolve()
    output_fn = Path(args.output_fn).resolve()
    log_dir = Path(args.log_dir).resolve()

    try:
        logger = setup_logging(log_dir=log_dir, log_level=args.log_level)
        logger.info(f"Starting create_store_sku_cluster_data")
        logger.info(f"Loading data from {data_fn}")
        logger.info(f"  Data fn: {data_fn}")
        logger.info(f"  Output dir: {output_dir}")
        logger.info(f"  Log dir: {log_dir}")

        df = load_raw_data(
            data_fn=data_fn,
            log_level=args.log_level,
        )
        save_parquets_by_cluster_pairs(
            df,
            output_fn=output_fn,
            log_level=args.log_level,
        )
        logger.info(f"Finished create_store_sku_cluster_data")
    except Exception as e:
        logger.error(f"Error creating store SKU cluster data: {e}")
        raise
