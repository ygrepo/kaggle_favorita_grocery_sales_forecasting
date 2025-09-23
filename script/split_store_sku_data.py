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

from src.utils import setup_logging, get_logger
from src.data_utils import load_raw_data, split_by_block_id

logger = get_logger(__name__)


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
        "--log_file",
        type=str,
        default="",
        help="Path to save script outputs (relative to project root)",
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
    # Parse command line arguments
    args = parse_args()
    # Convert paths to absolute paths relative to project root
    data_fn = Path(args.data_fn).resolve()
    output_dir = Path(args.output_dir).resolve()
    log_file = Path(args.log_file).resolve()
    # Set up logging
    setup_logging(log_file, args.log_level)

    try:
        logger.info("Starting")
        logger.info(f"Loading data from {data_fn}")
        logger.info(f"  Data fn: {data_fn}")
        logger.info(f"  Output dir: {output_dir}")
        logger.info(f"  Log fn: {log_file}")

        df = load_raw_data(
            data_fn=data_fn,
        )
        split_by_block_id(df, output_dir=output_dir)
        logger.info("Completed successfully")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
