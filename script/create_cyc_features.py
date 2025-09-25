#!/usr/bin/env python3
"""
Training script for the Favorita Grocery Sales Forecasting model.

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

from src.utils import setup_logging, get_logger, save_csv_or_parquet
from src.data_utils import compute_cluster_medians

logger = get_logger(__name__)


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
        "--output_fn",
        type=str,
        default="",
        help="Path to output file (relative to project root)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=16,
        help="Size of the lookback window",
    )
    parser.add_argument(
        "--log_fn",
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


def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    # Convert paths to absolute paths relative to project root
    data_fn = Path(args.data_fn).resolve()
    output_fn = Path(args.output_fn).resolve()
    log_fn = Path(args.log_fn).resolve()
    window_size = args.window_size

    # Set up logging
    logger = setup_logging(log_dir, args.log_level)
    logger.info(f"Log dir: {log_dir}")

    try:
        # Log configuration
        logger.info("Starting creating training features with configuration:")
        logger.info(f"  Data fn: {data_fn}")
        logger.info(f"  Output dir: {output_dir}")
        logger.info(f"  Log dir: {log_dir}")
        logger.info(f"  Window size: {window_size}")

        # Load and preprocess data
        # df = load_raw_data(data_fn)
        # store_item = "44_1503844"
        # logger.info(f"Selected store_item: {store_item}")
        # df = df[df["store_item"] == store_item]

        create_features(
            data_fn,
            window_size=window_size,
            log_level=args.log_level,
            output_dir=output_dir,
        )

    except Exception as e:
        logger.error(f"Error creating cyc features: {e}")
        raise


if __name__ == "__main__":
    main()
