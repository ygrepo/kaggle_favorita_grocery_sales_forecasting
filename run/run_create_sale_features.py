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
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_utils import load_raw_data, create_sale_features
from src.utils import setup_logging


def create_features(
    df: pd.DataFrame,
    window_size: int,
    log_level: str,
    fn: Path,
):
    """Create features for training the model."""
    logger = logging.getLogger(__name__)
    logger.info("Starting creating features")
    df = create_sale_features(
        df,
        window_size=window_size,
        calendar_aligned=True,
        fn=fn,
        log_level=log_level,
    )
    return df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create features for Favorita Grocery Sales Forecasting model"
    )
    parser.add_argument(
        "--data-fn",
        type=str,
        default="",
        help="Path to training data file (relative to project root)",
    )
    parser.add_argument(
        "--fn",
        type=str,
        default="",
        help="Path to sales file (relative to project root)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=16,
        help="Size of the lookback window",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="../output/logs",
        help="Directory to save script outputs (relative to project root)",
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
    data_fn = Path(args.data_fn).resolve()
    fn = Path(args.fn).resolve()
    log_dir = Path(args.log_dir).resolve()
    window_size = args.window_size

    # Set up logging
    print(f"Log dir: {log_dir}")
    logger = setup_logging(log_dir, args.log_level)

    try:
        # Log configuration
        logger.info("Starting creating training features with configuration:")
        logger.info(f"  Data fn: {data_fn}")
        logger.info(f"  Output fn: {fn}")
        logger.info(f"  Log dir: {log_dir}")
        logger.info(f"  Window size: {window_size}")

        # Load and preprocess data
        df = load_raw_data(data_fn)
        # store_item = "44_1503844"
        # logger.info(f"Selected store_item: {store_item}")
        # df = df[df["store_item"] == store_item]

        df = create_features(
            df,
            window_size=window_size,
            log_level=args.log_level,
            fn=fn,
        )

    except Exception as e:
        logger.error(f"Error creating training features: {e}")
        raise


if __name__ == "__main__":
    main()
