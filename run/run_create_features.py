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
from scipy.linalg import dft

# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_utils import create_features
from src.utils import setup_logging, str2bool


def create_sale_cyc_features(
    window_size: int,
    add_y_targets: bool,
    log_level: str,
    sales_fn: Path,
    cyc_fn: Path,
    fn: Path,
):
    """Create features for training the model."""
    logger = logging.getLogger(__name__)
    logger.info("Starting creating features")
    create_features(
        window_size=window_size,
        add_y_targets=add_y_targets,
        sales_fn=sales_fn,
        cyc_fn=cyc_fn,
        log_level=log_level,
        output_fn=fn,
    )
    logger.info("Features created successfully")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create features for Favorita Grocery Sales Forecasting model"
    )
    parser.add_argument(
        "--sales-fn",
        type=str,
        default="",
        help="Path to sales file (relative to project root)",
    )
    parser.add_argument(
        "--cyc-fn",
        type=str,
        default="",
        help="Path to cyc file (relative to project root)",
    )
    parser.add_argument(
        "--output-fn",
        type=str,
        default="",
        help="Path to output file (relative to project root)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=16,
        help="Size of the lookback window",
    )
    parser.add_argument(
        "--add-y-targets",
        type=str2bool,
        default=False,
        help="Add y targets to the features",
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
    sales_fn = Path(args.sales_fn).resolve()
    cyc_fn = Path(args.cyc_fn).resolve()
    output_fn = Path(args.output_fn).resolve()
    log_dir = Path(args.log_dir).resolve()
    window_size = args.window_size
    add_y_targets = str2bool(args.add_y_targets)

    # Set up logging
    logger = setup_logging(log_dir, args.log_level)
    logger.info(f"Log dir: {log_dir}")

    try:
        # Log configuration
        logger.info("Starting creating training features with configuration:")
        logger.info(f"  Sales fn: {sales_fn}")
        logger.info(f"  Cyc fn: {cyc_fn}")
        logger.info(f"  Output fn: {output_fn}")
        logger.info(f"  Log dir: {log_dir}")
        logger.info(f"  Window size: {window_size}")
        logger.info(f"  Add y targets: {add_y_targets}")

        create_sale_cyc_features(
            window_size=window_size,
            add_y_targets=add_y_targets,
            log_level=args.log_level,
            sales_fn=sales_fn,
            cyc_fn=cyc_fn,
            fn=output_fn,
        )

    except Exception as e:
        logger.error(f"Error creating training features: {e}")
        raise


if __name__ == "__main__":
    main()
