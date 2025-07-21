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


# def create_features(
#     sales_dir: Path,
#     cyc_dir: Path,
#     window_size: int,
#     *,
#     output_dir: Path,
#     prefix: str = "sale_cyc_features",
#     log_level: str = "INFO",
# ):
#     """
#     Process each Parquet file in a directory, apply feature creation,
#     and save the output with a prefix.

#     Parameters
#     ----------
#     sales_dir : Path
#         Path to a directory containing parquet files or a single parquet file.
#     cyc_dir : Path
#         Path to a directory containing parquet files or a single parquet file.
#     window_size : int
#         Rolling window size for feature creation.
#     log_level : str
#         Logging level (e.g., "INFO", "DEBUG").
#     prefix : str
#         Prefix to use when saving processed files.
#     """
#     logger = logging.getLogger(__name__)
#     logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

#     if sales_dir.is_file() and sales_dir.suffix == ".parquet":
#         files = [sales_dir]
#     else:
#         files = list(sales_dir.glob("*.parquet"))

#     logger.info(f"Processing {len(files)} Parquet files...")

#     for file_path in files:
#         logger.info(f"Processing {file_path.name}")
#         create_features(
#         window_size=window_size,
#         add_y_targets=add_y_targets,
#         sales_fn=sales_fn,
#         cyc_fn=cyc_fn,
#         log_level=log_level,
#         output_fn=fn,
#     )

#         df = load_raw_data(Path(file_path))
#         logger.info(f"{df['store_item'].unique()}")

#         out_path = output_dir / f"{prefix}_{file_path.stem}.parquet"
#         create_sale_features(
#             df,
#             window_size=window_size,
#             calendar_aligned=True,
#             fn=out_path,
#             log_level=log_level,
#         )


def create_sale_cyc_features(
    window_size: int,
    add_y_targets: bool,
    log_level: str,
    sales_dir: Path,
    cyc_dir: Path,
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
        "--sales_dir",
        type=str,
        default="",
        help="Path to sales directory (relative to project root)",
    )
    parser.add_argument(
        "--cyc_dir",
        type=str,
        default="",
        help="Path to cyc directory (relative to project root)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Path to output directory (relative to project root)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=16,
        help="Size of the lookback window",
    )
    parser.add_argument(
        "--add_y_targets",
        type=str2bool,
        default=False,
        help="Add y targets to the features",
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
    sales_dir = Path(args.sales_dir).resolve()
    cyc_dir = Path(args.cyc_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    log_dir = Path(args.log_dir).resolve()
    window_size = args.window_size
    add_y_targets = str2bool(args.add_y_targets)

    # Set up logging
    logger = setup_logging(log_dir, args.log_level)
    logger.info(f"Log dir: {log_dir}")

    try:
        # Log configuration
        logger.info("Starting creating training features with configuration:")
        logger.info(f"  Sales dir: {sales_dir}")
        logger.info(f"  Cyc dir: {cyc_dir}")
        logger.info(f"  Output dir: {output_dir}")
        logger.info(f"  Window size: {window_size}")
        logger.info(f"  Add y targets: {add_y_targets}")

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
