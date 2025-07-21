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


# def create_features(
#     df: pd.DataFrame,
#     window_size: int,
#     log_level: str,
#     fn: Path,
# ):
#     """Create features for training the model."""
#     logger = logging.getLogger(__name__)
#     logger.info("Starting creating features")
#     df = create_sale_features(
#         df,
#         window_size=window_size,
#         calendar_aligned=True,
#         fn=fn,
#         log_level=log_level,
#     )
#     return df


def create_features(
    data_fn: Path,
    window_size: int,
    *,
    output_dir: Path,
    prefix: str = "sale_features",
    log_level: str = "INFO",
):
    """
    Process each Parquet file in a directory, apply feature creation,
    and save the output with a prefix.

    Parameters
    ----------
    data_fn : Path
        Path to a directory containing parquet files or a single parquet file.
    window_size : int
        Rolling window size for feature creation.
    log_level : str
        Logging level (e.g., "INFO", "DEBUG").
    prefix : str
        Prefix to use when saving processed files.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if data_fn.is_file() and data_fn.suffix == ".parquet":
        files = [data_fn]
    else:
        files = list(data_fn.glob("*.parquet"))

    logger.info(f"Processing {len(files)} Parquet files...")

    for file_path in files:
        logger.info(f"Processing {file_path.name}")
        df = load_raw_data(Path(file_path))

        df = create_sale_features(
            df,
            window_size=window_size,
            calendar_aligned=True,
            fn=file_path,
            log_level=log_level,
        )

        out_path = output_dir / f"{prefix}_{file_path.stem}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved: {out_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create features for Favorita Grocery Sales Forecasting model"
    )
    parser.add_argument(
        "--data_fn",
        type=str,
        default="",
        help="Path to training data directory (relative to project root)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Path to sales files directory (relative to project root)",
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
    # Convert paths to absolute paths relative to project root
    data_fn = Path(args.data_fn).resolve()
    output_dir = Path(args.output_dir).resolve()
    log_dir = Path(args.log_dir).resolve()
    window_size = args.window_size

    # Set up logging
    print(f"Log dir: {log_dir}")
    logger = setup_logging(log_dir, args.log_level)

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
        logger.error(f"Error creating training features: {e}")
        raise


if __name__ == "__main__":
    main()
