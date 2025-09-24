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

from src.utils import setup_logging, get_logger
from src.data_utils import create_sale_features, load_raw_data

logger = get_logger(__name__)


def create_features(
    data_dir: Path,
    window_size: int,
    *,
    output_dir: Path,
):
    """
    Process each Parquet file in a directory, apply feature creation,
    and save the output with a prefix.

    Parameters
    ----------
    data_dir : Path
        Path to a directory containing parquet files or a single parquet file.
    window_size : int
        Rolling window size for feature creation.
    log_level : str
        Logging level (e.g., "INFO", "DEBUG").
    prefix : str
        Prefix to use when saving processed files.
    """
    files = list(data_dir.glob("*.parquet"))

    logger.info(f"Processing {len(files)} Parquet files...")

    for file_path in files:
        logger.info(f"Processing {file_path.name}")
        df = load_raw_data(Path(file_path))
        bid = df["block_id"].unique()

        if len(bid) == 1:
            logger.info(f"Block ID: {bid[0]}")
        else:
            logger.warning(f"Multiple block IDs found: {bid}")

        out_path = output_dir / f"{file_path.stem}.parquet"
        create_sale_features(
            df,
            window_size=window_size,
            calendar_aligned=True,
            fn=out_path,
        )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create features for Favorita Grocery Sales Forecasting model"
    )
    parser.add_argument(
        "--data_dir",
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
        default=1,
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
    # Parse command line arguments
    args = parse_args()
    # Convert paths to absolute paths relative to project root
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    log_file = Path(args.log_file).resolve()
    # Set up logging
    setup_logging(log_file, args.log_level)
    window_size = args.window_size

    try:
        logger.info("Starting")
        logger.info(f"Loading data from {data_dir}")
        logger.info(f"  Output dir: {output_dir}")
        logger.info(f"  Log fn: {log_file}")
        logger.info(f"  Window size: {window_size}")

        # Load and preprocess data
        # df = load_raw_data(data_fn)
        # store_item = "44_1503844"
        # logger.info(f"Selected store_item: {store_item}")
        # df = df[df["store_item"] == store_item]

        create_features(
            data_dir,
            window_size=args.window_size,
            output_dir=output_dir,
        )
        logger.info("Completed successfully")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
