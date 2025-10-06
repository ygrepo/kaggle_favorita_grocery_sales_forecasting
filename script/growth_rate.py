#!/usr/bin/env python3
"""
Clustering script for the Favorita Grocery Sales Forecasting model.

This script handles the complete clustering pipeline including:
- Data loading and preprocessing
- Logging
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_utils import (
    load_raw_data,
    save_csv_or_parquet,
    make_weekly_growth,
    fit_three_state_representatives,
)
from src.utils import setup_logging, get_logger

logger = get_logger(__name__)


def parse_range(range_str):
    try:
        start, end = map(int, range_str.split(":"))
        return range(start, end)
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Invalid range format: {range_str}. Use START:END"
        ) from e


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clustering for Favorita Grocery Sales Forecasting model"
    )

    parser.add_argument(
        "--data_fn",
        type=str,
        default="",
        help="Path to training data file (relative to project root)",
    )
    parser.add_argument(
        "--output_fn",
        type=str,
        default="",
        help="Path to output file (relative to project root)",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel processes to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for multiprocessing",
    )
    parser.add_argument(
        "--log_fn",
        type=str,
        default="",
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
    setup_logging(Path(args.log_fn), args.log_level)

    try:
        # Log configuration
        logger.info("Starting with configuration:")
        logger.info(f"  Data fn: {args.data_fn}")
        logger.info(f"  Output fn: {args.output_fn}")
        logger.info(f"  N jobs: {args.n_jobs}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Log fn: {args.log_fn}")
        logger.info(f"  Log level: {args.log_level}")

        data_fn = Path(args.data_fn).resolve()

        df = load_raw_data(data_fn)

        output_fn = Path(args.output_fn).resolve()
        df["unit_sales"] = df["unit_sales"].astype(float)
        df = make_weekly_growth(df)
        # fit_three_state_representatives(df)
        save_csv_or_parquet(df, output_fn)
        logger.info("Completed successfully")
    except Exception as e:
        logger.error(f"Error Generating growth rate data: {e}")
        raise


if __name__ == "__main__":
    main()
