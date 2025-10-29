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

# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tensor_model import fit_and_decompose
from src.utils import (
    setup_logging,
    read_csv_or_parquet,
    get_logger,
    str2bool,
    parse_range,
)

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clustering for Retail Sales Forecasting model"
    )

    parser.add_argument(
        "--data_fn",
        type=str,
        default="",
        help="Path to training data file (relative to project root)",
    )
    parser.add_argument(
        "--rank",
        type=parse_range,
        default=range(2, 5),
        help="Rank",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=500,
        help="Maximum number of iterations",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="Tolerance for convergence",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="",
        help="Comma-separated list of feature names",
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
    log_fn = Path(args.log_fn).resolve()

    # Set up logging
    setup_logging(log_fn, args.log_level)
    try:
        # Log configuration
        logger.info("Starting data clustering with configuration:")
        logger.info(f"  Data fn: {args.data_fn}")
        ranks = args.rank.split(",")
        logger.info(f"  Rank: {ranks}")
        logger.info(f"  Max iter: {args.max_iter}")
        logger.info(f"  Tolerance: {args.tol}")
        logger.info(f"  Log level: {args.log_level}")

        data_fn = Path(args.data_fn).resolve()
        # Load and preprocess data
        df = read_csv_or_parquet(data_fn)
        fit_and_decompose(
            df, args.features, args.rank, args.max_iter, args.tol
        )

        logger.info("Data clustering completed successfully")
    except Exception as e:
        logger.error(f"Error clustering data: {e}")
        raise


if __name__ == "__main__":
    main()
