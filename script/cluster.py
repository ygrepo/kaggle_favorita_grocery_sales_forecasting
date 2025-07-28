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

from src.data_utils import load_raw_data
from src.cluster_util import cluster_data
from src.utils import setup_logging


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
        "--item_fn",
        type=str,
        default="",
        help="Path to item file (relative to project root)",
    )
    parser.add_argument(
        "--store_fn",
        type=str,
        default="",
        help="Path to store file (relative to project root)",
    )
    parser.add_argument(
        "--store_item_matrix_fn",
        type=str,
        default="",
        help="Path to store item matrix file (relative to project root)",
    )
    parser.add_argument(
        "--output_fn",
        type=str,
        default="",
        help="Path to output file (relative to project root)",
    )
    parser.add_argument(
        "--row_range",
        type=parse_range,
        default=range(2, 5),
        help="Range of number of rows to cluster (format: START:END)",
    )
    parser.add_argument(
        "--col_range",
        type=parse_range,
        default=range(2, 5),
        help="Range of number of columns to cluster (format: START:END)",
    )
    parser.add_argument(
        "--cluster_output_fn",
        type=str,
        default="",
        help="Path to cluster output file (relative to project root)",
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
    item_fn = Path(args.item_fn).resolve()
    store_fn = Path(args.store_fn).resolve()
    if args.store_item_matrix_fn:
        store_item_matrix_fn = Path(args.store_item_matrix_fn).resolve()
    else:
        store_item_matrix_fn = None
    cluster_output_fn = Path(args.cluster_output_fn).resolve()
    output_fn = Path(args.output_fn).resolve()
    row_range = args.row_range
    col_range = args.col_range

    log_dir = Path(args.log_dir).resolve()

    # Set up logging
    logger = setup_logging(log_dir, args.log_level)
    try:
        # Log configuration
        logger.info("Starting data clustering with configuration:")
        logger.info(f"  Data fn: {data_fn}")
        logger.info(f"  Item fn: {item_fn}")
        logger.info(f"  Store fn: {store_fn}")
        logger.info(f"  Store item matrix fn: {store_item_matrix_fn}")
        logger.info(f"  Cluster output fn: {cluster_output_fn}")
        logger.info(f"  Output fn: {output_fn}")

        # Load and preprocess data
        df = load_raw_data(data_fn)
        cluster_data(
            df,
            store_item_matrix_fn=store_item_matrix_fn,
            cluster_output_fn=cluster_output_fn,
            item_fn=item_fn,
            store_fn=store_fn,
            output_fn=output_fn,
            row_range=row_range,
            col_range=col_range,
        )
        logger.info("Data clustering completed successfully")
    except Exception as e:
        logger.error(f"Error clustering data: {e}")
        raise


if __name__ == "__main__":
    main()
