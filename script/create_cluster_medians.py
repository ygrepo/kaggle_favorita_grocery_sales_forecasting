#!/usr/bin/env python3
"""
Create cluster medians script for the Favorita Grocery Sales Forecasting model.

This script handles the complete create cluster medians pipeline including:
- Data loading and preprocessing
- Logging
"""

import sys
import logging
import argparse
from pathlib import Path

import pandas as pd


# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import setup_logging, get_logger, save_csv_or_parquet
from src.data_utils import compute_cluster_medians

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create cluster medians for Shop Sales Forecasting model"
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


def load_data(
    data_fn: Path,
) -> pd.DataFrame:
    """Load and preprocess training data.

    Args:
        data_fn: Path to the training data file

    Returns:
        Preprocessed DataFrame
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from {data_fn}")

    try:
        if data_fn.suffix == ".parquet":
            df = pd.read_parquet(data_fn)
        else:
            logger.info("Loading")
            df = pd.read_csv(
                data_fn,
                low_memory=False,
            )
        df["store_item"] = df["store"].astype(str) + "_" + df["item"].astype(str)
        df.sort_values(["store_item", "date"], inplace=True)
        logger.info(f"Loaded data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    # Convert paths to absolute paths relative to project root
    data_fn = Path(args.data_fn).resolve()
    output_fn = Path(args.output_fn).resolve()
    log_fn = Path(args.log_fn).resolve()
    # Set up logging
    setup_logging(log_fn, args.log_level)
    try:
        # Log configuration
        logger.info("Starting:")
        logger.info(f"  Data fn: {data_fn}")
        logger.info(f"  Output fn: {output_fn}")
        logger.info(f"  Log fn: {log_fn}")

        df = load_data(data_fn)
        med_df = compute_cluster_medians(
            df, date_col="date", cluster_col="cluster_id", value_col="growth_rate"
        )
        logger.info(f"Unique stores: {df['store'].nunique()}")
        logger.info(f"Unique items: {df['item'].nunique()}")
        df = df.merge(med_df, on=["cluster_id", "date"], how="left")
        logger.info(f"Unique stores: {df['store'].nunique()}")
        logger.info(f"Unique items: {df['item'].nunique()}")
        med_df = compute_cluster_medians(
            df, date_col="date", cluster_col="cluster_id", value_col="sales"
        )
        logger.info(f"Unique stores: {df['store'].nunique()}")
        logger.info(f"Unique items: {df['item'].nunique()}")
        df = df.merge(med_df, on=["cluster_id", "date"], how="left")
        logger.info(f"Unique stores: {df['store'].nunique()}")
        logger.info(f"Unique items: {df['item'].nunique()}")
        save_csv_or_parquet(df, output_fn)
        logger.info("Completed successfully")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
