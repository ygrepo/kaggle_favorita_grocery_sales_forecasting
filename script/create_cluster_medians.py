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
import numpy as np

# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_utils import compute_cluster_medians
from src.utils import setup_logging


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
            dtype_dict = {
                "id": np.uint32,
                "store_item": str,
                "store": np.uint8,
                "item": np.uint32,
                "unit_sales": np.float32,
            }
            logger.info(f"Loading {nrows} rows")
            df = pd.read_csv(
                data_fn,
                dtype=dtype_dict,
                low_memory=False,
                parse_dates=["date"],
            )
        cols = ["date", "store_item", "store", "item", "unit_sales"] + [
            c
            for c in df.columns
            if c not in ("date", "store_item", "store", "item", "unit_sales")
        ]
        df = df[cols]
        df["date"] = pd.to_datetime(df["date"])
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
    item_fn = Path(args.item_fn).resolve()
    store_fn = Path(args.store_fn).resolve()
    output_fn = Path(args.output_fn).resolve()

    log_dir = Path(args.log_dir).resolve()

    # Set up logging
    log_level = args.log_level
    logger = setup_logging(log_dir, log_level)
    try:
        # Log configuration
        logger.info("Starting data clustering with configuration:")
        logger.info(f"  Data fn: {data_fn}")
        logger.info(f"  Item fn: {item_fn}")
        logger.info(f"  Store fn: {store_fn}")
        logger.info(f"  Output fn: {output_fn}")

        df = load_data(data_fn)
        (store_med, item_med) = compute_cluster_medians(
            df,
            store_fn=store_fn,
            item_fn=item_fn,
            log_level=log_level,
        )
        logger.info(f"Merged data with shape {df.shape}")
        logger.info(f"Unique stores: {df['store'].nunique()}")
        logger.info(f"Unique items: {df['item'].nunique()}")
        df = df.merge(store_med, on=["store_cluster", "date"], how="left").merge(
            item_med, on=["item_cluster", "date"], how="left"
        )
        if output_fn.suffix == ".parquet":
            df.to_parquet(output_fn)
        else:
            df.to_csv(output_fn, index=False)
        logger.info(f"Saved data to {output_fn}")

    except Exception as e:
        logger.error(f"Error creating training features: {e}")
        raise


if __name__ == "__main__":
    main()
