#!/usr/bin/env python3
"""
Data preprocessing script for the Favorita Grocery Sales Forecasting model.

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
from src.data_preprocessing import prepare_data
from src.utils import setup_logging, get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Data preprocessing for Favorita Grocery Sales Forecasting model"
    )
    parser.add_argument(
        "--data_fn",
        type=str,
        default="",
        help="Path to training data file (relative to project root)",
    )
    parser.add_argument(
        "--weights_fn",
        type=str,
        default="",
        help="Path to weights file (relative to project root)",
    )
    parser.add_argument(
        "--filtered_data_fn",
        type=str,
        default="",
        help="Path to filtered data file (relative to project root)",
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
    parser.add_argument(
        "--output_fn",
        type=str,
        default="",
        help="Path to output file (relative to project root)",
    )
    parser.add_argument(
        "--store_top_n",
        type=int,
        default=0,
        help="Number of top stores to return",
    )
    parser.add_argument(
        "--store_med_n",
        type=int,
        default=0,
        help="Number of top stores to return",
    )
    parser.add_argument(
        "--store_bottom_n",
        type=int,
        default=0,
        help="Number of bottom stores to return",
    )
    parser.add_argument(
        "--store_fn",
        type=str,
        default="",
        help="Path to store file (relative to project root)",
    )
    parser.add_argument(
        "--item_top_n",
        type=int,
        default=0,
        help="Number of top items to return",
    )
    parser.add_argument(
        "--item_med_n",
        type=int,
        default=0,
        help="Number of top items to return",
    )
    parser.add_argument(
        "--item_bottom_n",
        type=int,
        default=0,
        help="Number of bottom items to return",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=0,
        help="Number of rows to load",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="",
        help="Date to cap on",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="",
        help="Date to cap on",
    )
    parser.add_argument(
        "--item_fn",
        type=str,
        default="",
        help="Path to item file (relative to project root)",
    )
    parser.add_argument(
        "--group_store_column",
        type=str,
        default="store",
        help="Column to group by",
    )
    parser.add_argument(
        "--group_item_column",
        type=str,
        default="item",
        help="Column to group by",
    )
    parser.add_argument(
        "--value_column",
        type=str,
        default="unit_sales",
        help="Column to calculate percentages from",
    )
    return parser.parse_args()


def load_data(
    data_fn: Path,
    nrows: int = 0,
    start_date: str = "",
    end_date: str = "",
    fn: Path = None,
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
        dtype_dict = {
            "id": np.uint32,
            "store_item": str,
            "store": np.uint8,
            "item": np.uint32,
            "unit_sales": np.float32,
        }
        if nrows > 0:
            logger.info(f"Loading {nrows} rows")
            df = pd.read_csv(
                data_fn,
                dtype=dtype_dict,
                low_memory=False,
                nrows=nrows,
                parse_dates=["date"],
            )
        else:
            df = pd.read_csv(
                data_fn, dtype=dtype_dict, low_memory=False, parse_dates=["date"]
            )
        initial_stores = df["store"].unique().tolist()
        initial_items = df["item"].unique().tolist()
        if start_date:
            logger.info(f"Filtering data for start date {start_date}")
            logger.info(f"Before start date filtering: {df.shape}")
            df = df[df["date"] >= start_date]
            logger.info(f"After start date filtering: {df.shape}")
        if end_date:
            logger.info(f"Filtering data for end date {end_date}")
            logger.info(f"Before end date filtering: {df.shape}")
            df = df[df["date"] <= end_date]
            logger.info(f"After end date filtering: {df.shape}")
        cols = ["date", "store_item", "store", "item", "unit_sales"] + [
            c
            for c in df.columns
            if c not in ("date", "store_item", "store", "item", "unit_sales")
        ]
        df = df[cols]
        df["date"] = pd.to_datetime(df["date"])
        intersect_stores = np.intersect1d(initial_stores, df["store"].unique())
        intersect_items = np.intersect1d(initial_items, df["item"].unique())
        logger.info(f"# intersect stores: {len(intersect_stores)}")
        logger.info(f"# intersect items: {len(intersect_items)}")
        missing_stores = np.setdiff1d(initial_stores, df["store"].unique())
        missing_items = np.setdiff1d(initial_items, df["item"].unique())
        logger.info(f"# missing stores: {len(missing_stores)}")
        logger.info(f"# missing items: {len(missing_items)}")
        if fn:
            logger.info(f"Saving data to {fn}")
            df.to_csv(fn, index=False)
        logger.info(f"Loaded data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def load_weights(
    weights_fn: Path,
) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    logger.info(f"Loading weights from {weights_fn}")

    try:
        dtype_dict = {
            "item_nbr": np.uint32,
            "family": str,
            "class": str,
            "perishable": np.float32,
        }
        df = pd.read_csv(
            weights_fn,
            dtype=dtype_dict,
            low_memory=False,
        )
        df.rename(columns={"item_nbr": "item", "perishable": "weight"}, inplace=True)
        df = df[["item", "weight"]]
        df["weight"] = df["weight"].fillna(1)

        return df
    except Exception as e:
        logger.error(f"Error loading weights: {e}")
        raise


def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    setup_logging(Path(args.log_fn), args.log_level)

    # Parse command line arguments
    args = parse_args()
    # Convert paths to absolute paths relative to project root
    data_fn = Path(args.data_fn).resolve()
    output_fn = Path(args.output_fn).resolve()
    item_fn = Path(args.item_fn).resolve()
    store_fn = Path(args.store_fn).resolve()
    # filtered_data_fn = Path(args.filtered_data_fn).resolve()
    weights_fn = Path(args.weights_fn).resolve()

    try:
        # Log configuration
        logger.info("Starting data preprocessing with configuration:")
        logger.info(f"  Data fn: {data_fn}")
        logger.info(f"  Weights fn: {weights_fn}")
        logger.info(f"  Log fn: {args.log_fn}")
        logger.info(f"  Log level: {args.log_level}")
        logger.info(f"  Nrows: {args.nrows}")
        logger.info(f"  Start date: {args.start_date}")
        logger.info(f"  End date: {args.end_date}")
        logger.info(f"  Output fn: {output_fn}")
        # logger.info(f"  Filtered data fn: {filtered_data_fn}")
        logger.info(f"  Store top n: {args.store_top_n}")
        logger.info(f"  Store med n: {args.store_med_n}")
        logger.info(f"  Store bottom n: {args.store_bottom_n}")
        logger.info(f"  Store fn: { store_fn}")
        logger.info(f"  Item top n: {args.item_top_n}")
        logger.info(f"  Item med n: {args.item_med_n}")
        logger.info(f"  Item bottom n: {args.item_bottom_n}")
        logger.info(f"  Item fn: {item_fn}")
        logger.info(f"  Group store column: {args.group_store_column}")
        logger.info(f"  Group item column: {args.group_item_column}")
        logger.info(f"  Value column: {args.value_column}")

        # Load and preprocess data
        df = load_data(
            data_fn,
            nrows=args.nrows,
            start_date=args.start_date,
            end_date=args.end_date,
            fn=None,
        )
        # store_item = "44_1503844"
        # logger.info(f"Selected store_item: {store_item}")
        # df = df[df["store_item"] == store_item]
        # df.to_csv("./output/data/20250711_train_44_1503844.csv", index=False)

        # Merge with weights
        w_df = load_weights(weights_fn)
        w_df = w_df[["item", "weight"]]
        df = pd.merge(df, w_df, on=["item"], how="left")
        df["weight"] = df["weight"].fillna(1)

        # Create features
        df = prepare_data(
            df,
            group_store_column=args.group_store_column,
            group_item_column=args.group_item_column,
            value_column=args.value_column,
            store_top_n=args.store_top_n,
            store_med_n=args.store_med_n,
            store_bottom_n=args.store_bottom_n,
            item_top_n=args.item_top_n,
            item_med_n=args.item_med_n,
            item_bottom_n=args.item_bottom_n,
            item_fn=item_fn,
            store_fn=store_fn,
            fn=output_fn,
        )

        logger.info("Data preprocessing completed successfully")
    except Exception as e:
        logger.error(f"Error creating data preprocessing features: {e}")
        raise


if __name__ == "__main__":
    main()
