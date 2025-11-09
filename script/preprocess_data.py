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
from src.data_preprocessing import (
    prepare_data,
    select_extreme_and_median_neighbors,
)
from src.utils import setup_logging, get_logger, save_csv_or_parquet

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Data preprocessing for Favorita Grocery Sales Forecasting model"
    )
    parser.add_argument(
        "--data_fn",
        type=str,
        default="data/train.csv",
        help="Path to training data file (relative to project root)",
    )
    parser.add_argument(
        "--item_fn",
        type=str,
        default="data/items.csv",
        help="Path to item file (relative to project root)",
    )
    parser.add_argument(
        "--store_fn",
        type=str,
        default="data/stores.csv",
        help="Path to store file (relative to project root)",
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
    setup_logging(Path(args.log_fn), args.log_level)

    # Parse command line arguments
    args = parse_args()
    # Convert paths to absolute paths relative to project root
    data_fn = Path(args.data_fn).resolve()
    output_fn = Path(args.output_fn).resolve()
    item_fn = Path(args.item_fn).resolve()
    store_fn = Path(args.store_fn).resolve()

    try:
        # Log configuration
        logger.info("Starting data preprocessing with configuration:")
        logger.info(f"  Data fn: {data_fn}")
        logger.info(f"  Log fn: {args.log_fn}")
        logger.info(f"  Log level: {args.log_level}")
        logger.info(f"  Nrows: {args.nrows}")
        logger.info(f"  Start date: {args.start_date}")
        logger.info(f"  End date: {args.end_date}")
        logger.info(f"  Output fn: {output_fn}")
        logger.info(f"  Store top n: {args.store_top_n}")
        logger.info(f"  Store med n: {args.store_med_n}")
        logger.info(f"  Store bottom n: {args.store_bottom_n}")
        logger.info(f"  Store fn: { store_fn}")
        logger.info(f"  Item top n: {args.item_top_n}")
        logger.info(f"  Item med n: {args.item_med_n}")
        logger.info(f"  Item bottom n: {args.item_bottom_n}")
        logger.info(f"  Item fn: {item_fn}")

        # Load and preprocess data
        data_df = pd.read_csv(data_fn, low_memory=False)
        data_df.rename(
            columns={"store_nbr": "store", "item_nbr": "item"}, inplace=True
        )
        data_df.drop(["id", "onpromotion"], axis=1, inplace=True)
        data_df["date"] = pd.to_datetime(data_df["date"])
        data_df = data_df[
            (data_df["date"] >= args.start_date)
            & (data_df["date"] <= args.end_date)
        ]
        logger.info(f"Stores: {data_df['store'].nunique()}")
        ids = select_extreme_and_median_neighbors(
            data_df,
            n_col="unit_sales",
            group_column="store",
            M=args.store_top_n,
            m=args.store_bottom_n,
            med=args.store_med_n,
        )
        data_df = data_df[data_df["store"].isin(ids)]
        logger.info(f"Stores: {len(ids)}")
        logger.info(f"Items: {data_df['item'].nunique()}")
        ids = select_extreme_and_median_neighbors(
            data_df,
            n_col="unit_sales",
            group_column="item",
            M=args.item_top_n,
            m=args.item_bottom_n,
            med=args.item_med_n,
        )

        logger.info(f"Items: {len(ids)}")
        data_df = data_df[data_df["item"].isin(ids)]
        df = pd.read_csv("./data/transactions.csv")
        df.rename(columns={"store_nbr": "store"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        data_df = data_df.merge(df, on=["store", "date"], how="left")

        df = pd.read_csv("./data/stores.csv")
        df.rename(columns={"store_nbr": "store"}, inplace=True)
        df.drop(["city", "state"], axis=1, inplace=True)
        type_encoded = pd.get_dummies(
            df["type"], prefix="type", drop_first=True
        ).astype(int)
        df = pd.concat([df.drop("type", axis=1), type_encoded], axis=1)
        logger.info(f"Initial Stores: {df['store'].nunique()}")
        data_df = data_df.merge(df, on=["store"], how="left")
        logger.info(f"Stores: {data_df['store'].nunique()}")
        logger.info(f"Items: {data_df['item'].nunique()}")
        df = pd.read_csv("./data/items.csv")
        df.rename(columns={"item_nbr": "item"}, inplace=True)
        df = df[["item", "perishable"]]
        data_df = data_df.merge(df, on=["item"], how="left")

        data_df["store_item"] = (
            data_df["store"].astype(str) + "_" + data_df["item"].astype(str)
        )
        data_df.sort_values(["date", "store_item"], inplace=True)
        data_df.reset_index(drop=True, inplace=True)

        save_csv_or_parquet(data_df, output_fn)
        logger.info("Data preprocessing completed successfully")
    except Exception as e:
        logger.error(f"Error creating data preprocessing features: {e}")
        raise


if __name__ == "__main__":
    main()
