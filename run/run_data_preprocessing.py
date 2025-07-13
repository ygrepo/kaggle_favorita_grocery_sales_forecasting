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


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration.

    Args:
        log_dir: Directory to save log files
        log_level: Logging level (e.g., 'INFO', 'DEBUG')

    Returns:
        Configured logger instance
    """
    # Create output directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set up log file path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"create_features_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Data preprocessing for Favorita Grocery Sales Forecasting model"
    )
    parser.add_argument(
        "--data-fn",
        type=str,
        default="",
        help="Path to training data file (relative to project root)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="../output/logs",
        help="Directory to save script outputs (relative to project root)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--output-fn",
        type=str,
        default="",
        help="Path to output file (relative to project root)",
    )
    parser.add_argument(
        "--store-top-n",
        type=int,
        default=0,
        help="Number of top stores to return",
    )
    parser.add_argument(
        "--store-med-n",
        type=int,
        default=0,
        help="Number of top stores to return",
    )
    parser.add_argument(
        "--store-bottom-n",
        type=int,
        default=0,
        help="Number of bottom stores to return",
    )
    parser.add_argument(
        "--store-fn",
        type=str,
        default="",
        help="Path to store file (relative to project root)",
    )
    parser.add_argument(
        "--item-top-n",
        type=int,
        default=0,
        help="Number of top items to return",
    )
    parser.add_argument(
        "--item-med-n",
        type=int,
        default=0,
        help="Number of top items to return",
    )
    parser.add_argument(
        "--item-bottom-n",
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
        "--date",
        type=str,
        default="",
        help="Date to cap on",
    )
    parser.add_argument(
        "--item-fn",
        type=str,
        default="",
        help="Path to item file (relative to project root)",
    )
    parser.add_argument(
        "--group-store-column",
        type=str,
        default="store",
        help="Column to group by",
    )
    parser.add_argument(
        "--group-item-column",
        type=str,
        default="item",
        help="Column to group by",
    )
    parser.add_argument(
        "--value-column",
        type=str,
        default="unit_sales",
        help="Column to calculate percentages from",
    )
    return parser.parse_args()


def load_data(data_fn: Path, nrows: int = 0, date: str = "") -> pd.DataFrame:
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
        if date:
            logger.info(f"Filtering data for date {date}")
            logger.info(f"Before filtering:{df.shape}")
            df = df[df["date"] >= date]
            logger.info(f"After filtering:{df.shape}")
        cols = ["date", "store_item", "store", "item", "unit_sales"] + [
            c
            for c in df.columns
            if c not in ("date", "store_item", "store", "item", "unit_sales")
        ]
        df = df[cols]
        df["date"] = pd.to_datetime(df["date"])
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
    log_dir = Path(args.log_dir).resolve()
    output_fn = Path(args.output_fn).resolve()
    # Set up logging
    logger = setup_logging(log_dir, args.log_level)

    try:
        # Log configuration
        logger.info("Starting data preprocessing with configuration:")
        logger.info(f"  Data fn: {data_fn}")
        logger.info(f"  Log dir: {log_dir}")
        logger.info(f"  Log level: {args.log_level}")
        logger.info(f"  Nrows: {args.nrows}")
        logger.info(f"  Date: {args.date}")
        logger.info(f"  Output fn: {output_fn}")
        logger.info(f"  Store top n: {args.store_top_n}")
        logger.info(f"  Store med n: {args.store_med_n}")
        logger.info(f"  Store bottom n: {args.store_bottom_n}")
        logger.info(f"  Store fn: {args.store_fn}")
        logger.info(f"  Item top n: {args.item_top_n}")
        logger.info(f"  Item med n: {args.item_med_n}")
        logger.info(f"  Item bottom n: {args.item_bottom_n}")
        logger.info(f"  Item fn: {args.item_fn}")
        logger.info(f"  Group store column: {args.group_store_column}")
        logger.info(f"  Group item column: {args.group_item_column}")
        logger.info(f"  Value column: {args.value_column}")

        # Load and preprocess data
        df = load_data(data_fn, nrows=args.nrows, date=args.date)
        # store_item = "44_1503844"
        # logger.info(f"Selected store_item: {store_item}")
        # df = df[df["store_item"] == store_item]
        # df.to_csv("./output/data/20250711_train_44_1503844.csv", index=False)

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
            item_fn=args.item_fn,
            store_fn=args.store_fn,
            fn=output_fn,
        )

        logger.info("Data preprocessing completed successfully")
    except Exception as e:
        logger.error(f"Error creating data preprocessing features: {e}")
        raise


if __name__ == "__main__":
    main()
