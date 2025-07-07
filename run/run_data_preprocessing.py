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
        "--top-stores-n",
        type=int,
        default=10,
        help="Number of top stores to return",
    )
    parser.add_argument(
        "--top-items-n",
        type=int,
        default=500,
        help="Number of top items to return",
    )
    parser.add_argument(
        "--group-column",
        type=str,
        default="store",
        help="Column to group by",
    )
    parser.add_argument(
        "--value-column",
        type=str,
        default="unit_sales",
        help="Column to calculate percentages from",
    )
    return parser.parse_args()


def load_data(data_fn: Path) -> pd.DataFrame:
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
        df = pd.read_csv(data_fn, dtype=dtype_dict, low_memory=False)
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
        logger.info(f"  Output fn: {output_fn}")

        # Load and preprocess data
        df = load_data(data_fn)

        # Create features
        df = prepare_data(
            df,
            group_column=args.group_column,
            value_column=args.value_column,
            top_stores_n=args.top_stores_n,
            top_items_n=args.top_items_n,
            fn=output_fn,
        )

        logger.info("Data preprocessing completed successfully")
    except Exception as e:
        logger.error(f"Error creating training features: {e}")
        raise


if __name__ == "__main__":
    main()
