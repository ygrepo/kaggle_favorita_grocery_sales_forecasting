#!/usr/bin/env python3
"""
Clustering script for the Favorita Grocery Sales Forecasting model.

This script handles the complete clustering pipeline including:
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

from src.cluster_util import cluster_data


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
        if data_fn.suffix == ".parquet":
            df = pd.read_parquet(data_fn)
        else:
            dtype_dict = {
                "store": "uint16",
                "item": "uint32",
                "store_item": "string",  # allow NaNs as <NA>
                "unit_sales": "float32",
                "id": "Int64",  # nullable integer
                "onpromotion": "boolean",  # if you want True/False with nulls
            }
            df = pd.read_csv(
                data_fn,
                dtype=dtype_dict,
                parse_dates=["date"],
                keep_default_na=True,
                na_values=[""],
            )
        # Convert nullable Int64 or boolean to float64 with NaN
        cols = ["date", "store_item", "store", "item"] + [
            c for c in df.columns if c not in ("date", "store_item", "store", "item")
        ]
        df = df[cols]
        # df["id"] = df["id"].astype("float64")  # <NA> → np.nan
        # df["id"] = df["id"].astype(object).where(df["id"].notna(), np.nan)
        df["store_item"] = (
            df["store_item"].astype(object).where(df["store_item"].notna(), np.nan)
        )
        df["onpromotion"] = (
            df["onpromotion"].astype(object).where(df["onpromotion"].notna(), np.nan)
        )
        df["date"] = pd.to_datetime(df["date"])
        logger.info(f"Loaded data with shape {df.shape}")
        df.fillna(0, inplace=True)
        logger.info(f"Filled NaN values with 0")
        # df = df[df["unit_sales"].notna()]
        # logger.info(f"Dropped rows with NaN unit_sales, new shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


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
        "--store-item-matrix-fn",
        type=str,
        default="",
        help="Path to store item matrix file (relative to project root)",
    )
    parser.add_argument(
        "--output-fn",
        type=str,
        default="",
        help="Path to output file (relative to project root)",
    )
    parser.add_argument(
        "--row-range",
        type=parse_range,
        default=range(2, 5),
        help="Range of number of rows to cluster (format: START:END)",
    )
    parser.add_argument(
        "--col-range",
        type=parse_range,
        default=range(2, 5),
        help="Range of number of columns to cluster (format: START:END)",
    )
    parser.add_argument(
        "--cluster-output-fn",
        type=str,
        default="",
        help="Path to cluster output file (relative to project root)",
    )
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    # Convert paths to absolute paths relative to project root
    data_fn = Path(args.data_fn).resolve()
    row_range = args.row_range
    col_range = args.col_range
    store_item_matrix_fn = Path(args.store_item_matrix_fn).resolve()
    cluster_output_fn = Path(args.cluster_output_fn).resolve()
    output_fn = Path(args.output_fn).resolve()

    log_dir = Path(args.log_dir).resolve()

    # Set up logging
    logger = setup_logging(log_dir, args.log_level)
    try:
        # Log configuration
        logger.info("Starting data clustering with configuration:")
        logger.info(f"  Data fn: {data_fn}")
        logger.info(f"  Row range: {row_range}")
        logger.info(f"  Col range: {col_range}")
        logger.info(f"  Cluster output fn: {cluster_output_fn}")
        logger.info(f"  Output fn: {output_fn}")

        # Load and preprocess data
        df = load_data(data_fn)
        cluster_data(
            df,
            store_item_matrix_fn=store_item_matrix_fn,
            cluster_output_fn=cluster_output_fn,
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
