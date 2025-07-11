#!/usr/bin/env python3
"""
Training script for the Favorita Grocery Sales Forecasting model.

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

from src.utils import prepare_training_data_from_raw_df, build_feature_and_label_cols


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
            "store_nbr": np.uint8,
            "item_nbr": np.uint32,
            "unit_sales": np.float32,
        }
        df = pd.read_csv(data_fn, dtype=dtype_dict, low_memory=False)
        df.rename(columns={"store_nbr": "store", "item_nbr": "item"}, inplace=True)
        df["store_item"] = df["store"].astype(str) + "_" + df["item"].astype(str)
        cols = ["date", "store_item", "store", "item"] + [
            c for c in df.columns if c not in ("date", "store_item", "store", "item")
        ]
        df = df[cols]
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        logger.info(f"Loaded data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def load_cluster(cluster_fn: Path) -> pd.DataFrame:
    """Load and preprocess cluster data.

    Args:
        cluster_fn: Path to the cluster data file

    Returns:
        Preprocessed DataFrame
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading cluster from {cluster_fn}")

    try:
        df = pd.read_csv(cluster_fn)
        logger.info(f"Loaded cluster with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading cluster: {e}")
        raise


def create_features(
    df: pd.DataFrame, log_level: str, window_size: int, sales_fn: Path, cyc_fn: Path
):
    """Create features for training the model."""
    logger = logging.getLogger(__name__)
    logger.info("Starting creating features")
    final_df = prepare_training_data_from_raw_df(
        df,
        window_size=window_size,
        calendar_aligned=True,
        sales_fn=sales_fn,
        cyc_fn=cyc_fn,
        log_level=log_level,
    )
    return final_df


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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Expected a boolean value (true/false)")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create features for Favorita Grocery Sales Forecasting model"
    )
    parser.add_argument(
        "--data-fn",
        type=str,
        default="",
        help="Path to training data file (relative to project root)",
    )
    parser.add_argument(
        "--sales-fn",
        type=str,
        default="",
        help="Path to sales file (relative to project root)",
    )
    parser.add_argument(
        "--cyc-fn",
        type=str,
        default="",
        help="Path to cyc file (relative to project root)",
    )
    parser.add_argument(
        "--output-fn",
        type=str,
        default="",
        help="Path to output file (relative to project root)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=16,
        help="Size of the lookback window",
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
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    # Convert paths to absolute paths relative to project root
    data_fn = Path(args.data_fn).resolve()
    sales_fn = Path(args.sales_fn).resolve()
    cyc_fn = Path(args.cyc_fn).resolve()
    output_fn = Path(args.output_fn).resolve()
    log_dir = Path(args.log_dir).resolve()
    window_size = args.window_size

    # Set up logging
    print(f"Log dir: {log_dir}")
    logger = setup_logging(log_dir, args.log_level)

    try:
        # Log configuration
        logger.info("Starting creating training features with configuration:")
        logger.info(f"  Data fn: {data_fn}")
        logger.info(f"  Sales fn: {sales_fn}")
        logger.info(f"  Cyc fn: {cyc_fn}")
        logger.info(f"  Output fn: {output_fn}")
        logger.info(f"  Log dir: {log_dir}")
        logger.info(f"  Window size: {args.window_size}")

        # Load and preprocess data
        df = load_data(data_fn)
        # store_item = "44_1503844"
        # logger.info(f"Selected store_item: {store_item}")
        # df = df[df["store_item"] == store_item]

        final_df = create_features(
            df,
            window_size=window_size,
            sales_fn=sales_fn,
            cyc_fn=cyc_fn,
            log_level=args.log_level,
        )

        (
            meta_cols,
            _,
            _,
            x_feature_cols,
            label_cols,
            _,
            _,
        ) = build_feature_and_label_cols(window_size=window_size)
        # Save final_df to csv
        logger.info(f"Saving final_df to {output_fn}")
        final_df[meta_cols + x_feature_cols + label_cols].to_csv(output_fn)

    except Exception as e:
        logger.error(f"Error creating training features: {e}")
        raise


if __name__ == "__main__":
    main()
