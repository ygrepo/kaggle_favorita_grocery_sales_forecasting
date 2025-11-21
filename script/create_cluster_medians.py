#!/usr/bin/env python3
"""
Create cluster medians script for the Favorita Grocery Sales Forecasting model.

This script handles the complete create cluster medians pipeline including:
- Data loading and preprocessing
- Logging
"""

import sys
import argparse
from pathlib import Path
import torch

# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import setup_logging, get_logger, save_csv_or_parquet
from src.data_utils import compute_cluster_medians, load_raw_data

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create cluster medians for Shop Sales Forecasting model"
    )
    parser.add_argument(
        "--model_fn",
        type=str,
        default="",
        help="Path to trained model file (relative to project root)",
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


def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    # Convert paths to absolute paths relative to project root
    model_fn = Path(args.model_fn).resolve()
    data_fn = Path(args.data_fn).resolve()
    output_fn = Path(args.output_fn).resolve()
    log_fn = Path(args.log_fn).resolve()
    # Set up logging
    setup_logging(log_fn, args.log_level)
    try:
        # Log configuration
        logger.info("Starting:")
        logger.info(f"  Model fn: {model_fn}")
        logger.info(f"  Data fn: {data_fn}")
        logger.info(f"  Output fn: {output_fn}")
        logger.info(f"  Log fn: {log_fn}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dict = torch.load(
            model_fn, map_location=device, weights_only=False
        )

        df = load_raw_data(data_fn)
        assignments = model_dict["assignments"].query("factor_name == 'Store'")
        assignments.rename(columns={"item_name": "store"}, inplace=True)
        assignments["store"] = assignments["store"].astype(int)
        logger.info(f"Unique stores: {df['store'].nunique()}")
        df = df.merge(assignments, on=["cluster_id", "date"], how="left")
        logger.info(f"Unique stores: {df['store'].nunique()}")
        df = compute_cluster_medians(
            df,
            date_col="date",
            cluster_col="cluster_id",
            value_col="growth_rate",
        )

        assignments = model_dict["assignments"].query("factor_name == 'SKU'")
        assignments.rename(columns={"item_name": "sku"}, inplace=True)
        assignments["sku"] = assignments["sku"].astype(int)
        df = df.merge(assignments, on=["cluster_id", "date"], how="left")
        logger.info(f"Unique stores: {df['store'].nunique()}")
        df = compute_cluster_medians(
            df,
            date_col="date",
            cluster_col="cluster_id",
            value_col="growth_rate",
        )
        logger.info(f"Unique stores: {df['store'].nunique()}")
        logger.info(f"Unique skus: {df['sku'].nunique()}")
        save_csv_or_parquet(df, output_fn)
        logger.info("Completed successfully")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
