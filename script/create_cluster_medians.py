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
        logger.info(f"Final dataframe shape: {df.shape}")

        # Get store assignments and rename columns
        assignments = model_dict["assignments"].query("factor_name == 'Store'")
        assignments.rename(columns={"item_name": "store"}, inplace=True)
        assignments["store"] = assignments["store"].astype(int)
        # Add cluster IDs to main dataframe
        df = df.merge(
            assignments[["store", "cluster_id"]].rename(
                columns={"cluster_id": "store_cluster_id"}
            ),
            on="store",
            how="left",
        )
        # Compute store cluster medians by date
        medians = (
            df.groupby(["date", "store_cluster_id"])["growth_rate"]
            .median()
            .reset_index()
            .rename(columns={"growth_rate": "store_median"})
        )
        df = df.merge(medians, on=["date", "store_cluster_id"], how="left")

        # Get item assignments and rename columns
        assignments = model_dict["assignments"].query("factor_name == 'SKU'")
        assignments.rename(columns={"item_name": "item"}, inplace=True)
        assignments["item"] = assignments["item"].astype(int)

        df = df.merge(
            assignments[["item", "cluster_id"]].rename(
                columns={"cluster_id": "item_cluster_id"}
            ),
            on="item",
            how="left",
        )

        # Compute item cluster medians by date
        medians = (
            df.groupby(["date", "item_cluster_id"])["growth_rate"]
            .median()
            .reset_index()
            .rename(columns={"growth_rate": "item_median"})
        )
        df = df.merge(medians, on=["date", "item_cluster_id"], how="left")

        # Optional: drop intermediate cluster columns
        # df = df.drop(columns=["store_cluster_id", "item_cluster_id"])

        save_csv_or_parquet(df, output_fn)

        logger.info(f"Final dataframe shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")

        logger.info("Completed successfully")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
