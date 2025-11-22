#!/usr/bin/env python3
"""
Create cluster medians script for the Growth Rate Forecasting model.

This script handles the complete create cluster medians pipeline including:
- Data loading and preprocessing
- Logging
"""

import sys
import argparse
from pathlib import Path
import torch
import gc

# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import (
    setup_logging,
    get_logger,
    save_csv_or_parquet,
    read_csv_or_parquet,
)
from src.data_utils import load_raw_data

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create cluster medians for Growth Rate Forecasting model"
    )
    parser.add_argument(
        "--model_fn",
        type=Path,
        default="",
        help="Path to trained model file (relative to project root)",
    )
    parser.add_argument(
        "--data_fn",
        type=Path,
        default="",
        help="Path to training data file (relative to project root)",
    )
    parser.add_argument(
        "--output_fn",
        type=Path,
        default="",
        help="Path to output file (relative to project root)",
    )
    parser.add_argument(
        "--store_input_fn",
        type=Path,
        default="",
        help="Path to store input file (relative to project root)",
    )
    parser.add_argument(
        "--item_input_fn",
        type=Path,
        default="",
        help="Path to item input file (relative to project root)",
    )
    parser.add_argument(
        "--action",
        type=str,
        default="store",
        choices=["store", "item", "both"],
        help="Whether to compute store, item, or both cluster medians",
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
        logger.info(f"  Action: {args.action}")
        logger.info(f"  Log fn: {log_fn}")

        action = args.action

        if action in ["store", "item"]:
            # Load model for store/item actions
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            model_dict = torch.load(
                model_fn, map_location=device, weights_only=False
            )
            assignment_df = model_dict["assignments"].copy()
            del model_dict
            gc.collect()

            df = load_raw_data(data_fn)
            logger.info(f"Dataframe shape: {df.shape}")

        if action == "store":
            logger.info("Computing store cluster medians")
            # Get store assignments and rename columns
            assignments = (
                assignment_df.query("factor_name == 'Store'")
                .rename(columns={"item_name": "store"})
                .assign(store=lambda x: x["store"].astype(int))
            )

            # Add cluster IDs to main dataframe
            df = df[["date", "store", "growth_rate"]].merge(
                assignments[["store", "cluster_id"]].rename(
                    columns={"cluster_id": "store_cluster_id"}
                ),
                on="store",
                how="left",
            )
            del assignments
            gc.collect()
            logger.info("Merged store assignments")

            # Compute store cluster medians by date
            medians = (
                df.groupby(["date", "store_cluster_id"])["growth_rate"]
                .median()
                .reset_index()
                .rename(columns={"growth_rate": "store_median"})
            )
            # Don't include 'store' in the output - it's not in the groupby
            medians = medians[["date", "store_cluster_id", "store_median"]]
            save_csv_or_parquet(medians, output_fn)

        elif action == "item":
            logger.info("Computing item cluster medians")
            # Get item assignments and rename columns
            assignments = (
                assignment_df.query("factor_name == 'SKU'")
                .rename(columns={"item_name": "item"})
                .assign(item=lambda x: x["item"].astype(int))
            )

            df = df[["date", "item", "growth_rate"]].merge(
                assignments[["item", "cluster_id"]].rename(
                    columns={"cluster_id": "item_cluster_id"}
                ),
                on="item",
                how="left",
            )
            del assignments
            gc.collect()
            logger.info("Merged item assignments")

            # Compute item cluster medians by date
            medians = (
                df.groupby(["date", "item_cluster_id"])["growth_rate"]
                .median()
                .reset_index()
                .rename(columns={"growth_rate": "item_median"})
            )
            # FIX: Don't include 'item' in the output - it's not in the groupby
            medians = medians[["date", "item_cluster_id", "item_median"]]
            save_csv_or_parquet(medians, output_fn)

        elif action == "both":
            logger.info("Merging store and item cluster medians")

            # Convert Path objects to resolved paths
            store_input_fn = Path(args.store_input_fn).resolve()
            item_input_fn = Path(args.item_input_fn).resolve()

            # Load the main data
            df = load_raw_data(data_fn)
            logger.info(f"Initial dataframe shape: {df.shape}")

            # Load and merge store medians
            medians = read_csv_or_parquet(store_input_fn)
            logger.info(f"Store medians shape: {medians.shape}")

            # Need to get store cluster assignments first
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            model_dict = torch.load(
                model_fn, map_location=device, weights_only=False
            )
            assignment_df = model_dict["assignments"].copy()
            del model_dict
            gc.collect()

            # Get store assignments to add store_cluster_id to df
            assignments = (
                assignment_df.query("factor_name == 'Store'")
                .rename(columns={"item_name": "store"})
                .assign(store=lambda x: x["store"].astype(int))
            )

            df = df.merge(
                assignments[["store", "cluster_id"]].rename(
                    columns={"cluster_id": "store_cluster_id"}
                ),
                on="store",
                how="left",
            )

            # Now merge store medians
            df = df.merge(medians, on=["date", "store_cluster_id"], how="left")
            del medians, assignments
            gc.collect()
            logger.info("Merged store medians")

            # Load and merge item medians
            medians = read_csv_or_parquet(item_input_fn)
            logger.info(f"Item medians shape: {medians.shape}")

            # Get item assignments to add item_cluster_id to df
            assignments = (
                assignment_df.query("factor_name == 'SKU'")
                .rename(columns={"item_name": "item"})
                .assign(item=lambda x: x["item"].astype(int))
            )

            df = df.merge(
                assignments[["item", "cluster_id"]].rename(
                    columns={"cluster_id": "item_cluster_id"}
                ),
                on="item",
                how="left",
            )

            # Now merge item medians
            df = df.merge(medians, on=["date", "item_cluster_id"], how="left")
            del medians, assignments, assignment_df
            gc.collect()
            logger.info("Merged item medians")

            save_csv_or_parquet(df, output_fn)

        logger.info(f"Final dataframe shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info("Completed successfully")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
