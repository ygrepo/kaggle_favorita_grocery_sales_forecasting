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
import pandas as pd

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
            logger.info("Completed successfully")
            return
        if action == "both":
            logger.info("Merging store and item cluster medians")
            # Load the data
            df = load_raw_data(data_fn)
            df = df[["date", "store", "item", "growth_rate"]]
            logger.info(f"Initial dataframe shape: {df.shape}")

            # Need to get store and item cluster assignments first
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            model_dict = torch.load(
                model_fn, map_location=device, weights_only=False
            )
            assignment_df = model_dict["assignments"].copy()
            del model_dict
            gc.collect()

            # ---------------------------------------------------------
            # 1) ITEM ASSIGNMENTS (multi-membership via merge)
            # ---------------------------------------------------------
            logger.info("Merging item assignments to add item_cluster_id")
            item_assignments = (
                assignment_df.query("factor_name == 'SKU'")
                .rename(columns={"item_name": "item"})
                .assign(item=lambda x: x["item"].astype(int))
            )
            logger.info(f"Item assignments shape: {item_assignments.shape}")

            df = df.merge(
                item_assignments[["item", "cluster_id"]].rename(
                    columns={"cluster_id": "item_cluster_id"}
                ),
                on="item",
                how="left",
            )
            logger.info(
                f"After merging item assignments, df shape: {df.shape}"
            )
            del item_assignments
            gc.collect()

            # ---------------------------------------------------------
            # 2) ITEM MEDIANS via MultiIndex lookup (no big merge)
            # ---------------------------------------------------------
            logger.info(f"Loading item medians: {args.item_input_fn}")
            item_input_fn = Path(args.item_input_fn).resolve()
            item_medians = read_csv_or_parquet(item_input_fn)
            logger.info(f"Item medians shape: {item_medians.shape}")

            # Build MultiIndex -> item_median Series
            logger.info("Building MultiIndex for item medians")
            item_key_index = pd.MultiIndex.from_frame(
                item_medians[["date", "item_cluster_id"]]
            )
            item_median_series = pd.Series(
                item_medians["item_median"].to_numpy(), index=item_key_index
            )
            # Optional: downcast
            if pd.api.types.is_float_dtype(item_median_series):
                item_median_series = item_median_series.astype("float32")

            del item_medians, item_key_index
            gc.collect()

            # Align to df rows
            logger.info("Assigning item_median via reindex")
            df_item_index = pd.MultiIndex.from_arrays(
                [df["date"].values, df["item_cluster_id"].values]
            )
            df["item_median"] = item_median_series.reindex(
                df_item_index
            ).to_numpy()

            del item_median_series, df_item_index
            gc.collect()

            # ---------------------------------------------------------
            # 3) STORE ASSIGNMENTS (multi-membership via merge)
            # ---------------------------------------------------------
            logger.info("Merging store assignments to add store_cluster_id")
            store_assignments = (
                assignment_df.query("factor_name == 'Store'")
                .rename(columns={"item_name": "store"})
                .assign(store=lambda x: x["store"].astype(int))
            )
            logger.info(f"Store assignments shape: {store_assignments.shape}")

            df = df.merge(
                store_assignments[["store", "cluster_id"]].rename(
                    columns={"cluster_id": "store_cluster_id"}
                ),
                on="store",
                how="left",
            )
            logger.info(
                f"After merging store assignments, df shape: {df.shape}"
            )
            del store_assignments, assignment_df
            gc.collect()

            # ---------------------------------------------------------
            # 4) STORE MEDIANS via MultiIndex lookup (no big merge)
            # ---------------------------------------------------------
            logger.info(f"Loading store medians: {args.store_input_fn}")
            store_input_fn = Path(args.store_input_fn).resolve()
            store_medians = read_csv_or_parquet(store_input_fn)
            logger.info(f"Store medians shape: {store_medians.shape}")

            logger.info("Building MultiIndex for store medians")
            store_key_index = pd.MultiIndex.from_frame(
                store_medians[["date", "store_cluster_id"]]
            )
            store_median_series = pd.Series(
                store_medians["store_median"].to_numpy(), index=store_key_index
            )
            if pd.api.types.is_float_dtype(store_median_series):
                store_median_series = store_median_series.astype("float32")

            del store_medians, store_key_index
            gc.collect()

            logger.info("Assigning store_median via reindex")
            df_store_index = pd.MultiIndex.from_arrays(
                [df["date"].values, df["store_cluster_id"].values]
            )
            df["store_median"] = store_median_series.reindex(
                df_store_index
            ).to_numpy()

            del store_median_series, df_store_index
            gc.collect()

            logger.info(f"Final dataframe shape: {df.shape}")
            logger.info(f"Final columns: {df.columns.tolist()}")
            save_csv_or_parquet(df, output_fn)
            logger.info("Completed successfully")
            return

        if action == "item":
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
            medians = medians[["date", "item_cluster_id", "item_median"]]
            save_csv_or_parquet(medians, output_fn)
            logger.info("Completed successfully")
            return

        raise ValueError(f"Unknown action: {action}")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
