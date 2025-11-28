#!/usr/bin/env python3
"""
Training script for the Favorita Grocery Sales Forecasting model.

This script handles the complete training pipeline including:
- Data loading and preprocessing
- Logging
- Classical local models (ExponentialSmoothing, AutoARIMA, Theta, KalmanForecaster)
"""

import sys
import argparse
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import (
    setup_logging,
    get_logger,
    save_csv_or_parquet,
)
from src.data_utils import load_raw_data
from src.time_series_utils import (
    ModelType,
    parse_models_arg,
    create_model,
    prepare_store_item_series,
    eval_model_with_covariates,
    get_train_val_data_with_covariates,
    FUTURE_COV_COLS,
    PAST_COV_COLS,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Classical models for Favorita Grocery Sales Forecasting"
    )
    parser.add_argument(
        "--data_fn",
        type=Path,
        default="",
        help="Path to data file (relative to project root)",
    )
    parser.add_argument(
        "--metrics_fn",
        type=Path,
        default="",
        help="Path to metrics output file (relative to project root)",
    )
    parser.add_argument(
        "--split_point",
        type=float,
        default=0.8,
        help="Proportion of data to use for training",
    )
    parser.add_argument(
        "--min_train_data_points",
        type=int,
        default=15,
        help="Minimum number of data points to train on",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="EXPONENTIAL_SMOOTHING,AUTO_ARIMA,THETA,KALMAN",
        help=(
            "Comma-separated list of models to train. "
            "Options: EXPONENTIAL_SMOOTHING,AUTO_ARIMA,THETA,KALMAN"
        ),
    )
    parser.add_argument(
        "--N",
        type=int,
        default=0,
        help="Limit to first N combinations",
    )
    parser.add_argument(
        "--xl_design",
        type=bool,
        default=False,
        help="Use the XL design for the models",
    )
    parser.add_argument(
        "--log_fn",
        type=Path,
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


# ---------------------------------------------------------------------
# Data prep + per-(store,item) logic
# ---------------------------------------------------------------------


def process_store_item_combination(
    df: pd.DataFrame,
    store: int,
    item: int,
    split_point: float,
    min_train_data_points: int,
    metrics_df: pd.DataFrame,
    model_types: List[ModelType],
    xl_design: bool,
) -> pd.DataFrame:
    """Process a single store-item combination."""

    logger.info(f"Processing store {store}, item {item}")

    # Prepare the time series data with all features
    ts_df = prepare_store_item_series(
        df,
        store,
        item,
        store_medians_fn=None,
        item_medians_fn=None,
        store_assign_fn=None,
        item_assign_fn=None,
    )
    if ts_df.empty:
        logger.warning(f"No data for store {store}, item {item}")
        return metrics_df

    # Use the covariate-aware data preparation
    data_dict = get_train_val_data_with_covariates(
        ts_df=ts_df,
        store=store,
        item=item,
        split_point=split_point,
        min_train_data_points=min_train_data_points,
    )

    if data_dict is None:
        logger.warning(
            f"store:{store},item:{item}. Skipping, no training data."
        )
        return metrics_df

    # Train all models using the covariate-aware evaluation
    for mtype in model_types:
        model = create_model(mtype, xl_design=xl_design)

        # Classical models don't support covariates, but eval_model_with_covariates
        # will handle this by only using the target series
        metrics_df = eval_model_with_covariates(
            mtype.value,
            model,
            store,
            item,
            data_dict,
            metrics_df,
        )

    return metrics_df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    """Main training function."""
    args = parse_args()
    data_fn = Path(args.data_fn).resolve()
    output_metrics_fn = Path(args.metrics_fn).resolve()
    log_fn = args.log_fn.resolve()

    # Set up logging
    logger = setup_logging(log_fn, args.log_level)
    logger.info(f"Log fn: {log_fn}")

    # Parse models list
    model_types = parse_models_arg(args.models)
    logger.info(f"Models to train: {[m.value for m in model_types]}")

    try:
        logger.info(
            "Starting time series model benchmarking with configuration:"
        )
        logger.info(f"  Data fn: {data_fn}")
        logger.info(f"  Metrics fn: {output_metrics_fn}")
        logger.info(f"  Log fn: {log_fn}")
        logger.info(f"  Split point: {args.split_point}")
        logger.info(f"  Min train data points: {args.min_train_data_points}")
        logger.info(f"  N: {args.N}")
        logger.info(f"  XL design: {args.xl_design}")

        # Load raw data
        logger.info("Loading raw data...")
        df = load_raw_data(data_fn)

        logger.info("Finding unique store-item combinations...")
        unique_combinations = df[["store", "item"]].drop_duplicates()
        if args.N > 0:
            logger.info(f"Limiting to first {args.N} combinations")
            unique_combinations = unique_combinations.head(args.N)

        logger.info(f"Found {len(unique_combinations)} unique combinations")

        available_future_covs = [
            col for col in FUTURE_COV_COLS if col in df.columns
        ]
        available_past_covs = [
            col for col in PAST_COV_COLS if col in df.columns
        ]

        logger.info(
            f"Available future covariate columns: {available_future_covs}"
        )
        logger.info(f"Available past covariate columns: {available_past_covs}")

        logger.info("Running models...")

        metrics_df = pd.DataFrame(
            columns=[
                "Model",
                "Store",
                "Item",
                "RMSSE",
                "MASE",
                "SMAPE",
                "MARRE",
                "RMSE",
                "MAE",
                "OPE",
            ]
        )

        for _, row in tqdm(
            unique_combinations.iterrows(), total=len(unique_combinations)
        ):
            store = row["store"]
            item = row["item"]

            metrics_df = process_store_item_combination(
                df,
                store,
                item,
                args.split_point,
                args.min_train_data_points,
                metrics_df,
                model_types,
                args.xl_design,
            )

        logger.info(f"Saving results to {output_metrics_fn}")
        save_csv_or_parquet(metrics_df, output_metrics_fn)
        logger.info("Benchmarking completed successfully!")

        # Log summary statistics
        if not metrics_df.empty:
            logger.info("Summary of results:")
            for model in metrics_df["Model"].unique():
                model_metrics = metrics_df[metrics_df["Model"] == model]
                successful_runs = model_metrics.dropna(subset=["SMAPE"]).shape[
                    0
                ]
                total_runs = len(model_metrics)
                logger.info(
                    f"  {model}: {successful_runs}/{total_runs} successful runs"
                )

                if successful_runs > 0:
                    clean_metrics = model_metrics.dropna(
                        subset=["SMAPE", "RMSSE"]
                    )
                    if len(clean_metrics) > 0:
                        smape_mean = clean_metrics["SMAPE"].mean()
                        rmsse_mean = clean_metrics["RMSSE"].mean()
                        logger.info(
                            f"    Mean SMAPE: {smape_mean:.4f}, Mean RMSSE: {rmsse_mean:.4f}"
                        )
        else:
            logger.warning("No successful model runs completed!")

    except Exception as e:
        logger.error(f"Error in benchmarking: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
