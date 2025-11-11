#!/usr/bin/env python3
"""
Training script for the Favorita Grocery Sales Forecasting model.

This script handles the complete training pipeline including:
- Data loading and preprocessing
- Logging
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import AutoARIMA, ExponentialSmoothing, Theta
from darts.metrics import rmse, rmsse, mae, mase, mape, ope
from darts.models.forecasting.forecasting_model import LocalForecastingModel
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

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create features for Favorita Grocery Sales Forecasting model"
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
        help="Path to data file (relative to project root)",
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


def prepare_store_item_series(
    df: pd.DataFrame,
    store: int,
    item: int,
) -> pd.DataFrame:
    """Prepare time series for a specific store-item combination."""

    # Filter data for specific store-item combination
    mask = (df["store"] == store) & (df["item"] == item)
    series_df = df[mask].copy()

    if len(series_df) == 0:
        logger.warning(f"No data for store {store}, item {item}")
        return pd.DataFrame()

    # Sort by date and prepare time series
    series_df = series_df.sort_values("date")

    # Create time series DataFrame
    ts_df = series_df[["date", "growth_rate"]].copy()
    ts_df = ts_df.set_index("date")

    return ts_df


def process_store_item_combination(
    df: pd.DataFrame,
    store: int,
    item: int,
    split_point: float,
    min_train_data_points: int,
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Process a single store-item combination."""

    # Prepare time series
    ts_df = prepare_store_item_series(df, store, item)

    try:
        # Convert to Darts TimeSeries
        ts = TimeSeries.from_dataframe(
            ts_df, fill_missing_dates=True, freq="D"
        )

        # Split data
        train_ts, val_ts = ts.split_before(split_point)

        if len(val_ts) == 0:
            logger.warning(
                f"No validation data for store {store}, item {item}"
            )
            return metrics_df
        if len(train_ts) < min_train_data_points:
            logger.warning(
                f"Training series too short ({len(train_ts)} < {min_train_data_points}) for store {store}, item {item}. Skipping."
            )
            return metrics_df

        # Test models
        models = [
            ("ExponentialSmoothing", ExponentialSmoothing()),
            ("AutoARIMA", AutoARIMA()),
            ("Theta", Theta()),
        ]

        for model_name, model in models:
            metrics_df = eval_model(
                model_name,
                model,
                store,
                item,
                train_ts,
                val_ts,
                metrics_df,
            )

        return metrics_df

    except Exception as e:
        logger.warning(f"Failed to process store {store}, item {item}: {e}")
        return metrics_df


def calculate_metrics(
    val: TimeSeries,
    forecast: TimeSeries,
):
    """Calculate metrics with error handling."""
    try:
        return {
            "rmse": rmse(val, forecast),
            "rmsse": rmsse(val, forecast),
            "mae": mae(val, forecast),
            "mase": mase(val, forecast),
            "mape": mape(val, forecast),
            "ope": ope(val, forecast),
        }
    except Exception as e:
        logger.warning(f"Error calculating some metrics: {e}")
        # Return basic metrics only
        return {
            "rmse": rmse(val, forecast),
            "mae": mae(val, forecast),
            "rmsse": np.nan,
            "mase": np.nan,
            "mape": np.nan,
            "ope": np.nan,
        }


def eval_model(
    modelType: str,
    model: LocalForecastingModel,
    store: int,
    item: int,
    train: TimeSeries,
    val: TimeSeries,
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate a model with error handling."""
    try:
        logger.info(
            f"Training {modelType} model, store {store}, item {item}..."
        )
        model.fit(train)

        logger.info(
            f"Generating forecast with {modelType}, store {store}, item {item}..."
        )
        forecast = model.predict(len(val))

        metrics = calculate_metrics(val, forecast)

        new_row = pd.DataFrame(
            [
                {
                    "Model": modelType,
                    "Store": store,
                    "Item": item,
                    "RMSE": metrics["rmse"],
                    "RMSSE": metrics["rmsse"],
                    "MAE": metrics["mae"],
                    "MASE": metrics["mase"],
                    "MAPE": metrics["mape"],
                    "OPE": metrics["ope"],
                }
            ]
        )

        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
        logger.info(
            f"{modelType} completed successfully, store {store}, item {item}"
        )

    except Exception as e:
        logger.error(f"Error with {modelType}: {e}")
        # Add a row with NaN values to indicate failure
        new_row = pd.DataFrame(
            [
                {
                    "Model": modelType,
                    "Store": store,
                    "Item": item,
                    "RMSE": np.nan,
                    "RMSSE": np.nan,
                    "MAE": np.nan,
                    "MASE": np.nan,
                    "MAPE": np.nan,
                    "OPE": np.nan,
                }
            ]
        )
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    return metrics_df


def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    data_fn = Path(args.data_fn).resolve()
    output_metrics_fn = Path(args.metrics_fn).resolve()
    log_fn = args.log_fn.resolve()

    # Set up logging
    logger = setup_logging(log_fn, args.log_level)
    logger.info(f"Log fn: {log_fn}")

    try:
        # Log configuration
        logger.info(
            "Starting time series model benchmarking with configuration:"
        )
        logger.info(f"  Data fn: {data_fn}")
        logger.info(f"  Metrics fn: {output_metrics_fn}")
        logger.info(f"  Log fn: {log_fn}")
        logger.info(f"  Split point: {args.split_point}")

        # Load raw data
        logger.info("Loading raw data...")
        df = load_raw_data(data_fn)
        # Get unique store-item pairs
        logger.info("Finding unique store-item combinations...")
        unique_combinations = df[["store", "item"]].drop_duplicates()

        # Initialize metrics dataframe
        logger.info("Running models...")
        metrics_df = pd.DataFrame(
            columns=[
                "Model",
                "Store",
                "Item",
                "RMSE",
                "RMSSE",
                "MAE",
                "MASE",
                "MAPE",
                "OPE",
            ]
        )

        for idx, row in tqdm(
            unique_combinations.iterrows(), total=len(unique_combinations)
        ):
            store = row["store"]
            item = row["item"]

            metrics_df = process_store_item_combination(
                df, store, item, args.split_point, metrics_df
            )

        logger.info(f"Saving results to {output_metrics_fn}")
        save_csv_or_parquet(metrics_df, output_metrics_fn)
        logger.info("Benchmarking completed successfully!")

    except Exception as e:
        logger.error(f"Error in benchmarking: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
