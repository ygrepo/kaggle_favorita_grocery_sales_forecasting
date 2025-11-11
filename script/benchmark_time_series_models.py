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
    parser.add_argument(
        "--target_col",
        type=str,
        default="unit_sales",
        help="Name of the target column to forecast",
    )
    parser.add_argument(
        "--time_col",
        type=str,
        default="date",
        help="Name of the time column",
    )
    parser.add_argument(
        "--group_cols",
        type=str,
        nargs="*",
        default=["store_nbr", "item_nbr"],
        help="Columns to group by for individual time series",
    )
    return parser.parse_args()


def prepare_time_series_data(df, time_col, target_col, group_cols=None):
    """
    Prepare time series data from the loaded DataFrame.

    Args:
        df: Input DataFrame
        time_col: Name of the time column
        target_col: Name of the target column
        group_cols: List of columns to group by (if None, aggregate all data)

    Returns:
        Prepared DataFrame ready for TimeSeries conversion
    """
    logger.info(f"Preparing time series data...")
    logger.info(f"Original data shape: {df.shape}")
    logger.info(f"Available columns: {df.columns.tolist()}")

    # Check if required columns exist
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in data")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    # Convert time column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    # Sort by time
    df = df.sort_values(time_col)

    if group_cols:
        # Check if group columns exist
        missing_cols = [col for col in group_cols if col not in df.columns]
        if missing_cols:
            logger.warning(
                f"Group columns {missing_cols} not found. Available columns: {df.columns.tolist()}"
            )
            # Use available group columns only
            group_cols = [col for col in group_cols if col in df.columns]
            if not group_cols:
                logger.info(
                    "No valid group columns found. Aggregating all data."
                )
                group_cols = None

    if group_cols:
        logger.info(f"Grouping by: {group_cols}")
        # For now, let's take the first group to demonstrate
        # In a real scenario, you might want to process multiple groups
        first_group = df.groupby(group_cols).first().index[0]
        logger.info(f"Using first group: {first_group}")

        # Filter data for the first group
        mask = True
        for i, col in enumerate(group_cols):
            if isinstance(first_group, tuple):
                mask = mask & (df[col] == first_group[i])
            else:
                mask = mask & (df[col] == first_group)

        df_filtered = df[mask].copy()
    else:
        logger.info("Aggregating all data by time")
        # Aggregate all data by time
        df_filtered = df.groupby(time_col)[target_col].sum().reset_index()

    # Create the final DataFrame for TimeSeries
    ts_df = df_filtered[[time_col, target_col]].copy()
    ts_df = ts_df.set_index(time_col)

    # Remove any duplicate timestamps by taking the mean
    ts_df = ts_df.groupby(ts_df.index).mean()

    # Fill missing values if any
    ts_df = ts_df.fillna(method="ffill").fillna(method="bfill")

    logger.info(f"Prepared time series shape: {ts_df.shape}")
    logger.info(f"Date range: {ts_df.index.min()} to {ts_df.index.max()}")
    logger.info(f"Target column stats:\n{ts_df[target_col].describe()}")

    return ts_df


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
    train: TimeSeries,
    val: TimeSeries,
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate a model with error handling."""
    try:
        logger.info(f"Training {modelType} model...")
        model.fit(train)

        logger.info(f"Generating forecast with {modelType}...")
        forecast = model.predict(len(val))

        metrics = calculate_metrics(val, forecast)

        new_row = pd.DataFrame(
            [
                {
                    "Model": modelType,
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
        logger.info(f"{modelType} completed successfully")

    except Exception as e:
        logger.error(f"Error with {modelType}: {e}")
        # Add a row with NaN values to indicate failure
        new_row = pd.DataFrame(
            [
                {
                    "Model": modelType,
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
        logger.info(f"  Target column: {args.target_col}")
        logger.info(f"  Time column: {args.time_col}")
        logger.info(f"  Group columns: {args.group_cols}")

        # Load raw data
        logger.info("Loading raw data...")
        df = load_raw_data(data_fn)

        # Prepare time series data
        ts_df = prepare_time_series_data(
            df, args.time_col, args.target_col, args.group_cols
        )

        # Convert to Darts TimeSeries
        logger.info("Converting to Darts TimeSeries...")
        ts = TimeSeries.from_dataframe(ts_df)

        # Split the data
        logger.info(f"Splitting data at {args.split_point}")
        train_ts, val_ts = ts.split_before(args.split_point)

        logger.info(f"Train series length: {len(train_ts)}")
        logger.info(f"Validation series length: {len(val_ts)}")

        # Initialize metrics dataframe
        logger.info("Running models...")
        metrics_df = pd.DataFrame(
            columns=[
                "Model",
                "RMSE",
                "RMSSE",
                "MAE",
                "MASE",
                "MAPE",
                "OPE",
            ]
        )

        # Run models
        metrics_df = eval_model(
            "ExponentialSmoothing",
            ExponentialSmoothing(),
            train_ts,
            val_ts,
            metrics_df,
        )
        metrics_df = eval_model(
            "AutoARIMA", AutoARIMA(), train_ts, val_ts, metrics_df
        )
        metrics_df = eval_model("Theta", Theta(), train_ts, val_ts, metrics_df)

        # Save results
        logger.info(f"Saving results to {output_metrics_fn}")
        save_csv_or_parquet(metrics_df, output_metrics_fn)

        # Log results
        logger.info("Benchmarking Results:")
        logger.info(f"\n{metrics_df.to_string(index=False)}")

        logger.info("Benchmarking completed successfully!")

    except Exception as e:
        logger.error(f"Error in benchmarking: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
