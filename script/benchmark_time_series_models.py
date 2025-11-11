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


def calculate_metrics(
    val: TimeSeries,
    forecast: TimeSeries,
):
    return {
        "rmse": rmse(val, forecast),
        "rmsse": rmsse(val, forecast),
        "mae": mae(val, forecast),
        "mase": mase(val, forecast),
        "mape": mape(val, forecast),
        "ope": ope(val, forecast),
    }


def eval_model(
    modelType: str,
    model: LocalForecastingModel,
    train: TimeSeries,
    val: TimeSeries,
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    model.fit(train)
    forecast = model.predict(len(val))
    metrics = calculate_metrics(val, forecast)
    metrics_df = pd.concat(
        [
            metrics_df,
            pd.DataFrame(
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
            ),
        ],
        ignore_index=True,
    )
    return metrics_df


def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    data_fn = Path(args.data_fn).resolve()
    ouput_metrics_fn = Path(args.metrics_fn).resolve()
    log_fn = Path(args.log_fn).resolve()

    # Set up logging
    logger = setup_logging(log_fn, args.log_level)
    logger.info(f"Log fn: {log_fn}")

    try:
        # Log configuration
        logger.info("Starting creating data loaders with configuration:")
        logger.info(f"  Data fn: {data_fn}")
        logger.info(f"  Metrics fn: {ouput_metrics_fn}")
        logger.info(f"  Log fn: {log_fn}")
        logger.info(f"  Split point: {args.split_point}")
        df = load_raw_data(data_fn)
        df = TimeSeries.from_dataframe(df)
        train_df, val_df = df.split_before(args.split_point)

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

        metrics_df = eval_model(
            "ExponentialSmoothing",
            ExponentialSmoothing(),
            train_df,
            val_df,
            metrics_df,
        )
        metrics_df = eval_model(
            "AutoARIMA", AutoARIMA(), train_df, val_df, metrics_df
        )
        metrics_df = eval_model("Theta", Theta(), train_df, val_df, metrics_df)
        save_csv_or_parquet(metrics_df, ouput_metrics_fn)

    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        raise


if __name__ == "__main__":
    main()
