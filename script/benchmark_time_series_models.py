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
from enum import Enum
from typing import List

import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler

from darts.models import (
    AutoARIMA,
    ExponentialSmoothing,
    Theta,
    KalmanForecaster,
)
from darts.models.forecasting.forecasting_model import LocalForecastingModel
from darts.utils.utils import SeasonalityMode

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
    prepare_store_item_series,
    eval_model,
    get_train_val_data,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Enum + Factory
# ---------------------------------------------------------------------


class ModelType(str, Enum):
    EXPONENTIAL_SMOOTHING = "EXPONENTIAL_SMOOTHING"
    AUTO_ARIMA = "AUTO_ARIMA"
    THETA = "THETA"
    KALMAN = "KALMAN"


def parse_models_arg(models_string: str) -> List[ModelType]:
    """
    Convert --models "EXPONENTIAL_SMOOTHING,AUTO_ARIMA" into
    [ModelType.EXPONENTIAL_SMOOTHING, ModelType.AUTO_ARIMA].
    """
    names = [m.strip().upper() for m in models_string.split(",") if m.strip()]
    try:
        return [ModelType(name) for name in names]
    except ValueError as e:
        raise ValueError(
            f"Invalid --models argument: {models_string}. "
            f"Valid options: {[m.value for m in ModelType]}"
        ) from e


def create_local_model(model_type: ModelType) -> LocalForecastingModel:
    """
    Factory to create a classical Darts local forecasting model.
    """
    if model_type == ModelType.EXPONENTIAL_SMOOTHING:
        return ExponentialSmoothing()

    if model_type == ModelType.AUTO_ARIMA:
        return AutoARIMA()

    if model_type == ModelType.THETA:
        return Theta(season_mode=SeasonalityMode.ADDITIVE)

    if model_type == ModelType.KALMAN:
        # simple univariate Kalman; can tune dim_x later
        return KalmanForecaster(dim_x=1, random_state=42)

    raise ValueError(f"Unsupported model type: {model_type}")


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
) -> pd.DataFrame:
    """Process a single store-item combination."""

    logger.info(f"Processing store {store}, item {item}")

    ts_df = prepare_store_item_series(df, store, item)
    if ts_df.empty:
        return metrics_df

    ts, train_ts, val_ts = get_train_val_data(
        df=df,
        store=store,
        item=item,
        split_point=split_point,
        min_train_data_points=min_train_data_points,
    )

    # try:
    #     # basic training-data checks (non-missing, variance)
    #     train_data_for_std = ts_df.iloc[: int(len(ts_df) * split_point)]
    #     non_missing_count = train_data_for_std["growth_rate"].count()
    #     logger.info(
    #         f"training: Non-missing count for store {store}, item {item}: {non_missing_count}"
    #     )
    #     if non_missing_count < min_train_data_points:
    #         logger.warning(
    #             f"Training series has insufficient data "
    #             f"({non_missing_count} non-NaN < {min_train_data_points}) "
    #             f"for store {store}, item {item}. Skipping."
    #         )
    #         return metrics_df

    #     train_std = train_data_for_std["growth_rate"].std()
    #     if train_std == 0 or np.isnan(train_std):
    #         logger.warning(
    #             f"Training series has zero variance for store {store}, item {item}. Skipping."
    #         )
    #         return metrics_df

    #     # Build TimeSeries
    #     ts = TimeSeries.from_dataframe(
    #         ts_df, fill_missing_dates=True, freq="D"
    #     )
    #     train_ts, val_ts = ts.split_before(split_point)

    #     if len(val_ts) == 0:
    #         logger.warning(
    #             f"No validation data for store {store}, item {item}"
    #         )
    #         return metrics_df

    #     # Stronger checks on filled train
    #     train_series_pd = train_ts.to_series()
    #     non_missing_count = train_series_pd.count()
    #     logger.info(
    #         f"training: Non-missing count for store {store}, item {item}: {non_missing_count}"
    #     )
    #     if non_missing_count < min_train_data_points:
    #         logger.warning(
    #             f"Training series has insufficient data "
    #             f"({non_missing_count} non-NaN < {min_train_data_points}) "
    #             f"for store {store}, item {item}. Skipping."
    #         )
    #         return metrics_df

    #     train_std = train_series_pd.std()
    #     if train_std == 0 or np.isnan(train_std):
    #         logger.warning(
    #             f"Training series has zero variance for store {store}, item {item}. Skipping."
    #         )
    #         return metrics_df

    #     non_missing_count_val = val_ts.to_series().count()
    #     logger.info(
    #         f"Validation: Non-missing count for store {store}, item {item}: {non_missing_count_val}"
    #     )

    #     # re-create ts with fillna_value=0 if desired for some models
    #     ts_filled = TimeSeries.from_dataframe(
    #         ts_df, fill_missing_dates=True, freq="D", fillna_value=0
    #     )
    #     train_ts, val_ts = ts_filled.split_before(split_point)

    # evaluate each requested model
    for mtype in model_types:
        model = create_local_model(mtype)
        metrics_df = eval_model(
            mtype,
            model,
            store,
            item,
            train_ts,
            val_ts,
            metrics_df,
        )

    return metrics_df

    # except Exception as e:
    #     logger.warning(f"Failed to process store {store}, item {item}: {e}")
    #     return me


# # ---------------------------------------------------------------------
# # Metrics
# # ---------------------------------------------------------------------


# def calculate_rmsse(
#     train_vals: np.ndarray,
#     val_vals: np.ndarray,
#     fcst_vals: np.ndarray,
#     epsilon: float = np.finfo(float).eps,
# ) -> float:
#     """Calculates RMSSE manually using numpy."""
#     rmse_forecast = np.sqrt(np.mean(np.square(val_vals - fcst_vals)))
#     naive_train_sq_errors = np.square(train_vals[1:] - train_vals[:-1])
#     rmse_naive = np.sqrt(np.mean(naive_train_sq_errors))
#     return rmse_forecast / (rmse_naive + epsilon)


# def calculate_mase(
#     train_vals: np.ndarray,
#     val_vals: np.ndarray,
#     fcst_vals: np.ndarray,
#     epsilon: float = np.finfo(float).eps,
# ) -> float:
#     """Calculates MASE manually using numpy."""
#     mae_forecast = np.mean(np.abs(val_vals - fcst_vals))
#     naive_train_errors = np.abs(train_vals[1:] - train_vals[:-1])
#     mae_naive = np.mean(naive_train_errors)
#     return mae_forecast / (mae_naive + epsilon)


# def calculate_metrics(
#     train: TimeSeries,
#     val: TimeSeries,
#     forecast: TimeSeries,
# ):trics_df
#     """Calculate metrics with error handling."""
#     try:
#         return {
#             "rmse": rmse(val, forecast),
#             "rmsse": calculate_rmsse(
#                 train.values(), val.values(), forecast.values()
#             ),
#             "mae": mae(val, forecast),
#             "mase": calculate_mase(
#                 train.values(), val.values(), forecast.values()
#             ),
#             "smape": smape(val, forecast),
#             "ope": ope(val, forecast),
#         }
#     except Exception as e:
#         logger.warning(f"Error calculating some metrics: {e}")
#         return {
#             "rmse": rmse(val, forecast),
#             "rmsse": np.nan,
#             "mae": mae(val, forecast),
#             "mase": np.nan,
#             "smape": np.nan,
#             "ope": np.nan,
#         }


# def eval_model(
#     model_type: ModelType,
#     model: LocalForecastingModel,
#     store: int,
#     item: int,
#     train: TimeSeries,
#     val: TimeSeries,
#     metrics_df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """Evaluate a model with error handling."""
#     model_name = model_type.value
#     try:
#         logger.info(
#             f"Training {model_name} model, store {store}, item {item}..."
#         )
#         model.fit(train)

#         logger.info(
#             f"Generating forecast with {model_name}, store {store}, item {item}..."
#         )
#         forecast = model.predict(len(val))

#         metrics = calculate_metrics(train, val, forecast)

#         new_row = pd.DataFrame(
#             [
#                 {
#                     "Model": model_name,
#                     "Store": store,
#                     "Item": item,
#                     "RMSE": metrics["rmse"],
#                     "MAE": metrics["mae"],
#                     "SMAPE": metrics["smape"],
#                     "RMSSE": metrics["rmsse"],
#                     "OPE": metrics["ope"],
#                     "MASE": metrics["mase"],
#                 }
#             ]
#         )

#         metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
#         logger.info(
#             f"{model_name} completed successfully, store {store}, item {item}"
#         )

#     except Exception as e:
#         logger.error(f"Error with {model_name}: {e}")
#         new_row = pd.DataFrame(
#             [
#                 {
#                     "Model": model_name,
#                     "Store": store,
#                     "Item": item,
#                     "RMSE": np.nan,
#                     "MAE": np.nan,
#                     "SMAPE": np.nan,
#                     "RMSSE": np.nan,
#                     "OPE": np.nan,
#                     "MASE": np.nan,
#                 }
#             ]
#         )
#         metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

#     return metrics_df


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
        # Load raw data
        logger.info("Loading raw data...")
        df = load_raw_data(data_fn)

        logger.info("Finding unique store-item combinations...")
        unique_combinations = df[["store", "item"]].drop_duplicates()
        if args.N > 0:
            logger.info(f"Limiting to first {args.N} combinations")
            unique_combinations = unique_combinations.head(args.N)

        logger.info(f"Found {len(unique_combinations)} unique combinations")
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
