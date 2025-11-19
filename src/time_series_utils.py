import sys
from pathlib import Path
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import LocalForecastingModel
from darts.metrics import rmse, rmsse, mae, mase, mape, ope, smape
from typing import Optional


# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.utils import (
    get_logger,
)

logger = get_logger(__name__)


def prepare_store_item_series(
    df: pd.DataFrame,
    store: int,
    item: int,
) -> pd.DataFrame:
    """Prepare time series for a specific store-item combination."""

    # Filter data for specific store-item combination
    mask = (df["store"] == store) & (df["item"] == item)
    series_df = df[mask]

    if len(series_df) == 0:
        logger.warning(f"No data for store {store}, item {item}")
        return pd.DataFrame()

    # Sort by date and prepare time series
    series_df = series_df.sort_values("date")

    # Create time series DataFrame
    ts_df = series_df[["date", "growth_rate"]].copy()
    ts_df["growth_rate"] = pd.to_numeric(ts_df["growth_rate"], errors="coerce")
    ts_df = ts_df.set_index("date")
    ts_df = ts_df.replace([np.inf, -np.inf], np.nan)

    return ts_df


def get_train_val_data(
    df: pd.DataFrame,
    store: int,
    item: int,
    split_point: float,
    min_train_data_points: int,
) -> Optional[tuple[TimeSeries, TimeSeries, TimeSeries]]:
    """Split data into training and validation sets."""
    try:
        train_data_for_std = df.iloc[: int(len(df) * split_point)]
        non_missing_count = train_data_for_std["growth_rate"].count()
        logger.info(
            f"store:{store},item:{item},training: Non-missing count: "
            f"{non_missing_count}"
        )
        if non_missing_count < min_train_data_points:
            logger.warning(
                f"store:{store},item:{item},training: Non-missing count:"
                f"({non_missing_count} non-NaN < {min_train_data_points}).Skipping."
            )
            return None, None, None

        # Check variance
        train_std = train_data_for_std["growth_rate"].std()
        if train_std == 0 or np.isnan(train_std):
            logger.warning(
                f"store:{store},item:{item},training: Non-missing count:"
                f"({train_std}).Skipping."
            )
            return None, None, None

        # Convert to Darts TimeSeries
        ts = TimeSeries.from_dataframe(df, fill_missing_dates=True, freq="D")

        # Split data
        train_ts, val_ts = ts.split_before(split_point)

        if len(val_ts) == 0:
            logger.warning(
                f"store:{store},item:{item},validation: No validation data."
            )
            return None, None, None

        # Get the count of *actual* data points, ignoring NaNs
        train_series_pd = train_ts.to_series()
        non_missing_count = train_series_pd.count()
        logger.info(
            f"store:{store},item:{item},training: Non-missing count: "
            f"{non_missing_count}"
        )

        if non_missing_count < min_train_data_points:
            logger.warning(
                f"store:{store},item:{item},training: Non-missing count:"
                f"({non_missing_count} non-NaN < {min_train_data_points}).Skipping."
            )
            return None, None, None

        train_std = train_series_pd.std()
        if train_std == 0 or np.isnan(train_std):
            logger.warning(
                f"store:{store},item:{item},training: Non-missing count:"
                f"({train_std}).Skipping."
            )
            return None, None, None

        non_missing_count = val_ts.to_series().count()
        logger.info(
            f"store:{store},item:{item},validation: Non-missing count: "
            f"{non_missing_count}"
        )

        ts = TimeSeries.from_dataframe(
            df, fill_missing_dates=True, freq="D", fillna_value=0
        )

        # Split the filled series
        train_ts, val_ts = ts.split_before(split_point)

        return ts, train_ts, val_ts

    except Exception as e:
        logger.error(
            f"store:{store},item:{item},Error getting train/val data: {e}"
        )
        return None, None, None


def calculate_rmsse(
    train_vals: np.ndarray,
    val_vals: np.ndarray,
    fcst_vals: np.ndarray,
    epsilon: float = np.finfo(float).eps,
) -> float:
    """Calculates RMSSE manually using numpy."""
    # Input validation
    if train_vals is None or val_vals is None or fcst_vals is None:
        return np.nan

    # Flatten arrays to ensure 1D (handles (n,1) shapes)
    train_vals = np.asarray(train_vals).flatten()
    val_vals = np.asarray(val_vals).flatten()
    fcst_vals = np.asarray(fcst_vals).flatten()

    # Check for empty arrays
    if len(train_vals) == 0 or len(val_vals) == 0 or len(fcst_vals) == 0:
        return np.nan

    # Ensure forecast and validation have same length
    min_len = min(len(val_vals), len(fcst_vals))
    if min_len == 0:
        return np.nan

    val_vals = val_vals[:min_len]
    fcst_vals = fcst_vals[:min_len]

    # Need at least 2 training points for naive forecast
    if len(train_vals) < 2:
        return np.nan

    # Check for NaN values
    if (
        np.any(np.isnan(train_vals))
        or np.any(np.isnan(val_vals))
        or np.any(np.isnan(fcst_vals))
    ):
        logger.warning("NaN values found in input arrays")
        return np.nan

    # Numerator: RMSE of the forecast
    forecast_errors = val_vals - fcst_vals
    rmse_forecast = np.sqrt(np.mean(np.square(forecast_errors)))

    # Denominator: RMSE of the 1-step naive forecast in-sample
    naive_train_sq_errors = np.square(train_vals[1:] - train_vals[:-1])
    rmse_naive = np.sqrt(np.mean(naive_train_sq_errors))

    # Avoid division by zero
    if rmse_naive == 0:
        return np.inf if rmse_forecast > 0 else np.nan

    return rmse_forecast / (rmse_naive + epsilon)


def calculate_mase(
    train_vals: np.ndarray,
    val_vals: np.ndarray,
    fcst_vals: np.ndarray,
    epsilon: float = np.finfo(float).eps,
) -> float:
    """Calculates MASE manually using numpy."""
    # Flatten arrays to ensure 1D
    train_vals = train_vals.flatten()
    val_vals = val_vals.flatten()
    fcst_vals = fcst_vals.flatten()

    # Ensure forecast and validation have same length
    min_len = min(len(val_vals), len(fcst_vals))
    val_vals = val_vals[:min_len]
    fcst_vals = fcst_vals[:min_len]

    # Numerator: MAE of the forecast
    mae_forecast = np.mean(np.abs(val_vals - fcst_vals))

    # Denominator: MAE of the 1-step naive forecast in-sample
    if len(train_vals) < 2:
        return np.nan

    naive_train_errors = np.abs(train_vals[1:] - train_vals[:-1])
    mae_naive = np.mean(naive_train_errors)

    return mae_forecast / (mae_naive + epsilon)


def calculate_metrics(
    train: TimeSeries,
    val: TimeSeries,
    forecast: TimeSeries,
):
    """Calculate metrics with comprehensive error handling."""
    try:
        # Align series first to handle length mismatches
        common_start = max(val.start_time(), forecast.start_time())
        common_end = min(val.end_time(), forecast.end_time())

        val_aligned = val.slice(common_start, common_end)
        forecast_aligned = forecast.slice(common_start, common_end)

        if len(val_aligned) == 0 or len(forecast_aligned) == 0:
            logger.warning(
                "No overlapping data between validation and forecast"
            )
            return {
                k: np.nan
                for k in ["rmse", "rmsse", "mae", "mase", "smape", "ope"]
            }

        # Calculate basic metrics with error handling
        metrics = {}

        try:
            metrics["rmse"] = rmse(val_aligned, forecast_aligned)
        except Exception as e:
            logger.warning(f"RMSE calculation failed: {e}")
            metrics["rmse"] = np.nan

        try:
            metrics["mae"] = mae(val_aligned, forecast_aligned)
        except Exception as e:
            logger.warning(f"MAE calculation failed: {e}")
            metrics["mae"] = np.nan

        try:
            metrics["smape"] = smape(val_aligned, forecast_aligned)
        except Exception as e:
            logger.warning(f"SMAPE calculation failed: {e}")
            metrics["smape"] = np.nan

        try:
            metrics["ope"] = ope(val_aligned, forecast_aligned)
        except Exception as e:
            logger.warning(f"OPE calculation failed: {e}")
            metrics["ope"] = np.nan

        # Calculate custom metrics
        try:
            metrics["rmsse"] = calculate_rmsse(
                train.values(), val_aligned.values(), forecast_aligned.values()
            )
        except Exception as e:
            logger.warning(f"RMSSE calculation failed: {e}")
            metrics["rmsse"] = np.nan

        try:
            metrics["mase"] = calculate_mase(
                train.values(), val_aligned.values(), forecast_aligned.values()
            )
        except Exception as e:
            logger.warning(f"MASE calculation failed: {e}")
            metrics["mase"] = np.nan

        return metrics

    except Exception as e:
        logger.error(f"Error in calculate_metrics: {e}")
        return {
            k: np.nan for k in ["rmse", "rmsse", "mae", "mase", "smape", "ope"]
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

        metrics = calculate_metrics(train, val, forecast)

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
                    "SMAPE": metrics["smape"],
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
                    "SMAPE": np.nan,
                    "OPE": np.nan,
                }
            ]
        )
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    return metrics_df
