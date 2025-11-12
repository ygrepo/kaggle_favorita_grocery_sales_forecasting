import sys
from pathlib import Path
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import LocalForecastingModel
from darts.metrics import rmse, rmsse, mae, mase, mape, ope, smape


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
    series_df = df[mask].copy()

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
) -> tuple[TimeSeries, TimeSeries, TimeSeries] | [None, None, None]:

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
    # Numerator: RMSE of the forecast
    rmse_forecast = np.sqrt(np.mean(np.square(val_vals - fcst_vals)))

    # Denominator: RMSE of the 1-step naive forecast in-sample
    # This is the other part that fails in Darts
    naive_train_sq_errors = np.square(train_vals[1:] - train_vals[:-1])
    rmse_naive = np.sqrt(np.mean(naive_train_sq_errors))

    return rmse_forecast / (rmse_naive + epsilon)


def calculate_mase(
    train_vals: np.ndarray,
    val_vals: np.ndarray,
    fcst_vals: np.ndarray,
    epsilon: float = np.finfo(float).eps,
) -> float:
    """Calculates MASE manually using numpy."""
    # Numerator: MAE of the forecast
    mae_forecast = np.mean(np.abs(val_vals - fcst_vals))

    # Denominator: MAE of the 1-step naive forecast in-sample
    # This is the part that fails in Darts
    naive_train_errors = np.abs(train_vals[1:] - train_vals[:-1])
    mae_naive = np.mean(naive_train_errors)

    return mae_forecast / (mae_naive + epsilon)


def calculate_metrics(
    train: TimeSeries,
    val: TimeSeries,
    forecast: TimeSeries,
):
    """Calculate metrics with error handling."""
    try:
        # logger.info(f"train: {train}")
        # logger.info(f"val: {val}")
        # logger.info(f"forecast: {forecast}")
        return {
            "rmse": rmse(val, forecast),
            "rmsse": calculate_rmsse(
                train.values(), val.values(), forecast.values()
            ),
            "mae": mae(val, forecast),
            "mase": calculate_mase(
                train.values(), val.values(), forecast.values()
            ),
            "smape": smape(val, forecast),
            "ope": ope(val, forecast),
        }
    except Exception as e:
        logger.warning(f"Error calculating some metrics: {e}")
        # Return basic metrics only
        return {
            "rmse": rmse(val, forecast),
            "rmsse": np.nan,
            "mae": mae(val, forecast),
            "mase": np.nan,
            "smape": np.nan,
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
