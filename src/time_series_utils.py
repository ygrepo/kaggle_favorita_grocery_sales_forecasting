import sys
from pathlib import Path
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.metrics import rmse, mae, ope, smape, marre
from typing import Optional, Dict, Any
from sklearn.preprocessing import RobustScaler
from darts.dataprocessing.transformers import Scaler
import traceback

# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.utils import (
    get_logger,
)

logger = get_logger(__name__)

# Define Covariate Groups globally for easy modification
# Future: Known in advance (calendar features)
# Note: Raw 'dayofweek', 'weekofmonth', 'monthofyear' excluded to avoid multicollinearity with sin/cos versions.
FUTURE_COV_COLS = [
    "dayofweek_sin",
    "dayofweek_cos",
    "weekofmonth_sin",
    "weekofmonth_cos",
    "monthofyear_sin",
    "monthofyear_cos",
    "paycycle_sin",
    "paycycle_cos",
    "season_sin",
    "season_cos",
]

# Past: Only known up to present time (lagged/rolling features of target)
PAST_COV_COLS = [
    "unit_sales_rolling_median",
    "unit_sales_ewm_decay",
    "growth_rate_rolling_median",
    "growth_rate_ewm_decay",
]

TARGET_COL = "growth_rate"


def prepare_store_item_series(
    df: pd.DataFrame,
    store: int,
    item: int,
) -> pd.DataFrame:
    """
    Prepare time series DataFrame including target and all available covariates
    for a specific store-item combination.
    """

    mask = (df["store"] == store) & (df["item"] == item)
    series_df = df[mask].copy()

    if len(series_df) == 0:
        logger.warning(f"No data for store {store}, item {item}")
        return pd.DataFrame()

    series_df = series_df.sort_values("date")

    # Determine which requested covariate columns actually exist in the data
    available_future = [c for c in FUTURE_COV_COLS if c in series_df.columns]
    available_past = [c for c in PAST_COV_COLS if c in series_df.columns]

    cols_to_keep = ["date", TARGET_COL] + available_future + available_past

    ts_df = series_df[cols_to_keep].copy()

    # Ensure numeric types
    for col in ts_df.columns:
        if col != "date":
            ts_df[col] = pd.to_numeric(ts_df[col], errors="coerce")

    ts_df = ts_df.set_index("date")
    ts_df = ts_df.replace([np.inf, -np.inf], np.nan)

    # Handle NaNs:
    # 1. Target: Darts handles internal NaNs reasonably well, but leading/trailing might be issues.
    # 2. Covariates: Must be filled. A safe default for standardized features is 0 (mean).
    cov_cols = available_future + available_past
    if cov_cols:
        # Fill covariate NaNs with 0 (assuming they will be robustly scaled later, 0 is reasonable center)
        ts_df[cov_cols] = ts_df[cov_cols].fillna(0)

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


def get_train_val_data_with_covariates(
    ts_df: pd.DataFrame,
    store: int,
    item: int,
    split_point: float,
    min_train_data_points: int,
) -> Optional[Dict[str, Any]]:
    """
    Creates Darts TimeSeries objects for Target, Past Covariates, and Future Covariates,
    and splits them consistently into train/validation sets.
    """
    try:
        # Check minimum data requirements on the raw dataframe first
        total_len = len(ts_df)
        train_len_approx = int(total_len * split_point)

        train_df_subset = ts_df.iloc[:train_len_approx]
        non_missing_target = train_df_subset[TARGET_COL].count()

        if non_missing_target < min_train_data_points:
            logger.warning(
                f"S{store}/I{item}: Insufficient training data ({non_missing_target} < {min_train_data_points}). Skipping."
            )
            return None

        train_std = train_df_subset[TARGET_COL].std()
        if train_std == 0 or np.isnan(train_std):
            logger.warning(
                f"S{store}/I{item}: Target has zero or NaN variance in training set. Skipping."
            )
            return None

        # --- Create Darts TimeSeries ---
        full_ts = TimeSeries.from_dataframe(
            ts_df, fill_missing_dates=True, freq="D", fillna_value=0
        )

        # Target
        target_ts = full_ts[TARGET_COL]

        # Past covariates
        valid_p_cols = [c for c in PAST_COV_COLS if c in ts_df.columns]
        past_covs_ts = full_ts[valid_p_cols] if valid_p_cols else None

        # Future covariates
        valid_f_cols = [c for c in FUTURE_COV_COLS if c in ts_df.columns]
        future_covs_ts = full_ts[valid_f_cols] if valid_f_cols else None

        # --- Split target ---
        train_target, val_target = target_ts.split_before(split_point)

        if len(val_target) == 0 or len(train_target) == 0:
            logger.warning(
                f"S{store}/I{item}: Data split resulted in empty train or validation set."
            )
            return None

        # --- Split covariates consistently ---
        train_past, val_past = (None, None)
        if past_covs_ts is not None:
            train_past, val_past = past_covs_ts.split_before(split_point)

        train_future, val_future = (None, None)
        if future_covs_ts is not None:
            train_future, val_future = future_covs_ts.split_before(split_point)

        logger.info(
            f"S{store}/I{item}: Data prepared. Train len: {len(train_target)}, Val len: {len(val_target)}"
        )
        if past_covs_ts is not None:
            logger.debug(f"Past covs dim: {past_covs_ts.n_components}")
        if future_covs_ts is not None:
            logger.debug(f"Future covs dim: {future_covs_ts.n_components}")

        return {
            "train_target": train_target,
            "val_target": val_target,
            "train_past": train_past,
            "val_past": val_past,
            "train_future": train_future,
            "val_future": val_future,
            "full_past": past_covs_ts,
            "full_future": future_covs_ts,
        }

    except Exception as e:
        logger.error(
            f"store:{store},item:{item},Error preparing train/val data: {e}"
        )
        logger.error(traceback.format_exc())
        return None


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
        # logger.warning("NaN values found in input arrays for RMSSE")
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

        try:
            metrics["marre"] = marre(val_aligned, forecast_aligned)
        except Exception as e:
            logger.warning(f"MARRE calculation failed: {e}")
            metrics["marre"] = np.nan

        return metrics

    except Exception as e:
        logger.error(f"Error in calculate_metrics: {e}")
        return {
            k: np.nan
            for k in ["rmsse", "mse", "smape", "marre", "rmse", "mae", "ope"]
        }


def eval_model(
    modelType: str,
    model: ForecastingModel,
    store: int,
    item: int,
    train: TimeSeries,
    val: TimeSeries,
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate a model with error handling and automatic RobustScaling."""
    try:
        logger.info(
            f"Training {modelType} model, store {store}, item {item}..."
        )

        # START SCALING LOGIC
        # Initialize and fit the scaler on training data
        scaler = Scaler(RobustScaler())
        train_scaled = scaler.fit_transform(train)

        # Fit model on SCALED data (prevents Kalman explosion)
        model.fit(train_scaled)

        logger.info(f"Generating forecast with {modelType}...")

        # Predict (returns scaled output)
        forecast_scaled = model.predict(len(val))

        # Inverse transform to get Real Units back
        forecast = scaler.inverse_transform(forecast_scaled)
        # END SCALING LOGIC

        # Calculate metrics using Real Units (train, val, forecast are all unscaled now)
        metrics = calculate_metrics(train, val, forecast)

        new_row = pd.DataFrame(
            [
                {
                    "Model": modelType,
                    "Store": store,
                    "Item": item,
                    "RMSSE": metrics["rmsse"],
                    "MASE": metrics["mase"],
                    "SMAPE": metrics["smape"],
                    "MARRE": metrics["marre"],
                    "RMSE": metrics["rmse"],
                    "MAE": metrics["mae"],
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
                    "RMSSE": np.nan,
                    "MASE": np.nan,
                    "SMAPE": np.nan,
                    "MARRE": np.nan,
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "OPE": np.nan,
                }
            ]
        )
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    return metrics_df


def eval_model_with_covariates(
    modelType: str,
    model: ForecastingModel,
    store: int,
    item: int,
    data_dict: Dict[str, Any],
    metrics_df: pd.DataFrame,
    no_past_covs: bool = False,
    no_future_covs: bool = False,
) -> pd.DataFrame:
    """
    Evaluate a model handling scaling for targets and covariates independently,
    ensuring no data leakage from validation into training via scalers.
    """

    # Which models support which types of covariates
    supports_past = modelType in {
        "NBEATS",
        "TFT",
        "TSMIXER",
        "BLOCK_RNN",
        "TCN",
        "TIDE",
    }
    supports_future = modelType in {
        "TFT",
        "TSMIXER",
        "TIDE",
    } 

    try:
        train_target = data_dict["train_target"]
        val_target = data_dict["val_target"]
        forecast_horizon = len(val_target)

        # --- TARGET SCALER ---
        target_scaler = Scaler(RobustScaler())
        train_target_scaled = target_scaler.fit_transform(train_target)
        val_target_scaled = target_scaler.transform(val_target)

        # --- PAST COVARIATES SCALER ---
        train_past_scaled = None
        val_past_scaled = None
        past_covs_scaled_full = None

        if supports_past and data_dict.get("train_past") is not None:
            past_scaler = Scaler(RobustScaler())

            # Fit on training past covariates
            train_past_scaled = past_scaler.fit_transform(
                data_dict["train_past"]
            )

            # Transform validation slice if present
            if data_dict.get("val_past") is not None:
                val_past_scaled = past_scaler.transform(data_dict["val_past"])

            # Transform the full past covariate history for prediction
            if data_dict.get("full_past") is not None:
                past_covs_scaled_full = past_scaler.transform(
                    data_dict["full_past"]
                )

        # --- FUTURE COVARIATES SCALER ---
        future_covs_scaled_full = None
        val_future_scaled = None
        if (
            supports_future
            and data_dict.get("full_future") is not None
            and data_dict.get("train_future") is not None
        ):
            future_scaler = Scaler(RobustScaler())
            # Fit on train_future only (no leakage)
            future_scaler.fit(data_dict["train_future"])
            # Transform full history and validation slice
            future_covs_scaled_full = future_scaler.transform(
                data_dict["full_future"]
            )
            if data_dict.get("val_future") is not None:
                val_future_scaled = future_scaler.transform(
                    data_dict["val_future"]
                )

        # --- MODEL FITTING ---
        logger.debug(f"Fitting {modelType} S{store}/I{item}...")

        fit_kwargs: Dict[str, Any] = {
            "series": train_target_scaled,
            "val_series": val_target_scaled,
        }

        if (
            supports_past
            and train_past_scaled is not None
            and not no_past_covs
        ):
            fit_kwargs["past_covariates"] = train_past_scaled
            if val_past_scaled is not None:
                fit_kwargs["val_past_covariates"] = val_past_scaled

        if (
            supports_future
            and future_covs_scaled_full is not None
            and not no_future_covs
        ):
            # Darts uses the history; for validation consistency, pass val slice too
            fit_kwargs["future_covariates"] = future_covs_scaled_full
            if val_future_scaled is not None:
                fit_kwargs["val_future_covariates"] = val_future_scaled

        model.fit(**fit_kwargs)

        # --- FORECASTING ---
        logger.debug(
            f"Predicting {modelType} S{store}/I{item} (n={forecast_horizon})..."
        )

        predict_kwargs: Dict[str, Any] = {"n": forecast_horizon}

        # IMPORTANT: for prediction, covariates must extend up to the forecast horizon
        if (
            supports_past
            and past_covs_scaled_full is not None
            and not no_past_covs
        ):
            predict_kwargs["past_covariates"] = past_covs_scaled_full

        if (
            supports_future
            and future_covs_scaled_full is not None
            and not no_future_covs
        ):
            predict_kwargs["future_covariates"] = future_covs_scaled_full

        forecast_scaled = model.predict(**predict_kwargs)

        # --- 6. INVERSE TRANSFORM & METRICS ---
        forecast = target_scaler.inverse_transform(forecast_scaled)

        metrics = calculate_metrics(train_target, val_target, forecast)

        new_row_dict = {
            "Model": modelType,
            "Store": store,
            "Item": item,
            "RMSSE": metrics["rmsse"],
            "MASE": metrics["mase"],
            "SMAPE": metrics["smape"],
            "MARRE": metrics["marre"],
            "RMSE": metrics["rmse"],
            "MAE": metrics["mae"],
            "OPE": metrics["ope"],
        }

        new_row = pd.DataFrame([new_row_dict])

        # Downcast floats to save memory
        cols_to_downcast = [
            c for c in new_row.columns if c not in ["Model", "Store", "Item"]
        ]
        for col in cols_to_downcast:
            new_row[col] = pd.to_numeric(new_row[col], downcast="float")

        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
        logger.info(
            f"FINISHED {modelType} S{store}/I{item}. SMAPE: {metrics['smape']:.4f}"
        )

    except Exception as e:
        logger.error(
            f"Error fitting/evaluating {modelType} for S{store}/I{item}: {e}"
        )
        # logger.error(traceback.format_exc())
        nan_row_dict = {
            "Model": modelType,
            "Store": store,
            "Item": item,
            "RMSSE": np.nan,
            "MASE": np.nan,
            "SMAPE": np.nan,
            "MARRE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "OPE": np.nan,
        }
        new_row = pd.DataFrame([nan_row_dict])
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    return metrics_df
