import sys
from pathlib import Path
from enum import Enum
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.metrics import rmse, mae, ope, smape, marre
from sklearn.preprocessing import RobustScaler
from darts.dataprocessing.transformers import Scaler
import traceback

# Classical models
from darts.models import (
    AutoARIMA,
    ExponentialSmoothing,
    Theta,
    KalmanForecaster,
)
from darts.utils.utils import SeasonalityMode

# Deep learning models
from darts.models import (
    NBEATSModel,
    TFTModel,
    TSMixerModel,
    BlockRNNModel,
    TCNModel,
    TiDEModel,
)

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


# =====================================================================
# Unified ModelType Enum and Factory
# =====================================================================


class ModelType(str, Enum):
    """Unified enum for all supported time series models."""

    # Classical models
    EXPONENTIAL_SMOOTHING = "EXPONENTIAL_SMOOTHING"
    AUTO_ARIMA = "AUTO_ARIMA"
    THETA = "THETA"
    KALMAN = "KALMAN"
    # Deep learning models
    NBEATS = "NBEATS"
    TFT = "TFT"
    TSMIXER = "TSMIXER"
    BLOCK_RNN = "BLOCK_RNN"
    TCN = "TCN"
    TIDE = "TIDE"


def parse_models_arg(models_string: str) -> List[ModelType]:
    """
    Convert comma-separated model names to ModelType list.
    Example: "EXPONENTIAL_SMOOTHING,AUTO_ARIMA" -> [ModelType.EXPONENTIAL_SMOOTHING, ModelType.AUTO_ARIMA]
    """
    names = [m.strip().upper() for m in models_string.split(",") if m.strip()]
    try:
        return [ModelType(name) for name in names]
    except ValueError as e:
        raise ValueError(
            f"Invalid models argument: {models_string}. "
            f"Valid options: {[m.value for m in ModelType]}"
        ) from e


def create_model(
    model_type: ModelType,
    batch_size: int = 64,
    torch_kwargs: Optional[Dict[str, Any]] = None,
    n_epochs: int = 300,
    dropout: float = 0.5,
) -> ForecastingModel:
    """
    Factory for classical and deep learning models.

    Deep learning configurations are 'medium-sized' and explicitly tuned for:
      - short per-series data (~200–300 time steps),
      - up to 300 epochs with early stopping,
      - strong regularization (dropout + weight decay),
      - no global parameter sharing (one model per (store, item)).
    """
    if torch_kwargs is None:
        torch_kwargs = {}

    # =====================================================================
    # Classical Models
    # =====================================================================
    if model_type == ModelType.EXPONENTIAL_SMOOTHING:
        return ExponentialSmoothing()

    if model_type == ModelType.AUTO_ARIMA:
        return AutoARIMA()

    if model_type == ModelType.THETA:
        return Theta(season_mode=SeasonalityMode.ADDITIVE)

    if model_type == ModelType.KALMAN:
        return KalmanForecaster(dim_x=1, random_state=42)

    # =====================================================================
    # Deep Learning Models (per-series, medium capacity + strong reg)
    # =====================================================================

    # Shared configuration for all DL models
    base_kwargs = dict(
        input_chunk_length=40,  # ~6 weeks of context for next-day growth
        output_chunk_length=1,  # 1-step ahead forecasting
        n_epochs=n_epochs,
        batch_size=batch_size,
        random_state=42,
        save_checkpoints=False,
        force_reset=True,
        optimizer_kwargs={
            "lr": 1e-3,
            "weight_decay": 1e-4,  # important for per-series overfitting control
        },
        **torch_kwargs,
    )

    # -------------------------
    # N-BEATS (medium)
    # -------------------------
    if model_type == ModelType.NBEATS:
        # Medium architecture: smaller than your original (10 stacks x 512),
        # but not tiny. Good compromise for short per-series data.
        return NBEATSModel(
            generic_architecture=True,
            num_stacks=4,  # medium: more expressiveness than 2, far less than 10
            num_blocks=1,
            num_layers=2,
            layer_widths=128,  # medium width
            # dropout is supported in newer Darts; if your version does not support it,
            # you can safely remove this argument.
            dropout=dropout,
            **base_kwargs,
        )

    # -------------------------
    # TFT (medium)
    # -------------------------
    if model_type == ModelType.TFT:
        # Medium-size TFT: enough capacity to model covariates + nonlinearity,
        # but not the huge original configs from papers.
        return TFTModel(
            hidden_size=32,  # medium
            lstm_layers=1,  # deep stacks are overkill per-series
            dropout=dropout,
            num_attention_heads=4,  # medium number of heads
            add_relative_index=True,
            **base_kwargs,
        )

    # -------------------------
    # TSMixer (medium)
    # -------------------------
    if model_type == ModelType.TSMIXER:
        # TSMixer is lightweight; a moderate hidden_size is cheap
        # and handles multiple covariates well.
        return TSMixerModel(
            hidden_size=64,  # medium
            dropout=dropout,
            **base_kwargs,
        )

    # -------------------------
    # BlockRNN (medium)
    # -------------------------
    if model_type == ModelType.BLOCK_RNN:
        # Modest LSTM capacity; more than tiny, smaller than your 64x2 config.
        return BlockRNNModel(
            model="LSTM",
            hidden_dim=32,  # medium
            n_rnn_layers=1,  # deeper layers are not justified per-series
            dropout=dropout,
            **base_kwargs,
        )

    # -------------------------
    # TCN (medium)
    # -------------------------
    if model_type == ModelType.TCN:
        # TCN with moderate number of channels; good at local patterns
        # without exploding parameter count.
        return TCNModel(
            kernel_size=3,
            num_filters=32,  # medium (between 16 and 64)
            dilation_base=2,
            weight_norm=True,
            dropout=dropout,
            **base_kwargs,
        )

    # -------------------------
    # TiDE (medium)
    # -------------------------
    if model_type == ModelType.TIDE:
        # TiDE with medium hidden size; handles many covariates robustly.
        return TiDEModel(
            hidden_size=64,  # medium
            dropout=dropout,
            use_layer_norm=True,
            **base_kwargs,
        )

    raise ValueError(f"Unsupported model type: {model_type}")


# def create_model(
#     model_type: ModelType,
#     batch_size: int = 32,
#     torch_kwargs: Optional[Dict[str, Any]] = None,
#     n_epochs: int = 100,
#     dropout: float = 0.1,
# ) -> ForecastingModel:
#     """
#     Factory to create a Darts model instance (classical or deep learning).

#     Args:
#         model_type: ModelType enum value
#         batch_size: Batch size for deep learning models
#         torch_kwargs: PyTorch/Lightning kwargs for deep learning models
#         n_epochs: Number of epochs for deep learning models
#         dropout: Dropout rate for deep learning models

#     Returns:
#         ForecastingModel instance
#     """
#     if torch_kwargs is None:
#         torch_kwargs = {}

#     # =====================================================================
#     # Classical Models (no batch_size, epochs, or torch_kwargs needed)
#     # =====================================================================

#     if model_type == ModelType.EXPONENTIAL_SMOOTHING:
#         return ExponentialSmoothing()

#     if model_type == ModelType.AUTO_ARIMA:
#         return AutoARIMA()

#     if model_type == ModelType.THETA:
#         return Theta(season_mode=SeasonalityMode.ADDITIVE)

#     if model_type == ModelType.KALMAN:
#         return KalmanForecaster(dim_x=1, random_state=42)

#     # =====================================================================
#     # Deep Learning Models
#     # =====================================================================

#     # Base configuration shared by all deep learning models
#     base_kwargs = dict(
#         input_chunk_length=30,
#         output_chunk_length=1,
#         n_epochs=n_epochs,
#         batch_size=batch_size,
#         random_state=42,
#         save_checkpoints=False,
#         force_reset=True,
#         **torch_kwargs,
#     )

#     if model_type == ModelType.NBEATS:
#         return NBEATSModel(
#             generic_architecture=True,
#             num_stacks=10,
#             num_blocks=1,
#             num_layers=4,
#             layer_widths=512,
#             **base_kwargs,
#         )

#     if model_type == ModelType.TFT:
#         return TFTModel(
#             hidden_size=64,
#             lstm_layers=1,
#             dropout=dropout,
#             num_attention_heads=4,
#             add_relative_index=True,
#             **base_kwargs,
#         )

#     if model_type == ModelType.TSMIXER:
#         return TSMixerModel(
#             hidden_size=64,
#             dropout=dropout,
#             **base_kwargs,
#         )

#     if model_type == ModelType.BLOCK_RNN:
#         return BlockRNNModel(
#             model="LSTM",
#             hidden_dim=64,
#             n_rnn_layers=2,
#             dropout=dropout,
#             **base_kwargs,
#         )

#     if model_type == ModelType.TCN:
#         return TCNModel(
#             kernel_size=3,
#             num_filters=64,
#             dilation_base=2,
#             weight_norm=True,
#             dropout=dropout,
#             **base_kwargs,
#         )

#     if model_type == ModelType.TIDE:
#         return TiDEModel(
#             hidden_size=64,
#             dropout=dropout,
#             use_layer_norm=True,
#             **base_kwargs,
#         )

#     raise ValueError(f"Unsupported model type: {model_type}")


# def prepare_store_item_series(
#     df: pd.DataFrame,
#     store: int,
#     item: int,
#     store_medians_fn: Path,
#     item_medians_fn: Path,
#     store_assign_fn: Path,
#     item_assign_fn: Path,
# ) -> pd.DataFrame:
#     """
#     Memory-safe preparation of the (store, item) time series.
#     Loads only the cluster-median rows needed for this specific (store, item).
#     """

#     # ----------------------------------------------------------------------
#     # Extract base series (already filtered)
#     # ----------------------------------------------------------------------
#     mask = (df["store"] == store) & (df["item"] == item)
#     series_df = df[mask].copy()

#     if series_df.empty:
#         logger.warning(f"No data for store {store}, item {item}")
#         return pd.DataFrame()

#     series_df = series_df.sort_values("date")
#     dates = series_df["date"].unique()

#     # ----------------------------------------------------------------------
#     # Load cluster assignments only for the current store and item
#     # ----------------------------------------------------------------------
#     store_assign = pd.read_csv(store_assign_fn)
#     item_assign = pd.read_csv(item_assign_fn)

#     store_clusters = (
#         store_assign.loc[store_assign["store"] == store, "cluster_id"]
#         .drop_duplicates()
#         .tolist()
#     )

#     item_clusters = (
#         item_assign.loc[item_assign["item"] == item, "cluster_id"]
#         .drop_duplicates()
#         .tolist()
#     )

#     # ----------------------------------------------------------------------
#     # Load only the necessary cluster median rows
#     #  This avoids loading giant full-sized tables.
#     # ----------------------------------------------------------------------
#     store_medians = pd.read_parquet(
#         store_medians_fn,
#         filters=[
#             ("store_cluster_id", "in", store_clusters),
#             ("date", "in", dates.tolist()),
#         ],
#         columns=["date", "store_cluster_id", "store_cluster_median"],
#     )

#     item_medians = pd.read_parquet(
#         item_medians_fn,
#         filters=[
#             ("item_cluster_id", "in", item_clusters),
#             ("date", "in", dates.tolist()),
#         ],
#         columns=["date", "item_cluster_id", "item_cluster_median"],
#     )

#     # ----------------------------------------------------------------------
#     # Pivot cluster medians wide (1 row per date)
#     # ----------------------------------------------------------------------
#     if not store_medians.empty:
#         store_medians_wide = (
#             store_medians.pivot(
#                 index="date",
#                 columns="store_cluster_id",
#                 values="store_cluster_median",
#             )
#             .add_prefix("store_cluster_median_")
#             .reset_index()
#         )
#         series_df = series_df.merge(store_medians_wide, on="date", how="left")

#     if not item_medians.empty:
#         item_medians_wide = (
#             item_medians.pivot(
#                 index="date",
#                 columns="item_cluster_id",
#                 values="item_cluster_median",
#             )
#             .add_prefix("item_cluster_median_")
#             .reset_index()
#         )
#         series_df = series_df.merge(item_medians_wide, on="date", how="left")

#     # ----------------------------------------------------------------------
#     # Select covariates
#     # ----------------------------------------------------------------------
#     # base cov lists
#     base_future = [c for c in FUTURE_COV_COLS if c in series_df.columns]
#     base_past = [c for c in PAST_COV_COLS if c in series_df.columns]

#     # dynamically detect cluster future covariates
#     cluster_cols = [
#         c
#         for c in series_df.columns
#         if c.startswith("store_cluster_median_")
#         or c.startswith("item_cluster_median_")
#     ]

#     available_future = sorted(set(base_future + cluster_cols))

#     cols_to_keep = ["date", TARGET_COL] + available_future + base_past
#     ts_df = series_df[cols_to_keep].copy()

#     # (rest of your function unchanged)
#     for col in ts_df.columns:
#         if col != "date":
#             ts_df[col] = pd.to_numeric(ts_df[col], errors="coerce")

#     ts_df = ts_df.set_index("date")
#     ts_df = ts_df.replace([np.inf, -np.inf], np.nan)

#     cov_cols = available_future + base_past
#     if cov_cols:
#         ts_df[cov_cols] = ts_df[cov_cols].fillna(0)

#     return ts_df


# def get_train_val_data_with_covariates(
#     ts_df: pd.DataFrame,
#     store: int,
#     item: int,
#     split_point: float,
#     min_train_data_points: int,
# ) -> Optional[Dict[str, Any]]:
#     """
#     Creates Darts TimeSeries objects for Target, Past Covariates, and Future Covariates,
#     and splits them consistently into train/validation sets.

#     Future covariates are now determined dynamically:
#     - Calendar covariates listed in FUTURE_COV_COLS (static)
#     - Cluster medians (store_cluster_median_*, item_cluster_median_*)
#     - PARAFAC latent factors (parafac_*), or any other *_covariate_* prefix.
#     """

#     try:
#         # ------------------------------------------------------------------
#         # Basic validation of target availability
#         # ------------------------------------------------------------------
#         total_len = len(ts_df)
#         train_len_approx = int(total_len * split_point)

#         train_df_subset = ts_df.iloc[:train_len_approx]
#         non_missing_target = train_df_subset[TARGET_COL].count()

#         if non_missing_target < min_train_data_points:
#             logger.warning(
#                 f"S{store}/I{item}: Insufficient training data "
#                 f"({non_missing_target} < {min_train_data_points}). Skipping."
#             )
#             return None

#         train_std = train_df_subset[TARGET_COL].std()
#         if train_std == 0 or np.isnan(train_std):
#             logger.warning(
#                 f"S{store}/I{item}: Target has zero or NaN variance. Skipping."
#             )
#             return None

#         # ------------------------------------------------------------------
#         # Build full TimeSeries from DataFrame
#         # ------------------------------------------------------------------
#         full_ts = TimeSeries.from_dataframe(
#             ts_df, fill_missing_dates=True, freq="D", fillna_value=0
#         )

#         # Target
#         target_ts = full_ts[TARGET_COL]

#         # ------------------------------------------------------------------
#         # Past covariates (static list)
#         # ------------------------------------------------------------------
#         valid_p_cols = [c for c in PAST_COV_COLS if c in ts_df.columns]
#         past_covs_ts = full_ts[valid_p_cols] if valid_p_cols else None

#         # ------------------------------------------------------------------
#         # Dynamic detection of future covariates
#         # ------------------------------------------------------------------

#         # Static calendar-based covariates
#         base_future_cols = [c for c in FUTURE_COV_COLS if c in ts_df.columns]

#         # Dynamic cluster / PARAFAC covariates
#         dynamic_future_cols = [
#             c
#             for c in ts_df.columns
#             if (
#                 c.startswith("store_cluster_median_")
#                 or c.startswith("item_cluster_median_")
#             )
#         ]

#         # Merge both lists, ensure uniqueness
#         valid_f_cols = sorted(set(base_future_cols + dynamic_future_cols))

#         future_covs_ts = full_ts[valid_f_cols] if len(valid_f_cols) else None

#         # ------------------------------------------------------------------
#         # Train–Validation Split
#         # ------------------------------------------------------------------
#         train_target, val_target = target_ts.split_before(split_point)

#         if len(train_target) == 0 or len(val_target) == 0:
#             logger.warning(
#                 f"S{store}/I{item}: Empty train or validation split."
#             )
#             return None

#         # Past covariates
#         if past_covs_ts is not None:
#             train_past, val_past = past_covs_ts.split_before(split_point)
#         else:
#             train_past, val_past = None, None

#         # Future covariates
#         if future_covs_ts is not None:
#             train_future, val_future = future_covs_ts.split_before(split_point)
#         else:
#             train_future, val_future = None, None

#         # ------------------------------------------------------------------
#         # 6. Logging
#         # ------------------------------------------------------------------
#         logger.info(
#             f"S{store}/I{item}: Data prepared. "
#             f"Train={len(train_target)}, Val={len(val_target)}, "
#             f"Past Dim={past_covs_ts.n_components if past_covs_ts else 0}, "
#             f"Future Dim={future_covs_ts.n_components if future_covs_ts else 0}"
#         )

#         # ------------------------------------------------------------------
#         # 7. Return time series dictionary
#         # ------------------------------------------------------------------
#         return {
#             "train_target": train_target,
#             "val_target": val_target,
#             "train_past": train_past,
#             "val_past": val_past,
#             "train_future": train_future,
#             "val_future": val_future,
#             "full_past": past_covs_ts,
#             "full_future": future_covs_ts,
#         }

#     except Exception as e:
#         logger.error(
#             f"store:{store},item:{item},Error preparing train/val data: {e}"
#         )
#         logger.error(traceback.format_exc())
#         return None


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

    # Which models support validation series during training (deep learning models)
    supports_val_series = modelType in {
        "NBEATS",
        "TFT",
        "TSMIXER",
        "BLOCK_RNN",
        "TCN",
        "TIDE",
        "TRANSFORMER",
        "NHITS",
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

        if (
            supports_past
            and data_dict.get("train_past") is not None
            and not no_past_covs
        ):
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
            and not no_future_covs
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

        # Start with basic fit arguments
        fit_kwargs: Dict[str, Any] = {
            "series": train_target_scaled,
        }

        # Add validation series only for models that support it
        if supports_val_series:
            fit_kwargs["val_series"] = val_target_scaled

        # Add past covariates if supported
        if (
            supports_past
            and train_past_scaled is not None
            and not no_past_covs
        ):
            fit_kwargs["past_covariates"] = train_past_scaled
            if supports_val_series and val_past_scaled is not None:
                fit_kwargs["val_past_covariates"] = val_past_scaled

        # Add future covariates if supported
        if (
            supports_future
            and future_covs_scaled_full is not None
            and not no_future_covs
        ):
            fit_kwargs["future_covariates"] = future_covs_scaled_full
            if supports_val_series and val_future_scaled is not None:
                fit_kwargs["val_future_covariates"] = val_future_scaled

        logger.debug(f"Fit kwargs keys: {list(fit_kwargs.keys())}")
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

        logger.debug(f"Predict kwargs keys: {list(predict_kwargs.keys())}")
        forecast_scaled = model.predict(**predict_kwargs)

        # --- INVERSE TRANSFORM & METRICS ---
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
        logger.error(traceback.format_exc())  # Uncomment for debugging
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
