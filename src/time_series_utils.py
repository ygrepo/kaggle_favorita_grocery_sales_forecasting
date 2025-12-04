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
    LightGBMModel,
    RandomForestModel,
    LinearRegressionModel,
    XGBModel,
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
    read_csv_or_parquet,
)

logger = get_logger(__name__)

# Static covariates: constant for a given (store, item) series
STATIC_COV_COLS = [
    "store",
    "item",
    "city_id",
    "state_id",
    "cluster_id",
    "class_id",
    "type_B",
    "type_C",
    "type_D",
    "type_E",
]


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
    "is_holiday_any",
    "is_holiday_national",
    "is_holiday_regional",
    "is_holiday_local",
]

# Past: Only known up to present time (lagged/rolling features of target)
PAST_COV_COLS = [
    "unit_sales_rolling_median",
    "unit_sales_ewm_decay",
    "growth_rate_rolling_median",
    "growth_rate_ewm_decay",
    "transactions",
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
    # Tree-based regression models
    LIGHTGBM = "LIGHTGBM"
    RANDOM_FOREST = "RANDOM_FOREST"
    LINEAR_REGRESSION = "LINEAR_REGRESSION"
    XGBOOST = "XGBOOST"
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


# Helper: choose lags depending on xl_design and covariate flags
def _lags_for_regression(
    xl_design: bool, past_covs: bool, future_covs: bool
) -> tuple[int, Optional[int], Optional[tuple[int, int]]]:
    if xl_design:
        lags = 60
        lags_past = 60 if past_covs else None
        # Use only same-time or past "future covariates" (no positive lags)
        lags_future = (15, 0) if future_covs else None  # lags -15..-1
    else:
        lags = 30
        lags_past = 30 if past_covs else None
        lags_future = (15, 0) if future_covs else None  # lags -15..-1
    return lags, lags_past, lags_future


def create_model(
    model_type: ModelType,
    batch_size: int = 32,
    torch_kwargs: Optional[Dict[str, Any]] = None,
    n_epochs: int = 100,
    dropout: float = 0.1,
    xl_design: bool = False,
    past_covs: bool = False,
    future_covs: bool = False,
) -> ForecastingModel:
    """
    Factory to create a Darts model instance (classical or deep learning).

    Args:
        model_type: ModelType enum value.
        batch_size: Batch size for deep learning models.
        torch_kwargs: PyTorch/Lightning kwargs for deep learning models.
        n_epochs: Number of epochs for deep learning models.
        dropout: Dropout rate for deep learning models.
        xl_design: If True, use larger 'XL' deep learning architectures.
                   If False, use the medium-sized architectures.

    Returns:
        ForecastingModel instance.
    """
    if torch_kwargs is None:
        torch_kwargs = {}

    # =====================================================================
    # Classical Models (no batch_size, epochs, or torch_kwargs needed)
    # =====================================================================

    logger.info(
        f"Creating model: {model_type.value}, past_covs={past_covs}, future_covs={future_covs}, xl_design={xl_design}"
    )
    if model_type == ModelType.EXPONENTIAL_SMOOTHING:
        return ExponentialSmoothing()

    if model_type == ModelType.AUTO_ARIMA:
        return AutoARIMA()

    if model_type == ModelType.THETA:
        return Theta(season_mode=SeasonalityMode.ADDITIVE)

    if model_type == ModelType.KALMAN:
        return KalmanForecaster(dim_x=1, random_state=42)

    lags, lags_past, lags_future = _lags_for_regression(
        xl_design, past_covs, future_covs
    )

    if model_type == ModelType.LINEAR_REGRESSION:
        if xl_design:
            # XL: using longer history and more covariate lags
            return LinearRegressionModel(
                lags=lags,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                output_chunk_length=1,
                use_static_covariates=True,
                # Linear regression does not have hyperparameters besides fit_intercept, etc.
            )
        else:
            # Medium
            return LinearRegressionModel(
                lags=lags,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                output_chunk_length=1,
                use_static_covariates=True,
            )
    # ---------------------------------------------------------------------
    # Tree-based Regression Models (LightGBM, RandomForestModel)
    # Use lags + covariate lags so they can exploit past & future covariates
    # and static covariates embedded in the target series.
    # ---------------------------------------------------------------------

    if model_type == ModelType.LIGHTGBM:
        if xl_design:
            # XL: more history + more covariate lags
            return LightGBMModel(
                lags=lags,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                output_chunk_length=1,
                use_static_covariates=True,
                random_state=42,
            )
        else:
            # Medium config
            return LightGBMModel(
                lags=lags,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                output_chunk_length=1,
                use_static_covariates=True,
                random_state=42,
            )

    if model_type == ModelType.RANDOM_FOREST:
        if xl_design:
            # XL: more history + larger forest
            return RandomForestModel(
                lags=lags,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                n_estimators=500,
                max_depth=None,
                n_jobs=-1,
                use_static_covariates=True,
                random_state=42,
            )
        else:
            # Medium config
            return RandomForestModel(
                lags=lags,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                n_estimators=200,
                max_depth=None,
                n_jobs=-1,
                use_static_covariates=True,
                random_state=42,
            )

    if model_type == ModelType.XGBOOST:
        if xl_design:
            # XL: more history + slightly larger booster
            return XGBModel(
                lags=lags,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                output_chunk_length=1,
                use_static_covariates=True,
                random_state=42,
                # objective="reg:squarederror", max_depth=8, n_estimators=500, learning_rate=0.05
            )
        else:
            # Medium config
            return XGBModel(
                lags=lags,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                output_chunk_length=1,
                use_static_covariates=True,
                random_state=42,
                # objective="reg:squarederror", max_depth=6, n_estimators=300, learning_rate=0.1
            )

    # =====================================================================
    # Deep Learning Models
    # =====================================================================

    # Base configuration shared by all deep learning models
    if xl_design:
        # Slightly longer history + optional optimizer tuning for XL nets
        base_kwargs = dict(
            input_chunk_length=60,
            output_chunk_length=1,
            n_epochs=n_epochs,
            batch_size=batch_size,
            random_state=42,
            save_checkpoints=False,
            force_reset=True,
            optimizer_kwargs={"lr": 1e-3, "weight_decay": 1e-4},
            **torch_kwargs,
        )
    else:
        # Medium Config
        base_kwargs = dict(
            input_chunk_length=30,
            output_chunk_length=1,
            n_epochs=n_epochs,
            batch_size=batch_size,
            random_state=42,
            save_checkpoints=False,
            force_reset=True,
            **torch_kwargs,
        )

    # -------------------------
    # N-BEATS
    # -------------------------
    if model_type == ModelType.NBEATS:
        if xl_design:
            # XL: larger and deeper than Medium
            return NBEATSModel(
                generic_architecture=True,
                num_stacks=16,
                num_blocks=2,
                num_layers=6,
                layer_widths=1024,
                dropout=dropout,
                **base_kwargs,
            )
        else:
            # Medium design
            return NBEATSModel(
                generic_architecture=True,
                num_stacks=10,
                num_blocks=1,
                num_layers=4,
                layer_widths=512,
                **base_kwargs,
            )

    # -------------------------
    # TFT
    # -------------------------
    if model_type == ModelType.TFT:
        if xl_design:
            # XL: larger and deeper than Medium
            return TFTModel(
                hidden_size=128,
                lstm_layers=4,
                dropout=dropout,
                num_attention_heads=12,
                add_relative_index=True,
                **base_kwargs,
            )
        else:
            # Medium design
            return TFTModel(
                hidden_size=64,
                lstm_layers=2,
                dropout=dropout,
                num_attention_heads=10,
                add_relative_index=True,
                **base_kwargs,
            )

    # -------------------------
    # TSMIXER
    # -------------------------
    if model_type == ModelType.TSMIXER:
        if xl_design:
            # XL: larger and deeper than Medium
            return TSMixerModel(
                hidden_size=128,
                ff_size=256,
                activation="LeakyReLU",
                n_blocks=16,
                dropout=dropout,
                **base_kwargs,
            )
        else:
            # Medium design
            return TSMixerModel(
                hidden_size=64,
                ff_size=128,
                activation="LeakyReLU",
                n_blocks=8,
                dropout=dropout,
                **base_kwargs,
            )

    # -------------------------
    # BlockRNN (LSTM)
    # -------------------------
    if model_type == ModelType.BLOCK_RNN:
        if xl_design:
            # XL: larger and deeper than Medium
            return BlockRNNModel(
                model="GRU",
                hidden_dim=128,
                n_rnn_layers=3,
                dropout=dropout,
                **base_kwargs,
            )
        else:
            # Medium design
            return BlockRNNModel(
                model="GRU",
                hidden_dim=64,
                n_rnn_layers=2,
                dropout=dropout,
                **base_kwargs,
            )

    # -------------------------
    # TCN
    # -------------------------
    if model_type == ModelType.TCN:
        if xl_design:
            # XL: larger and deeper than Medium
            return TCNModel(
                kernel_size=2,  # used to be 3
                num_filters=128,
                dilation_base=7,  # used to be 2
                weight_norm=True,
                dropout=dropout,
                **base_kwargs,
            )
        else:
            # Medium design
            return TCNModel(
                kernel_size=2,  # used to be 3
                num_filters=64,
                dilation_base=7,  # used to be 2
                weight_norm=True,
                dropout=dropout,
                **base_kwargs,
            )

    # -------------------------
    # TiDE
    # -------------------------
    if model_type == ModelType.TIDE:
        if xl_design:
            # XL: larger and deeper than Medium
            return TiDEModel(
                hidden_size=128,
                dropout=dropout,
                use_layer_norm=True,
                **base_kwargs,
            )
        else:
            # Medium design
            return TiDEModel(
                hidden_size=64,
                dropout=dropout,
                use_layer_norm=True,
                **base_kwargs,
            )

    raise ValueError(f"Unsupported model type: {model_type}")


def prepare_store_item_series(
    df: pd.DataFrame,
    store: int,
    item: int,
    store_medians_fn: Path | None = None,
    item_medians_fn: Path | None = None,
    store_assign_fn: Path | None = None,
    item_assign_fn: Path | None = None,
) -> pd.DataFrame:
    """
    Memory-safe preparation of the (store, item) time series.
    Loads only the cluster-median rows needed for this specific (store, item).
    """

    # ----------------------------------------------------------------------
    # Extract base series (already filtered)
    # ----------------------------------------------------------------------
    mask = (df["store"] == store) & (df["item"] == item)
    series_df = df[mask].copy()

    if series_df.empty:
        logger.warning(f"No data for store {store}, item {item}")
        return pd.DataFrame()

    series_df = series_df.sort_values("date")
    if len(series_df) < 30:
        logger.warning(
            f"Insufficient data for store {store}, item {item}. "
            f"Found {len(series_df)} rows, need at least 30."
        )
        return pd.DataFrame()

    dates = series_df["date"].unique()

    # ----------------------------------------------------------------------
    # Load cluster assignments only for the current store and item
    # ----------------------------------------------------------------------
    if store_assign_fn is not None:
        store_assign = read_csv_or_parquet(store_assign_fn)
        store_clusters = (
            store_assign.loc[store_assign["store"] == store, "cluster_id"]
            .drop_duplicates()
            .tolist()
        )

    if item_assign_fn is not None:
        item_assign = read_csv_or_parquet(item_assign_fn)
        item_clusters = (
            item_assign.loc[item_assign["item"] == item, "cluster_id"]
            .drop_duplicates()
            .tolist()
        )

    # ----------------------------------------------------------------------
    # Load only the necessary cluster median rows
    #  This avoids loading giant full-sized tables.
    # ----------------------------------------------------------------------
    if store_medians_fn is not None:
        store_medians = pd.read_parquet(
            store_medians_fn,
            filters=[
                ("store_cluster_id", "in", store_clusters),
                ("date", "in", dates.tolist()),
            ],
            columns=["date", "store_cluster_id", "store_median"],
        )

    if item_medians_fn is not None:
        item_medians = pd.read_parquet(
            item_medians_fn,
            filters=[
                ("item_cluster_id", "in", item_clusters),
                ("date", "in", dates.tolist()),
            ],
            columns=["date", "item_cluster_id", "item_median"],
        )

    # ----------------------------------------------------------------------
    # Pivot cluster medians wide (1 row per date)
    # ----------------------------------------------------------------------
    if store_medians_fn is not None and not store_medians.empty:
        store_medians_wide = (
            store_medians.pivot(
                index="date",
                columns="store_cluster_id",
                values="store_median",
            )
            .add_prefix("store_cluster_median_")
            .reset_index()
        )
        series_df = series_df.merge(store_medians_wide, on="date", how="left")

    if item_medians_fn is not None and not item_medians.empty:
        item_medians_wide = (
            item_medians.pivot(
                index="date",
                columns="item_cluster_id",
                values="item_median",
            )
            .add_prefix("item_cluster_median_")
            .reset_index()
        )
        series_df = series_df.merge(item_medians_wide, on="date", how="left")

    # ----------------------------------------------------------------------
    # Select covariates
    # ----------------------------------------------------------------------

    # Determine which requested covariate columns actually exist in the data
    available_future = [c for c in FUTURE_COV_COLS if c in series_df.columns]

    # dynamically detect cluster future covariates
    cluster_cols = [
        c
        for c in series_df.columns
        if c.startswith("store_cluster_median_")
        or c.startswith("item_cluster_median_")
    ]

    available_past = [
        c for c in PAST_COV_COLS if c in series_df.columns
    ] + cluster_cols

    # Keep static covariates if present
    static_cols_present = [
        c for c in STATIC_COV_COLS if c in series_df.columns
    ]

    cols_to_keep = (
        ["date", TARGET_COL]
        + static_cols_present
        + available_future
        + available_past
    )

    ts_df = series_df[cols_to_keep].copy()

    # Ensure numeric types
    for col in ts_df.columns:
        if col != "date":
            ts_df[col] = pd.to_numeric(ts_df[col], errors="coerce")

    ts_df = ts_df.set_index("date")
    ts_df = ts_df.replace([np.inf, -np.inf], np.nan)

    # Handle NaNs:
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

    Future covariates are now determined dynamically:
    - Calendar covariates listed in FUTURE_COV_COLS (static)
    - Cluster medians (store_cluster_median_*, item_cluster_median_*)
    - PARAFAC latent factors (parafac_*), or any other *_covariate_* prefix.

    Additionally, store, item, cluster_id, and class_id are used as static covariates.
    """

    try:
        # ------------------------------------------------------------------
        # Basic validation of target availability
        # ------------------------------------------------------------------
        total_len = len(ts_df)
        train_len_approx = int(total_len * split_point)

        train_df_subset = ts_df.iloc[:train_len_approx]
        non_missing_target = train_df_subset[TARGET_COL].count()

        if non_missing_target < min_train_data_points:
            logger.warning(
                f"S{store}/I{item}: Insufficient training data "
                f"({non_missing_target} < {min_train_data_points}). Skipping."
            )
            return None

        train_std = train_df_subset[TARGET_COL].std()
        if train_std == 0 or np.isnan(train_std):
            logger.warning(
                f"S{store}/I{item}: Target has zero or NaN variance. Skipping."
            )
            return None

        # ------------------------------------------------------------------
        # Extract static covariates (one vector per (store, item))
        # ------------------------------------------------------------------
        static_cov_df = None
        static_cols_present = [
            c for c in STATIC_COV_COLS if c in ts_df.columns
        ]

        if static_cols_present:
            # Take first row; should be constant across time for the series
            static_values = {
                col: ts_df[col].iloc[0] for col in static_cols_present
            }
            # 1 x n_static DataFrame (index arbitrary)
            static_cov_df = pd.DataFrame([static_values])

        # ------------------------------------------------------------------
        # Build full TimeSeries from DataFrame (excluding static cols)
        # ------------------------------------------------------------------
        # Drop static columns so they are not treated as dynamic covariates
        ts_df_dynamic = ts_df.drop(
            columns=static_cols_present, errors="ignore"
        )

        # Ensure unique date index (required by Darts TimeSeries)
        # ts_df should already have date as index from prepare_store_item_series
        if ts_df_dynamic.index.has_duplicates:
            logger.warning(
                f"S{store}/I{item}: Duplicate dates detected ({ts_df_dynamic.index.name}). "
                "Keeping the first occurrence per date."
            )
            ts_df_dynamic = ts_df_dynamic[
                ~ts_df_dynamic.index.duplicated(keep="first")
            ]
            # Alternative: aggregate with mean instead of keeping first:
            # ts_df_dynamic = ts_df_dynamic.groupby(ts_df_dynamic.index).mean()

        full_ts = TimeSeries.from_dataframe(
            ts_df_dynamic, fill_missing_dates=True, freq="D", fillna_value=0
        )

        # Target
        target_ts = full_ts[TARGET_COL]

        # Attach static covariates to the target TimeSeries
        if static_cov_df is not None:
            logger.info(
                f"Attaching static covariates: {static_cov_df.columns}"
            )
            target_ts = target_ts.with_static_covariates(static_cov_df)

        # ------------------------------------------------------------------
        # Past covariates: rolling/ewm + cluster medians (observed)
        # ------------------------------------------------------------------
        cluster_cols = [
            c
            for c in ts_df_dynamic.columns
            if c.startswith("store_cluster_median_")
            or c.startswith("item_cluster_median_")
        ]

        valid_p_cols = [
            c for c in PAST_COV_COLS if c in ts_df_dynamic.columns
        ] + cluster_cols
        past_covs_ts = full_ts[valid_p_cols] if valid_p_cols else None

        # ------------------------------------------------------------------
        # Future covariates: only things truly known in advance (calendar, etc.)
        # ------------------------------------------------------------------
        valid_f_cols = [
            c for c in FUTURE_COV_COLS if c in ts_df_dynamic.columns
        ]
        future_covs_ts = full_ts[valid_f_cols] if valid_f_cols else None

        # ------------------------------------------------------------------
        # Train–Validation Split (static covs are carried automatically)
        # ------------------------------------------------------------------
        train_target, val_target = target_ts.split_before(split_point)

        if len(train_target) == 0 or len(val_target) == 0:
            logger.warning(
                f"S{store}/I{item}: Empty train or validation split."
            )
            return None

        # Past covariates
        if past_covs_ts is not None:
            train_past, val_past = past_covs_ts.split_before(split_point)
        else:
            train_past, val_past = None, None

        # Future covariates
        if future_covs_ts is not None:
            train_future, val_future = future_covs_ts.split_before(split_point)
        else:
            train_future, val_future = None, None

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        logger.info(
            f"S{store}/I{item}: Data prepared. "
            f"Train={len(train_target)}, Val={len(val_target)}, "
            f"Past Dim={past_covs_ts.n_components if past_covs_ts else 0}, "
            f"Future Dim={future_covs_ts.n_components if future_covs_ts else 0}, "
            f"Static Dim={static_cov_df.shape[1] if static_cov_df is not None else 0}"
        )

        # ------------------------------------------------------------------
        # Return time series dictionary
        # ------------------------------------------------------------------
        return {
            "train_target": train_target,
            "val_target": val_target,
            "train_past": train_past,
            "val_past": val_past,
            "train_future": train_future,
            "val_future": val_future,
            "full_past": past_covs_ts,
            "full_future": future_covs_ts,
            "static_covariates": static_cov_df,
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
    past_covs: bool = False,
    future_covs: bool = False,
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
        "RANDOM_FOREST",
        "LINEAR_REGRESSION",
        "XGBOOST",
        "LIGHTGBM",
    }
    supports_future = modelType in {
        "TFT",
        "TSMIXER",
        "TIDE",
        "RANDOM_FOREST",
        "LIGHTGBM",
        "LINEAR_REGRESSION",
        "XGBOOST",
    }

    # Which models support validation series during training (deep learning models)
    supports_val_series = modelType in {
        "NBEATS",
        "TFT",
        "TSMIXER",
        "BLOCK_RNN",
        "TCN",
        "TIDE",
        "LIGHTGBM",
    }

    # ------------------------------------------------------------------
    # Helper to append a NaN row in a consistent way
    # ------------------------------------------------------------------
    def _append_nan_row(reason: str) -> pd.DataFrame:
        logger.warning(f"Skipping {modelType} for S{store}/I{item}: {reason}")
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
        return pd.concat([metrics_df, new_row], ignore_index=True)

    try:
        train_target = data_dict["train_target"]
        val_target = data_dict["val_target"]
        forecast_horizon = len(val_target)

        # ------------------------------------------------------------------
        # Guard against too-short training series for models like NBEATS
        # ------------------------------------------------------------------
        # For most Darts DL models, the minimum length is roughly:
        #   input_chunk_length + output_chunk_length
        # We infer it from attributes if they exist; otherwise default to 1.
        input_chunk_length = getattr(model, "input_chunk_length", None)
        output_chunk_length = getattr(model, "output_chunk_length", 1)

        if input_chunk_length is not None:
            # Use a conservative minimum required length
            min_required_length = int(input_chunk_length) + int(
                max(output_chunk_length, 1)
            )
        else:
            # Fallback if the model does not expose these attributes
            min_required_length = 1

        if len(train_target) < min_required_length:
            reason = (
                f"training series too short (len={len(train_target)}) "
                f"< required={min_required_length}"
            )
            return _append_nan_row(reason)

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
            and past_covs
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
            and future_covs
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
        if supports_past and train_past_scaled is not None and past_covs:
            fit_kwargs["past_covariates"] = train_past_scaled
            if supports_val_series and val_past_scaled is not None:
                fit_kwargs["val_past_covariates"] = val_past_scaled

        # Add future covariates if supported
        if (
            supports_future
            and future_covs_scaled_full is not None
            and future_covs
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
        if supports_past and past_covs_scaled_full is not None and past_covs:
            predict_kwargs["past_covariates"] = past_covs_scaled_full

        if (
            supports_future
            and future_covs_scaled_full is not None
            and future_covs
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
