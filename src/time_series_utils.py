import sys
from pathlib import Path
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Sequence
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.metrics import rmse, mae, ope, smape, marre
from sklearn.preprocessing import RobustScaler
from darts.dataprocessing.transformers import Scaler
import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection
import traceback
from darts.utils.callbacks import TFMProgressBar
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

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
# Note: Raw 'dayofweek', 'weekofmonth', 'monthofyear' excluded to avoid
# multicollinearity with sin/cos versions.
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

from enum import Enum


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

    # ------------------------------------------------------------------
    # SUPPORT FLAGS
    # ------------------------------------------------------------------
    @property
    def supports_past(self) -> bool:
        return self in {
            ModelType.NBEATS,
            ModelType.TFT,
            ModelType.TSMIXER,
            ModelType.BLOCK_RNN,
            ModelType.TCN,
            ModelType.TIDE,
            ModelType.RANDOM_FOREST,
            ModelType.LINEAR_REGRESSION,
            ModelType.XGBOOST,
            ModelType.LIGHTGBM,
        }

    @property
    def supports_future(self) -> bool:
        return self in {
            ModelType.TFT,
            ModelType.TSMIXER,
            ModelType.TIDE,
            ModelType.RANDOM_FOREST,
            ModelType.LIGHTGBM,
            ModelType.LINEAR_REGRESSION,
            ModelType.XGBOOST,
        }

    @property
    def supports_val(self) -> bool:
        # Some classical ML models like LightGBM support val_series in Darts
        return self in {
            ModelType.NBEATS,
            ModelType.TFT,
            ModelType.TSMIXER,
            ModelType.BLOCK_RNN,
            ModelType.TCN,
            ModelType.TIDE,
            ModelType.LIGHTGBM,
        }

    @property
    def supports_global(self) -> bool:
        return self in {
            ModelType.RANDOM_FOREST,
            ModelType.LINEAR_REGRESSION,
            ModelType.XGBOOST,
            ModelType.NBEATS,
            ModelType.TFT,
            ModelType.TSMIXER,
            ModelType.BLOCK_RNN,
            ModelType.TCN,
            ModelType.TIDE,
        }

    @property
    def supports_local(self) -> bool:
        return self in {
            ModelType.EXPONENTIAL_SMOOTHING,
            ModelType.AUTO_ARIMA,
            ModelType.THETA,
            ModelType.KALMAN,
        }


# Helper: which models are deep-learning?
DL_MODELS = {
    ModelType.NBEATS,
    ModelType.TFT,
    ModelType.TSMIXER,
    ModelType.BLOCK_RNN,
    ModelType.TCN,
    ModelType.TIDE,
}


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
                num_blocks=16,
                dropout=dropout,
                **base_kwargs,
            )
        else:
            # Medium design
            return TSMixerModel(
                hidden_size=64,
                ff_size=128,
                activation="LeakyReLU",
                num_blocks=8,
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
                num_filters=16,
                dilation_base=2,  # used to be 2
                num_layers=2,
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


def generate_torch_kwargs(
    gpu_id: Optional[int],
    working_dir: Path,
    train_series: TimeSeries,  # Darts TimeSeries (train portion)
    patience: int = 8,
) -> Dict:
    """
    Return pl_trainer_kwargs + torch_metrics dict to pass into create_model()
    with early stopping on val_rmsse.
    """

    # Get numpy training values from Darts series
    train_vals = train_series.univariate_values()  # shape [N], np.ndarray

    # Compute RMSSE scale (rmse_naive) using your logic
    rmse_naive = compute_rmsse_scale_from_train(train_vals)

    # Early stopping on val_rmsse
    early_stopper = EarlyStopping(
        monitor="val_rmsse",
        min_delta=0.0001,
        patience=patience,
        verbose=True,
        mode="min",
    )

    # Device selection
    if gpu_id is not None and torch.cuda.is_available():
        accelerator = "gpu"
        devices = [gpu_id]
        logger.debug(f"Using GPU {gpu_id}")
    else:
        accelerator = "auto"
        devices = "auto"

    # Torch metrics: RMSSE (you can add others if you want)
    metrics = MetricCollection(
        {
            "rmsse": RMSSEMetric(rmse_naive),
        }
    )

    torch_kwargs = {
        "pl_trainer_kwargs": {
            "accelerator": accelerator,
            "devices": devices,
            "callbacks": [
                early_stopper,
                TFMProgressBar(enable_train_bar_only=True),
            ],
            "default_root_dir": str(working_dir),
            "logger": TensorBoardLogger(
                save_dir=str(working_dir),
                name="tcn_rmsse",  # or dynamic per model
            ),
        },
        "torch_metrics": metrics,
    }

    return torch_kwargs


def generate_torch_kwargs_global(
    gpu_id: Optional[int],
    working_dir: Path,
    train_series: Sequence[
        TimeSeries
    ],  # list of Darts TimeSeries (train portions)
    patience: int = 8,
    run_name: str = "global_model_rmsse",
) -> Dict[str, Any]:
    """
    Return pl_trainer_kwargs + torch_metrics dict to pass into create_model()
    for a *global* model, with early stopping on val_rmsse.

    The RMSSE scale (rmse_naive) is computed across all training series.
    """

    if not train_series:
        raise ValueError(
            "train_series is empty; cannot compute global RMSSE scale."
        )

    # ----------------------------------------------------------------------
    # Compute global RMSSE denominator over all training series
    # ----------------------------------------------------------------------
    rmse_naive = compute_global_rmsse_scale_from_train_list(train_series)
    logger.info(f"Global rmse_naive for RMSSE: {rmse_naive:.6f}")

    # ----------------------------------------------------------------------
    # Early stopping on validation RMSSE
    # ----------------------------------------------------------------------
    early_stopper = EarlyStopping(
        monitor="val_rmsse",
        min_delta=0.0001,
        patience=patience,
        verbose=True,
        mode="min",
    )

    # ----------------------------------------------------------------------
    # Device selection
    # ----------------------------------------------------------------------
    if gpu_id is not None and torch.cuda.is_available():
        accelerator = "gpu"
        devices = [gpu_id]
        logger.debug(f"Using GPU {gpu_id}")
    else:
        accelerator = "auto"
        devices = "auto"
        logger.debug("Using accelerator=auto, devices=auto")

    # ----------------------------------------------------------------------
    # Torch metrics: RMSSE (global denominator)
    # ----------------------------------------------------------------------
    metrics = MetricCollection(
        {
            "rmsse": RMSSEMetric(rmse_naive),
        }
    )

    torch_kwargs: Dict[str, Any] = {
        "pl_trainer_kwargs": {
            "accelerator": accelerator,
            "devices": devices,
            "callbacks": [
                early_stopper,
                TFMProgressBar(enable_train_bar_only=True),
            ],
            "default_root_dir": str(working_dir),
            "logger": TensorBoardLogger(
                save_dir=str(working_dir),
                name=run_name,  # e.g. "tcn_global_rmsse"
            ),
        },
        "torch_metrics": metrics,
    }

    return torch_kwargs


def build_global_train_val_lists(
    df: pd.DataFrame,
    split_point: float = 0.8,
    min_train_data_points: int = 60,
    store_medians_fn: Path | None = None,
    item_medians_fn: Path | None = None,
    store_assign_fn: Path | None = None,
    item_assign_fn: Path | None = None,
    past_covs: bool = False,
    future_covs: bool = False,
) -> Tuple[
    List[TimeSeries],
    List[TimeSeries],
    List[TimeSeries] | None,
    List[TimeSeries] | None,
    List[TimeSeries] | None,
    List[TimeSeries] | None,
    List[Dict[str, Any]],
]:
    """
    Build lists of train/val target series and covariates for a GLOBAL model.
    Returns:
        train_targets, val_targets,
        train_pasts, val_pasts,
        train_futures, val_futures,
        meta_list (each meta has store, item, and data_dict)
    """
    train_targets: List[TimeSeries] = []
    val_targets: List[TimeSeries] = []
    train_pasts: List[TimeSeries] = []
    val_pasts: List[TimeSeries] = []
    train_futures: List[TimeSeries] = []
    val_futures: List[TimeSeries] = []
    meta_list: List[Dict[str, Any]] = []

    # Loop over all (store, item) combos
    for (store, item), _ in df.groupby(["store", "item"]):
        ts_df = prepare_store_item_series(
            df=df,
            store=store,
            item=item,
            store_medians_fn=store_medians_fn,
            item_medians_fn=item_medians_fn,
            store_assign_fn=store_assign_fn,
            item_assign_fn=item_assign_fn,
        )

        if ts_df.empty:
            continue

        data_dict = get_train_val_data_with_covariates(
            ts_df=ts_df,
            store=store,
            item=item,
            split_point=split_point,
            min_train_data_points=min_train_data_points,
            exclude_cluster_covs=True,
        )

        if data_dict is None:
            continue

        train_target = data_dict["train_target"]
        val_target = data_dict["val_target"]

        train_targets.append(train_target)
        val_targets.append(val_target)

        # Past covariates
        if past_covs and data_dict.get("train_past") is not None:
            train_pasts.append(data_dict["train_past"])
            val_pasts.append(data_dict["val_past"])
        else:
            # keep shapes aligned (use None later)
            pass

        # Future covariates
        if future_covs and data_dict.get("train_future") is not None:
            train_futures.append(data_dict["train_future"])
            val_futures.append(data_dict["val_future"])
        else:
            pass

        meta_list.append(
            {
                "store": store,
                "item": item,
                "data_dict": data_dict,
            }
        )

    # If no series found
    if len(train_targets) == 0:
        raise ValueError("No valid (store, item) series for global model.")

    # Normalize covariate lists: if none used, set to None
    if not past_covs or len(train_pasts) == 0:
        train_pasts = None
        val_pasts = None
    if not future_covs or len(train_futures) == 0:
        train_futures = None
        val_futures = None

    return (
        train_targets,
        val_targets,
        train_pasts,
        val_pasts,
        train_futures,
        val_futures,
        meta_list,
    )


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
    exclude_cluster_covs: bool = False,
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
        cluster_cols = []
        if not exclude_cluster_covs:
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


def compute_rmsse_scale_from_train(
    train_vals: np.ndarray,
    epsilon: float = np.finfo(float).eps,
) -> float:
    """
    Compute the RMSSE scale (denominator) from training values only:
    rmse_naive = sqrt(mean((train[t] - train[t-1])^2))
    """
    train_vals = np.asarray(train_vals).flatten()

    if train_vals is None or len(train_vals) < 2:
        return np.nan

    if np.any(np.isnan(train_vals)):
        return np.nan

    naive_train_sq_errors = np.square(train_vals[1:] - train_vals[:-1])
    rmse_naive = np.sqrt(np.mean(naive_train_sq_errors))

    if rmse_naive == 0:
        # Avoid division by zero downstream; use epsilon so RMSSE ≈ rmse_forecast/epsilon
        rmse_naive = epsilon

    return float(rmse_naive)


def compute_global_rmsse_scale_from_train_list(
    train_series: Sequence[TimeSeries],
    epsilon: float = np.finfo(float).eps,
) -> float:
    """
    Compute a single RMSSE scale (rmse_naive) over a list of training TimeSeries,
    by pooling all 1-step naive squared errors across series.

    rmse_naive = sqrt(mean( (y_t - y_{t-1})^2 over all series ))
    """
    all_sq_errors = []

    for ts in train_series:
        vals = ts.univariate_values()  # shape [N]
        vals = np.asarray(vals).flatten()

        if vals is None or len(vals) < 2:
            continue
        if np.any(np.isnan(vals)):
            continue

        diffs = vals[1:] - vals[:-1]
        all_sq_errors.append(diffs**2)

    if not all_sq_errors:
        # Fallback: no usable series; avoid crash downstream
        return float(epsilon)

    all_sq_errors = np.concatenate(all_sq_errors)
    rmse_naive = float(np.sqrt(np.mean(all_sq_errors)))

    if rmse_naive == 0:
        rmse_naive = float(epsilon)

    return rmse_naive


class RMSSEMetric(Metric):
    """
    RMSSE as a torchmetrics.Metric using a fixed scale (rmse_naive)
    computed from the training data.

    rmsse = sqrt(mean((y - y_hat)^2)) / rmse_naive
    """

    full_state_update: bool = False

    def __init__(self, rmse_naive: float, epsilon: float = 1e-12):
        super().__init__()
        # This is your denominator
        self.rmse_naive = (
            float(rmse_naive) if rmse_naive is not None else float("nan")
        )
        self.epsilon = float(epsilon)

        self.add_state(
            "squared_error",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "n_obs",
            default=torch.tensor(0),
            dist_reduce_fx="sum",
        )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        preds, target: typically [batch, output_chunk_length, channels]
        We just flatten them and accumulate squared errors.
        """
        if preds.numel() == 0 or target.numel() == 0:
            return

        # Ensure same shape
        # (Darts should already ensure this, but be defensive)
        min_len = min(preds.numel(), target.numel())
        preds_flat = preds.reshape(-1)[:min_len]
        target_flat = target.reshape(-1)[:min_len]

        se = (target_flat - preds_flat) ** 2
        self.squared_error += se.sum()
        self.n_obs += torch.tensor(min_len, device=target.device)

    def compute(self) -> Tensor:
        if self.n_obs == 0:
            return torch.tensor(float("nan"))

        mse = self.squared_error / self.n_obs
        rmse_forecast = torch.sqrt(mse)

        denom = self.rmse_naive if self.rmse_naive > 0 else self.epsilon
        return rmse_forecast / denom


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
    modelType: ModelType,
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

    # ------------------------------------------------------------------
    # Helper to append a NaN row in a consistent way
    # ------------------------------------------------------------------
    def _append_nan_row(reason: str) -> pd.DataFrame:
        logger.warning(f"Skipping {modelType} for S{store}/I{item}: {reason}")
        nan_row_dict = {
            "Model": modelType.value,
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

        logger.debug(
            f"input_chunk_length: {input_chunk_length}, output_chunk_length: {output_chunk_length}"
        )

        # --- TARGET SCALER ---
        target_scaler = Scaler(RobustScaler())
        train_target_scaled = target_scaler.fit_transform(train_target)
        val_target_scaled = target_scaler.transform(val_target)

        # --- PAST COVARIATES SCALER ---
        train_past_scaled = None
        val_past_scaled = None
        past_covs_scaled_full = None

        if (
            modelType.supports_past
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
            modelType.supports_future
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
        if modelType.supports_val:
            fit_kwargs["val_series"] = val_target_scaled

        # Add past covariates if supported
        if (
            modelType.supports_past
            and train_past_scaled is not None
            and past_covs
        ):
            fit_kwargs["past_covariates"] = train_past_scaled
            if modelType.supports_val and val_past_scaled is not None:
                fit_kwargs["val_past_covariates"] = val_past_scaled

        # Add future covariates if supported
        if (
            modelType.supports_future
            and future_covs_scaled_full is not None
            and future_covs
        ):
            fit_kwargs["future_covariates"] = future_covs_scaled_full
            if modelType.supports_val and val_future_scaled is not None:
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
            modelType.supports_past
            and past_covs_scaled_full is not None
            and past_covs
        ):
            predict_kwargs["past_covariates"] = past_covs_scaled_full

        if (
            modelType.supports_future
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


def eval_global_model_with_covariates(
    model_type: ModelType,
    model: ForecastingModel,
    series_meta: List[Dict[str, Any]],
    metrics_df: pd.DataFrame,
    past_covs: bool = False,
    future_covs: bool = False,
) -> pd.DataFrame:
    """
    Global version of eval_model_with_covariates.

    - series_meta: list of dicts with keys "store", "item", "data_dict"
      where data_dict contains:
        "train_target", "val_target", "train_past", "val_past",
        "train_future", "val_future", "full_past", "full_future"

    - Fits the model ONCE on all training series.
    - Then predicts and computes metrics for each (store, item).

    Returns:
        Updated metrics_df with one row per (store, item).
    """

    # ----------------------------------------------------------------------
    # Helper to append a NaN row in a consistent way
    # ----------------------------------------------------------------------
    def _append_nan_row(store: int, item: int, reason: str) -> None:
        nonlocal metrics_df
        logger.warning(f"Skipping {model_type} for S{store}/I{item}: {reason}")
        nan_row_dict = {
            "Model": model_type.value,
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

    # ----------------------------------------------------------------------
    # 1) Preprocess all series: scaling + checks
    # ----------------------------------------------------------------------
    train_targets_scaled: List[TimeSeries] = []
    val_targets_scaled: List[TimeSeries] = []
    train_pasts_scaled: List[TimeSeries] = []
    val_pasts_scaled: List[TimeSeries] = []
    full_pasts_scaled: List[TimeSeries] = []

    train_futures_scaled: List[TimeSeries] = []
    val_futures_scaled: List[TimeSeries] = []
    full_futures_scaled: List[TimeSeries] = []

    # Metadata per VALID series (only those included in the global fit)
    valid_series_meta: List[Dict[str, Any]] = []

    # For global min-length checks, read model attributes once
    input_chunk_length = getattr(model, "input_chunk_length", None)
    output_chunk_length = getattr(model, "output_chunk_length", 1)
    if input_chunk_length is not None:
        min_required_length = int(input_chunk_length) + int(
            max(output_chunk_length, 1)
        )
    else:
        min_required_length = 1

    for meta in series_meta:
        try:
            store = meta["store"]
            item = meta["item"]
            data_dict = meta["data_dict"]
            train_target: TimeSeries = data_dict["train_target"]
            val_target: TimeSeries = data_dict["val_target"]
            forecast_horizon = len(val_target)

            if forecast_horizon == 0:
                _append_nan_row(store, item, "empty validation series")
                continue

            # --------------------------------------------------------------
            # Length / variance guards (same as local version)
            # --------------------------------------------------------------
            if len(train_target) < min_required_length:
                reason = (
                    f"training series too short (len={len(train_target)}) "
                    f"< required={min_required_length}"
                )
                _append_nan_row(store, item, reason)
                continue

            train_vals = train_target.univariate_values()
            if np.std(train_vals) == 0 or np.isnan(np.std(train_vals)):
                _append_nan_row(
                    store,
                    item,
                    "target has zero or NaN variance",
                )
                continue

            # --------------------------------------------------------------
            # TARGET SCALER (per series)
            # --------------------------------------------------------------
            target_scaler = Scaler(RobustScaler())
            train_target_scaled = target_scaler.fit_transform(train_target)
            val_target_scaled = target_scaler.transform(val_target)

            # --------------------------------------------------------------
            # PAST COVARIATES SCALER (per series)
            # --------------------------------------------------------------
            train_past_scaled = None
            val_past_scaled = None
            full_past_scaled = None

            if (
                model_type.supports_past
                and data_dict.get("train_past") is not None
                and past_covs
            ):
                past_scaler = Scaler(RobustScaler())
                train_past_scaled = past_scaler.fit_transform(
                    data_dict["train_past"]
                )

                if data_dict.get("val_past") is not None:
                    val_past_scaled = past_scaler.transform(
                        data_dict["val_past"]
                    )

                if data_dict.get("full_past") is not None:
                    full_past_scaled = past_scaler.transform(
                        data_dict["full_past"]
                    )

            # --------------------------------------------------------------
            # FUTURE COVARIATES SCALER (per series)
            # --------------------------------------------------------------
            train_future_scaled = None
            val_future_scaled = None
            full_future_scaled = None

            if (
                model_type.supports_future
                and data_dict.get("full_future") is not None
                and data_dict.get("train_future") is not None
                and future_covs
            ):
                future_scaler = Scaler(RobustScaler())
                train_future_scaled = future_scaler.fit_transform(
                    data_dict["train_future"]
                )

                if data_dict.get("val_future") is not None:
                    val_future_scaled = future_scaler.transform(
                        data_dict["val_future"]
                    )

                full_future_scaled = future_scaler.transform(
                    data_dict["full_future"]
                )

            # --------------------------------------------------------------
            # Collect scaled series for global fit
            # --------------------------------------------------------------
            train_targets_scaled.append(train_target_scaled)
            val_targets_scaled.append(val_target_scaled)

            if train_past_scaled is not None:
                train_pasts_scaled.append(train_past_scaled)
                val_pasts_scaled.append(val_past_scaled)
                full_pasts_scaled.append(full_past_scaled)
            else:
                # keep alignment by storing None placeholders if needed
                train_pasts_scaled.append(None)
                val_pasts_scaled.append(None)
                full_pasts_scaled.append(None)

            if train_future_scaled is not None:
                train_futures_scaled.append(train_future_scaled)
                val_futures_scaled.append(val_future_scaled)
                full_futures_scaled.append(full_future_scaled)
            else:
                train_futures_scaled.append(None)
                val_futures_scaled.append(None)
                full_futures_scaled.append(None)

            valid_series_meta.append(
                {
                    "store": store,
                    "item": item,
                    "train_target": train_target,
                    "val_target": val_target,
                    "forecast_horizon": forecast_horizon,
                    "target_scaler": target_scaler,
                    "train_target_scaled": train_target_scaled,
                    "val_target_scaled": val_target_scaled,
                    "train_past_scaled": train_past_scaled,
                    "val_past_scaled": val_past_scaled,
                    "full_past_scaled": full_past_scaled,
                    "train_future_scaled": train_future_scaled,
                    "val_future_scaled": val_future_scaled,
                    "full_future_scaled": full_future_scaled,
                }
            )

        except Exception as e:
            logger.error(
                f"Error preparing series for global {model_type} "
                f"S{store}/I{item}: {e}"
            )
            logger.error(traceback.format_exc())
            _append_nan_row(store, item, "exception during preprocessing")
            continue

    # ----------------------------------------------------------------------
    # 2) Fit global model ONCE on all training series
    # ----------------------------------------------------------------------
    if len(valid_series_meta) == 0:
        logger.warning(
            f"No valid series to fit global model {model_type}. "
            "Returning metrics_df as is."
        )
        return metrics_df

    logger.info(
        f"Fitting GLOBAL {model_type.value} on {len(valid_series_meta)} series..."
    )

    try:
        fit_kwargs: Dict[str, Any] = {
            "series": train_targets_scaled,
        }

        # Validation series for early stopping, if supported
        if model_type.supports_val:
            fit_kwargs["val_series"] = val_targets_scaled

        # Past covariates - pass list if any series actually has them
        # IMPORTANT: Keep None placeholders to maintain alignment with series list
        if (
            model_type.supports_past
            and past_covs
            and any(ts is not None for ts in train_pasts_scaled)
        ):
            fit_kwargs["past_covariates"] = train_pasts_scaled
            if model_type.supports_val:
                fit_kwargs["val_past_covariates"] = val_pasts_scaled

        # Future covariates
        # IMPORTANT: Keep None placeholders to maintain alignment with series list
        if (
            model_type.supports_future
            and future_covs
            and any(ts is not None for ts in train_futures_scaled)
        ):
            fit_kwargs["future_covariates"] = train_futures_scaled
            if model_type.supports_val:
                fit_kwargs["val_future_covariates"] = val_futures_scaled

        logger.debug(f"Global fit kwargs keys: {list(fit_kwargs.keys())}")
        model.fit(**fit_kwargs)

    except Exception as e:
        logger.error(f"Error fitting global {model_type}: {e}")
        logger.error(traceback.format_exc())
        # If fit fails, we already appended NaNs for invalid series; but we
        # now need to append NaNs for all valid_series_meta as well.
        for meta in valid_series_meta:
            _append_nan_row(meta["store"], meta["item"], "global fit failed")
        return metrics_df

    # ----------------------------------------------------------------------
    # 3) Per-series prediction + metrics
    # ----------------------------------------------------------------------
    logger.info(
        f"Predicting and computing metrics for GLOBAL {model_type.value}..."
    )

    for meta in valid_series_meta:
        store = meta["store"]
        item = meta["item"]
        train_target = meta["train_target"]
        val_target = meta["val_target"]
        forecast_horizon = meta["forecast_horizon"]
        target_scaler = meta["target_scaler"]
        train_target_scaled = meta["train_target_scaled"]
        full_past_scaled = meta["full_past_scaled"]
        full_future_scaled = meta["full_future_scaled"]

        try:
            predict_kwargs: Dict[str, Any] = {
                "n": forecast_horizon,
                "series": train_target_scaled,
            }

            # IMPORTANT: for prediction, covariates must extend through horizon
            if (
                model_type.supports_past
                and past_covs
                and full_past_scaled is not None
            ):
                predict_kwargs["past_covariates"] = full_past_scaled

            if (
                model_type.supports_future
                and future_covs
                and full_future_scaled is not None
            ):
                predict_kwargs["future_covariates"] = full_future_scaled

            logger.debug(
                f"Predict kwargs keys for S{store}/I{item}: "
                f"{list(predict_kwargs.keys())}"
            )
            forecast_scaled = model.predict(**predict_kwargs)

            # Inverse-transform the forecast
            forecast = target_scaler.inverse_transform(forecast_scaled)

            # Metrics in original scale
            metrics = calculate_metrics(train_target, val_target, forecast)

            new_row_dict = {
                "Model": model_type.value,
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

            cols_to_downcast = [
                c
                for c in new_row.columns
                if c not in ["Model", "Store", "Item"]
            ]
            for col in cols_to_downcast:
                new_row[col] = pd.to_numeric(new_row[col], downcast="float")

            metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

            logger.info(
                f"FINISHED GLOBAL {model_type.value} S{store}/I{item}. "
                f"SMAPE: {metrics['smape']:.4f}"
            )

        except Exception as e:
            logger.error(
                f"Error predicting/evaluating {model_type.value} "
                f"for S{store}/I{item}: {e}"
            )
            logger.error(traceback.format_exc())
            _append_nan_row(store, item, "prediction/eval failed")

    return metrics_df
