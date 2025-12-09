import json
import yaml
import sys
from pathlib import Path
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Sequence, Callable
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
import optuna


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


def load_model_config(
    config_path: Optional[Path],
) -> Dict[ModelType, Dict[str, Any]]:
    """
    Load per-model hyperparameters from a JSON or YAML file.

    Expected format (keys are model type names, matching ModelType values):

    JSON:
    {
      "NBEATS": {
        "batch_size": 512,
        "n_epochs": 30,
        "lr": 0.001,
        "dropout": 0.2,
        "patience": 10,
        "model_kwargs": {
          "num_stacks": 16,
          "num_blocks": 2,
          "num_layers": 6,
          "layer_widths": 1024
        }
      },
      "TFT": {
        "batch_size": 256,
        "n_epochs": 40,
        "lr": 0.0005,
        "dropout": 0.1
      }
    }

    YAML equivalent is also supported if PyYAML is installed.
    """
    if config_path is None:
        return {}

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Model config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix == ".json":
        with config_path.open("r") as f:
            raw = json.load(f)
    elif suffix in (".yml", ".yaml"):
        if yaml is None:
            raise ImportError(
                "PyYAML is not installed but a YAML config file was provided. "
                "Install with `pip install pyyaml` or use JSON instead."
            )
        with config_path.open("r") as f:
            raw = yaml.safe_load(f)
    else:
        raise ValueError(
            f"Unsupported config file extension '{suffix}'. "
            "Use .json, .yaml or .yml."
        )

    model_config: Dict[ModelType, Dict[str, Any]] = {}
    for key, val in raw.items():
        if not isinstance(val, dict):
            logger.warning(
                "Skipping config for key '%s' because value is not a dict.",
                key,
            )
            continue
        try:
            mtype = ModelType(key.upper())
        except ValueError:
            logger.warning(
                "Unknown model type '%s' in config file; skipping.", key
            )
            continue
        model_config[mtype] = val

    return model_config


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
    n_epochs: int = 100,
    lr: float = 3e-4,
    dropout: float = 0.1,
    model_cfg: Optional[Dict[str, Any]] = None,
    torch_kwargs: Optional[Dict[str, Any]] = None,
    xl_design: bool = False,
    past_covs: bool = False,
    future_covs: bool = False,
) -> ForecastingModel:
    """
    Factory to create a Darts model instance (classical or deep learning).

    Args:
        model_type: ModelType enum value.
        model_cfg: Per-model hyperparameters for THIS model_type, e.g.:

            {
              "lr": 0.001,
              "batch_size": 512,
              "n_epochs": 40,
              "dropout": 0.2,
              "input_chunk_length": 90,
              "model_kwargs": { ... }   # model-specific kwargs
            }

        batch_size, n_epochs, lr, dropout: global defaults used if not
            overridden in model_cfg.
        torch_kwargs: PyTorch/Lightning kwargs for DL models
                      (e.g. optimizer_kwargs, pl_trainer_kwargs, input_chunk_length).
        xl_design, past_covs, future_covs: architecture / covariate flags.

    Returns:
        ForecastingModel instance.
    """
    if torch_kwargs is None:
        torch_kwargs = {}

    cfg = model_cfg or {}
    model_kwargs = cfg.get("model_kwargs", {})  # free-form kwargs per model

    # =====================================================================
    # Classical Models (no batch_size, epochs, or torch_kwargs needed)
    # =====================================================================

    if model_type == ModelType.EXPONENTIAL_SMOOTHING:
        return ExponentialSmoothing(**model_kwargs)

    if model_type == ModelType.AUTO_ARIMA:
        return AutoARIMA(**model_kwargs)

    if model_type == ModelType.THETA:
        # allow overriding season_mode in model_kwargs
        return Theta(season_mode=SeasonalityMode.ADDITIVE, **model_kwargs)

    if model_type == ModelType.KALMAN:
        # keep special handling for dim_x, random_state, but allow overrides
        return KalmanForecaster(
            dim_x=model_kwargs.get("dim_x", 1),
            random_state=model_kwargs.get("random_state", 42),
            **{
                k: v
                for k, v in model_kwargs.items()
                if k not in {"dim_x", "random_state"}
            },
        )

    # =====================================================================
    # Regression / Tree-based models (lags + covariate lags)
    # =====================================================================

    lags, lags_past, lags_future = _lags_for_regression(
        xl_design, past_covs, future_covs
    )

    if model_type == ModelType.LINEAR_REGRESSION:
        base = dict(
            lags=lags,
            lags_past_covariates=lags_past,
            lags_future_covariates=lags_future,
            output_chunk_length=1,
            use_static_covariates=True,
        )
        base.update(model_kwargs)
        return LinearRegressionModel(**base)

    if model_type == ModelType.LIGHTGBM:
        base = dict(
            lags=lags,
            lags_past_covariates=lags_past,
            lags_future_covariates=lags_future,
            output_chunk_length=1,
            use_static_covariates=True,
            random_state=42,
        )
        base.update(model_kwargs)
        return LightGBMModel(**base)

    if model_type == ModelType.RANDOM_FOREST:
        base = dict(
            lags=lags,
            lags_past_covariates=lags_past,
            lags_future_covariates=lags_future,
            n_estimators=500 if xl_design else 200,
            max_depth=None,
            n_jobs=-1,
            use_static_covariates=True,
            random_state=42,
        )
        base.update(model_kwargs)
        return RandomForestModel(**base)

    if model_type == ModelType.XGBOOST:
        base = dict(
            lags=lags,
            lags_past_covariates=lags_past,
            lags_future_covariates=lags_future,
            output_chunk_length=1,
            use_static_covariates=True,
            random_state=42,
        )
        base.update(model_kwargs)
        return XGBModel(**base)

    # =====================================================================
    # Deep Learning Models
    # =====================================================================

    # Global-level overrides from cfg
    batch_size_eff = int(cfg.get("batch_size", batch_size))
    n_epochs_eff = int(cfg.get("n_epochs", n_epochs))
    lr_eff = float(cfg.get("lr", lr))
    dropout_eff = float(cfg.get("dropout", dropout))
    input_chunk_length_eff = int(cfg.get("input_chunk_length", 60))

    logger.info(
        f"Creating model: {model_type.value}, "
        f"past_covs={past_covs}, future_covs={future_covs}, xl_design={xl_design}, "
        f"batch_size={batch_size_eff}, n_epochs={n_epochs_eff}, lr={lr_eff}, "
        f"dropout={dropout_eff}, input_chunk_length={input_chunk_length_eff}"
    )

    base_kwargs = dict(
        input_chunk_length=input_chunk_length_eff,
        output_chunk_length=1,
        n_epochs=n_epochs_eff,
        batch_size=batch_size_eff,
        random_state=42,
        save_checkpoints=False,
        force_reset=True,
        optimizer_kwargs={
            "lr": lr_eff,
            # allow passing extra optimizer kwargs in cfg
            **cfg.get("optimizer_kwargs", {}),
        },
    )

    # torch_kwargs can override defaults in base_kwargs
    base_kwargs.update(torch_kwargs)

    # -------------------------
    # N-BEATS
    # -------------------------
    if model_type == ModelType.NBEATS:
        if xl_design:
            return NBEATSModel(
                generic_architecture=True,
                num_stacks=model_kwargs.get("num_stacks", 16),
                num_blocks=model_kwargs.get("num_blocks", 2),
                num_layers=model_kwargs.get("num_layers", 6),
                layer_widths=model_kwargs.get("layer_widths", 1024),
                dropout=dropout_eff,
                **base_kwargs,
            )
        else:
            return NBEATSModel(
                generic_architecture=True,
                num_stacks=model_kwargs.get("num_stacks", 10),
                num_blocks=model_kwargs.get("num_blocks", 1),
                num_layers=model_kwargs.get("num_layers", 4),
                layer_widths=model_kwargs.get("layer_widths", 512),
                dropout=dropout_eff,
                **base_kwargs,
            )

    # -------------------------
    # TFT
    # -------------------------
    if model_type == ModelType.TFT:
        if xl_design:
            return TFTModel(
                hidden_size=model_kwargs.get("hidden_size", 128),
                lstm_layers=model_kwargs.get("lstm_layers", 4),
                dropout=dropout_eff,
                num_attention_heads=model_kwargs.get(
                    "num_attention_heads", 12
                ),
                add_relative_index=model_kwargs.get(
                    "add_relative_index", True
                ),
                **base_kwargs,
            )
        else:
            return TFTModel(
                hidden_size=model_kwargs.get("hidden_size", 64),
                lstm_layers=model_kwargs.get("lstm_layers", 2),
                dropout=dropout_eff,
                num_attention_heads=model_kwargs.get(
                    "num_attention_heads", 10
                ),
                add_relative_index=model_kwargs.get(
                    "add_relative_index", True
                ),
                **base_kwargs,
            )

    # -------------------------
    # TSMIXER
    # -------------------------
    if model_type == ModelType.TSMIXER:
        if xl_design:
            return TSMixerModel(
                hidden_size=model_kwargs.get("hidden_size", 128),
                ff_size=model_kwargs.get("ff_size", 256),
                activation=model_kwargs.get("activation", "LeakyReLU"),
                num_blocks=model_kwargs.get("num_blocks", 16),
                dropout=dropout_eff,
                **base_kwargs,
            )
        else:
            return TSMixerModel(
                hidden_size=model_kwargs.get("hidden_size", 64),
                ff_size=model_kwargs.get("ff_size", 128),
                activation=model_kwargs.get("activation", "LeakyReLU"),
                num_blocks=model_kwargs.get("num_blocks", 8),
                dropout=dropout_eff,
                **base_kwargs,
            )

    # -------------------------
    # BlockRNN (LSTM/GRU)
    # -------------------------
    if model_type == ModelType.BLOCK_RNN:
        if xl_design:
            return BlockRNNModel(
                model=model_kwargs.get("model", "GRU"),
                hidden_dim=model_kwargs.get("hidden_dim", 128),
                n_rnn_layers=model_kwargs.get("n_rnn_layers", 3),
                dropout=dropout_eff,
                **base_kwargs,
            )
        else:
            return BlockRNNModel(
                model=model_kwargs.get("model", "GRU"),
                hidden_dim=model_kwargs.get("hidden_dim", 64),
                n_rnn_layers=model_kwargs.get("n_rnn_layers", 2),
                dropout=dropout_eff,
                **base_kwargs,
            )

    # -------------------------
    # TCN
    # -------------------------
    if model_type == ModelType.TCN:
        if xl_design:
            return TCNModel(
                kernel_size=model_kwargs.get("kernel_size", 2),
                num_filters=model_kwargs.get("num_filters", 128),
                dilation_base=model_kwargs.get("dilation_base", 7),
                num_layers=model_kwargs.get("num_layers", None),
                weight_norm=model_kwargs.get("weight_norm", True),
                dropout=dropout_eff,
                **base_kwargs,
            )
        else:
            return TCNModel(
                kernel_size=model_kwargs.get("kernel_size", 2),
                num_filters=model_kwargs.get("num_filters", 16),
                dilation_base=model_kwargs.get("dilation_base", 2),
                num_layers=model_kwargs.get("num_layers", 2),
                weight_norm=model_kwargs.get("weight_norm", True),
                dropout=dropout_eff,
                **base_kwargs,
            )

    # -------------------------
    # TiDE
    # -------------------------
    if model_type == ModelType.TIDE:
        if xl_design:
            return TiDEModel(
                hidden_size=model_kwargs.get("hidden_size", 128),
                dropout=dropout_eff,
                use_layer_norm=model_kwargs.get("use_layer_norm", True),
                **base_kwargs,
            )
        else:
            return TiDEModel(
                hidden_size=model_kwargs.get("hidden_size", 64),
                dropout=dropout_eff,
                use_layer_norm=model_kwargs.get("use_layer_norm", True),
                **base_kwargs,
            )

    raise ValueError(f"Unsupported model type: {model_type}")


def evaluate_model_with_covariates_optuna(
    model_type: ModelType,
    model: ForecastingModel,
    series_meta: List[Dict[str, Any]],
    past_covs: bool,
    future_covs: bool,
) -> float:
    """
    Wrapper around `eval_global_model_with_covariates` for Optuna.

    - Creates a fresh metrics_df for this trial.
    - Runs the full global pipeline (fit + per-series evaluation).
    - Returns the mean RMSSE across all valid (store,item) rows.

    Parameters
    ----------
    model_type : ModelType
        Enum indicating the model type (e.g., NBEATS, TSMIXER).
    model : ForecastingModel
        Unfitted Darts model instance for this trial.
    series_meta : List[Dict[str, Any]]
        List of metadata dicts for each (store,item), as expected by
        `eval_global_model_with_covariates`. Each dict must have keys:
        "store", "item", "data_dict".
    past_covs : bool
        Whether to use past covariates (if supported by the model type).
    future_covs : bool
        Whether to use future covariates (if supported by the model type).

    Returns
    -------
    float
        Mean RMSSE over all valid runs. If no valid runs, returns +inf.
    """

    # Fresh metrics_df for this trial, same schema as main pipeline
    metrics_cols = [
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
    metrics_df = pd.DataFrame(columns=metrics_cols)

    metrics_df = eval_global_model_with_covariates(
        model_type=model_type,
        model=model,
        series_meta=series_meta,
        metrics_df=metrics_df,
        past_covs=past_covs,
        future_covs=future_covs,
    )

    if metrics_df.empty or "RMSSE" not in metrics_df.columns:
        logger.warning(
            f"[Optuna] No metrics or RMSSE column for {model_type.value}; "
            "returning +inf."
        )
        return float("inf")

    clean = metrics_df.dropna(subset=["RMSSE"])
    if clean.empty:
        logger.warning(
            f"[Optuna] All RMSSE values are NaN for {model_type.value}; "
            "returning +inf."
        )
        return float("inf")

    rmsse_mean = float(clean["RMSSE"].mean())
    logger.info(
        f"[Optuna] {model_type.value} trial completed. "
        f"Mean RMSSE = {rmsse_mean:.4f} over {len(clean)} series."
    )
    return rmsse_mean


def make_optuna_objective_global(
    model_type: ModelType,
    train_series: List[TimeSeries],
    series_meta: List[Dict[str, Any]],
    past_covs: bool = True,
    future_covs: bool = True,
    xl_design: bool = False,
    patience: int = 8,
) -> Callable[[optuna.Trial], float]:

    if model_type not in DL_MODELS:
        raise ValueError(
            f"make_optuna_objective_global currently supports only DL models; "
            f"got {model_type.value}"
        )

    use_past = past_covs and model_type.supports_past
    use_future = future_covs and model_type.supports_future

    if past_covs and not model_type.supports_past:
        logger.warning(
            "Model %s does not support past covariates; "
            "disabling past_covs for HPO.",
            model_type.value,
        )
    if future_covs and not model_type.supports_future:
        logger.warning(
            "Model %s does not support future covariates; "
            "disabling future_covs for HPO.",
            model_type.value,
        )

    def objective(trial: optuna.Trial) -> float:
        # --------------------------------------------------------------
        # 1) Training-level hyperparameters
        # --------------------------------------------------------------
        input_chunk_length = trial.suggest_int(
            "input_chunk_length", 30, 60, step=10
        )
        n_epochs = trial.suggest_int("n_epochs", 50, 300, step=25)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.0, 0.3)

        lr = trial.suggest_float("lr", 5e-4, 3e-3, log=True)

        use_weight_decay = trial.suggest_categorical(
            "use_weight_decay", [False, True]
        )
        if use_weight_decay:
            weight_decay = trial.suggest_float(
                "weight_decay", 1e-6, 1e-3, log=True
            )
        else:
            weight_decay = 0.0

        # --------------------------------------------------------------
        # 2) Global RMSSE scale + metrics + EarlyStopping on val_RMSSE
        # --------------------------------------------------------------
        rmse_naive = compute_global_rmsse_scale_from_train_list(train_series)
        torch_metrics = MetricCollection({"RMSSE": RMSSEMetric(rmse_naive)})

        es_callback = EarlyStopping(
            monitor="val_RMSSE",
            patience=patience,
            mode="min",
            verbose=False,
        )

        # torch_kwargs: ONLY trainer/metrics-related stuff
        torch_kwargs: Dict[str, Any] = {
            "torch_metrics": torch_metrics,
            "pl_trainer_kwargs": {
                "callbacks": [es_callback],
                "enable_checkpointing": False,
            },
        }

        # --------------------------------------------------------------
        # 3) Model-specific architecture hyperparameters
        # --------------------------------------------------------------
        model_kwargs: Dict[str, Any] = {}

        if model_type == ModelType.NBEATS:
            model_kwargs["num_stacks"] = trial.suggest_int("num_stacks", 8, 16)
            model_kwargs["num_blocks"] = trial.suggest_int("num_blocks", 1, 3)
            model_kwargs["num_layers"] = trial.suggest_int("num_layers", 3, 6)
            model_kwargs["layer_widths"] = trial.suggest_categorical(
                "layer_widths", [256, 512, 1024]
            )

        elif model_type == ModelType.TFT:
            model_kwargs["hidden_size"] = trial.suggest_categorical(
                "hidden_size", [32, 64, 128]
            )
            model_kwargs["lstm_layers"] = trial.suggest_int(
                "lstm_layers", 1, 4
            )
            model_kwargs["num_attention_heads"] = trial.suggest_int(
                "num_attention_heads", 4, 12
            )

        elif model_type == ModelType.TSMIXER:
            model_kwargs["hidden_size"] = trial.suggest_categorical(
                "hidden_size", [64, 128, 256]
            )
            model_kwargs["ff_size"] = trial.suggest_categorical(
                "ff_size", [128, 256, 512]
            )
            model_kwargs["num_blocks"] = trial.suggest_int(
                "num_blocks", 4, 16, step=4
            )
            model_kwargs["activation"] = trial.suggest_categorical(
                "activation", ["LeakyReLU", "ELU", "GELU", "RReLU", "SELU"]
            )

        elif model_type == ModelType.BLOCK_RNN:
            model_kwargs["hidden_dim"] = trial.suggest_categorical(
                "hidden_dim", [32, 64, 128, 256]
            )
            model_kwargs["n_rnn_layers"] = trial.suggest_int(
                "n_rnn_layers", 1, 4
            )
            model_kwargs["model"] = trial.suggest_categorical(
                "rnn_type", ["GRU", "LSTM"]
            )

        elif model_type == ModelType.TCN:
            model_kwargs["kernel_size"] = trial.suggest_int(
                "kernel_size", 2, 4
            )
            model_kwargs["num_filters"] = trial.suggest_categorical(
                "num_filters", [16, 32, 64, 128]
            )
            model_kwargs["dilation_base"] = trial.suggest_categorical(
                "dilation_base", [2, 4, 7]
            )
            model_kwargs["num_layers"] = trial.suggest_int("num_layers", 1, 4)

        elif model_type == ModelType.TIDE:
            model_kwargs["hidden_size"] = trial.suggest_categorical(
                "hidden_size", [32, 64, 128, 256]
            )
            model_kwargs["num_encoder_layers"] = trial.suggest_int(
                "num_encoder_layers", 1, 4
            )
            model_kwargs["num_decoder_layers"] = trial.suggest_int(
                "num_decoder_layers", 1, 4
            )
            model_kwargs["decoder_output_dim"] = trial.suggest_int(
                "decoder_output_dim", 16, 128, step=16
            )
            model_kwargs["temporal_width_past"] = trial.suggest_int(
                "temporal_width_past", 4, 10, step=2
            )
            model_kwargs["temporal_width_future"] = trial.suggest_int(
                "temporal_width_past", 4, 10, step=2
            )
            model_kwargs["temporal_decoder_hidden"] = trial.suggest_int(
                "temporal_decoder_hidden", 16, 128, step=16
            )

        # --------------------------------------------------------------
        # 4) Pack everything into model_cfg
        # --------------------------------------------------------------
        model_cfg = {
            "input_chunk_length": input_chunk_length,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "dropout": dropout,
            "optimizer_kwargs": {"lr": lr, "weight_decay": weight_decay},
            "model_kwargs": model_kwargs,
        }

        # Create model via your factory
        model: ForecastingModel = create_model(
            model_type=model_type,
            batch_size=batch_size,  # will be overridden by cfg
            n_epochs=n_epochs,  # idem
            lr=lr,  # idem
            dropout=dropout,  # idem
            model_cfg=model_cfg,
            torch_kwargs=torch_kwargs,
            xl_design=xl_design,
            past_covs=use_past,
            future_covs=use_future,
        )

        # --------------------------------------------------------------
        # 5) Evaluate via the global covariate-aware pipeline
        # --------------------------------------------------------------
        score = evaluate_model_with_covariates_optuna(
            model_type=model_type,
            model=model,
            series_meta=series_meta,
            past_covs=use_past,
            future_covs=use_future,
        )

        return float(score)

    return objective


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
) -> Tuple[
    List[TimeSeries],
    List[Dict[str, Any]],
]:
    """
    Build lists of train/val target series and covariates for a GLOBAL model.
    Returns:
        train_targets,
        meta_list (each meta has store, item, and data_dict)
    """
    train_targets: List[TimeSeries] = []
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
        train_targets.append(train_target)

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

    return train_targets, meta_list


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
    """

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

    # ------------------------------------------------------------------ #
    # Guard: local-only models cannot be trained as global models
    # ------------------------------------------------------------------ #
    if model_type.supports_local:
        logger.error(
            f"Model {model_type.value} is local-only in Darts and cannot be "
            "trained as a global model (expects a single TimeSeries, not a list)."
        )
        return metrics_df

    # Decide once whether we require past/future covariates globally
    needs_past = model_type.supports_past and past_covs
    needs_future = model_type.supports_future and future_covs

    # ------------------------------------------------------------------ #
    # 1) Preprocess all series: scaling + checks
    # ------------------------------------------------------------------ #
    train_targets_scaled: List[TimeSeries] = []
    val_targets_scaled: List[TimeSeries] = []

    train_pasts_scaled: List[TimeSeries] = []
    val_pasts_scaled: List[TimeSeries] = []
    full_pasts_scaled: List[TimeSeries] = []

    train_futures_scaled: List[TimeSeries] = []
    val_futures_scaled: List[TimeSeries] = []
    full_futures_scaled: List[TimeSeries] = []

    valid_series_meta: List[Dict[str, Any]] = []

    min_required_length = getattr(model, "min_train_series_length", None)
    # If that fails (some older custom models), manually calculate based on type
    if min_required_length is None:
        input_chunk = getattr(model, "input_chunk_length", None)
        output_chunk = getattr(model, "output_chunk_length", 1)

        # Check for Regression Lags
        lags_dict = getattr(model, "lags", None)

        if input_chunk is not None:
            min_required_length = input_chunk + output_chunk
        elif lags_dict is not None:
            # If lags are present, we need at least max(lags) + output
            # (Simplification: just ensure we have enough for 1 training sample)
            if isinstance(lags_dict, int):
                min_required_length = lags_dict + output_chunk
            elif isinstance(lags_dict, (list, tuple)):
                min_required_length = max(lags_dict) + output_chunk
            elif isinstance(lags_dict, dict):
                # Extract max lag from dictionary keys/values
                all_lags = []
                for k, v in lags_dict.items():
                    if v is None:
                        continue
                    if isinstance(v, int):
                        all_lags.append(v)
                    elif isinstance(v, list):
                        all_lags.extend(v)
                min_required_length = (
                    max(all_lags) + output_chunk if all_lags else 1
                )
        else:
            min_required_length = 1
    logger.info(f"min_required_length: {min_required_length}")
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

            # Length / variance guards
            if len(train_target) < min_required_length:
                reason = (
                    f"training series too short (len={len(train_target)}) "
                    f"< required={min_required_length}"
                )
                _append_nan_row(store, item, reason)
                continue

            if (
                model_type.supports_val
                and len(val_target) < min_required_length
            ):
                reason = (
                    f"validation series too short (len={len(val_target)}) "
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

            # If we intend to use past/future covariates globally,
            # drop series that do not have them.
            if needs_past and data_dict.get("train_past") is None:
                _append_nan_row(
                    store,
                    item,
                    "missing past covariates for global model",
                )
                continue

            if needs_future and data_dict.get("train_future") is None:
                _append_nan_row(
                    store,
                    item,
                    "missing future covariates for global model",
                )
                continue

            # -------------------------- TARGET SCALER --------------------------
            target_scaler = Scaler(RobustScaler())
            train_target_scaled = target_scaler.fit_transform(train_target)
            val_target_scaled = target_scaler.transform(val_target)

            train_targets_scaled.append(train_target_scaled)
            val_targets_scaled.append(val_target_scaled)

            # ------------------------ PAST COVARIATES --------------------------
            train_past_scaled = None
            val_past_scaled = None
            full_past_scaled = None

            if needs_past:
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

                train_pasts_scaled.append(train_past_scaled)
                val_pasts_scaled.append(val_past_scaled)
                full_pasts_scaled.append(full_past_scaled)

            # ----------------------- FUTURE COVARIATES -------------------------
            train_future_scaled = None
            val_future_scaled = None
            full_future_scaled = None

            if needs_future:
                future_scaler = Scaler(RobustScaler())
                train_future_scaled = future_scaler.fit_transform(
                    data_dict["train_future"]
                )

                if data_dict.get("val_future") is not None:
                    val_future_scaled = future_scaler.transform(
                        data_dict["val_future"]
                    )

                if data_dict.get("full_future") is not None:
                    full_future_scaled = future_scaler.transform(
                        data_dict["full_future"]
                    )

                train_futures_scaled.append(train_future_scaled)
                val_futures_scaled.append(val_future_scaled)
                full_futures_scaled.append(full_future_scaled)

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
                    "full_past_scaled": full_past_scaled,
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

    # ------------------------------------------------------------------ #
    # 2) Fit global model ONCE on all training series
    # ------------------------------------------------------------------ #
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

        if model_type.supports_val:
            fit_kwargs["val_series"] = val_targets_scaled

        if needs_past and len(train_pasts_scaled) > 0:
            fit_kwargs["past_covariates"] = train_pasts_scaled
            if model_type.supports_val:
                fit_kwargs["val_past_covariates"] = val_pasts_scaled

        if needs_future and len(train_futures_scaled) > 0:
            fit_kwargs["future_covariates"] = train_futures_scaled
            if model_type.supports_val:
                fit_kwargs["val_future_covariates"] = val_futures_scaled

        logger.debug(f"Global fit kwargs keys: {list(fit_kwargs.keys())}")
        model.fit(**fit_kwargs)

    except Exception as e:
        logger.error(
            f"Error fitting global {model_type}-S:{store}/I:{item}: {e}"
        )
        logger.error(traceback.format_exc())
        for meta in valid_series_meta:
            _append_nan_row(meta["store"], meta["item"], "global fit failed")
        return metrics_df

    # ------------------------------------------------------------------ #
    # 3) Per-series prediction + metrics
    # ------------------------------------------------------------------ #
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

            if needs_past and full_past_scaled is not None:
                predict_kwargs["past_covariates"] = full_past_scaled

            if needs_future and full_future_scaled is not None:
                predict_kwargs["future_covariates"] = full_future_scaled

            logger.debug(
                f"Predict kwargs keys for S{store}/I{item}: "
                f"{list(predict_kwargs.keys())}"
            )
            forecast_scaled = model.predict(**predict_kwargs)
            forecast = target_scaler.inverse_transform(forecast_scaled)

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
                f"RMSSE: {metrics['rmsse']:.4f}, "
                f"SMAPE: {metrics['smape']:.4f}, "
                f"MARRE: {metrics['marre']:.4f}"
            )

        except Exception as e:
            logger.error(
                f"Error predicting/evaluating {model_type.value} "
                f"for S{store}/I{item}: {e}"
            )
            logger.error(traceback.format_exc())
            _append_nan_row(store, item, "prediction/eval failed")

    return metrics_df
