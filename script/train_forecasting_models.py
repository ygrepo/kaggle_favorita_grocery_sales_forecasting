#!/usr/bin/env python3
"""
Unified training script for Favorita / Growth Rate Forecasting.

Supports:
- Classical local models (ExponentialSmoothing, AutoARIMA, Theta, KalmanForecaster)
- Tree-based models (LightGBM, RandomForest, LinearRegressionModel)
- Deep learning models (NBEATS, TFT, TSMIXER, BLOCK_RNN, TCN, TIDE)

All models share the same covariate-aware pipeline:
- prepare_store_item_series()
- get_train_val_data_with_covariates()
- eval_model_with_covariates()
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, List

import traceback

from darts import TimeSeries
import pandas as pd


# Optional DL / GPU imports (used only for deep-learning models)
import torch
from darts.utils.callbacks import TFMProgressBar
from torchmetrics import MetricCollection
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything


# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import (
    setup_logging,
    get_logger,
    save_csv_or_parquet,
    str2bool,
)
from src.data_utils import load_raw_data
from src.time_series_utils import (
    ModelType,
    parse_models_arg,
    create_model,
    generate_torch_kwargs_global,
    build_global_train_val_lists,
    eval_global_model_with_covariates,
    compute_rmsse_scale_from_train,
    RMSSEMetric,
    FUTURE_COV_COLS,
    PAST_COV_COLS,
    STATIC_COV_COLS,
    DL_MODELS,
)
from src.model_utils import get_first_free_gpu

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Torch / Lightning configuration (for DL models only)
# ---------------------------------------------------------------------


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


# ---------------------------------------------------------------------
# Per-(store, item) processing
# ---------------------------------------------------------------------


def train(
    df: pd.DataFrame,
    model_types: List[ModelType],
    split_point: float,
    min_train_data_points: int,
    model_dir: Path,
    patience: int,
    batch_size: int,
    n_epochs: int,
    dropout: float,
    past_covs: bool,
    future_covs: bool,
    xl_design: bool,
    metrics_df: pd.DataFrame,
    gpu_id: Optional[int],
    store_medians_fn: Path | None,
    item_medians_fn: Path | None,
    store_assign_fn: Path | None,
    item_assign_fn: Path | None,
):
    try:
        logger.info("Processing all (store, item) combinations")
        (
            train_targets,
            *_,
            meta_list,
        ) = build_global_train_val_lists(
            df,
            split_point=split_point,
            min_train_data_points=min_train_data_points,
            store_medians_fn=store_medians_fn,
            item_medians_fn=item_medians_fn,
            store_assign_fn=store_assign_fn,
            item_assign_fn=item_assign_fn,
            past_covs=past_covs,
            future_covs=future_covs,
        )

        # Split requested models into global-capable vs local-only
        global_model_types = [m for m in model_types if m.supports_global]
        local_only_model_types = [m for m in model_types if m.supports_local]

        if local_only_model_types:
            logger.warning(
                "The following models are local-only in Darts and will be "
                "skipped in the GLOBAL pipeline: "
                f"{[m.value for m in local_only_model_types]}"
            )

        # Train each model requested
        for mtype in global_model_types:
            logger.info(f"=== Starting model {mtype.value} ===")

            if not mtype.supports_local:
                logger.info(f"Skipping {mtype.value} ( local)")
                continue

            # For DL models, we set up a model_dir and torch_kwargs
            if mtype in DL_MODELS:
                model_dir_for_model = model_dir.resolve() / mtype.value
                model_dir_for_model.mkdir(parents=True, exist_ok=True)
                torch_kwargs = generate_torch_kwargs_global(
                    gpu_id=gpu_id,
                    working_dir=model_dir_for_model,
                    train_series=train_targets,
                    patience=patience,
                )
                batch_size_for_model = batch_size
                n_epochs_for_model = n_epochs
                dropout_for_model = dropout
            else:
                # Classical / tree-based models do not need PL trainer
                torch_kwargs = None
                batch_size_for_model = batch_size
                n_epochs_for_model = n_epochs
                dropout_for_model = dropout

            model = create_model(
                model_type=mtype,
                batch_size=batch_size_for_model,
                torch_kwargs=torch_kwargs,
                n_epochs=n_epochs_for_model,
                dropout=dropout_for_model,
                xl_design=xl_design,
                past_covs=past_covs,
                future_covs=future_covs,
            )

            metrics_df = eval_global_model_with_covariates(
                mtype,
                model,
                meta_list,
                metrics_df,
                past_covs=past_covs,
                future_covs=future_covs,
            )

        return metrics_df

    except Exception as e:
        logger.error(f"ERROR: {e}")
        logger.error(traceback.format_exc())
        return metrics_df


# ---------------------------------------------------------------------
# Argparse + main
# ---------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Unified Favorita / Growth Rate forecasting benchmark "
            "with classical, tree-based, and deep-learning models."
        )
    )

    # Data + paths
    parser.add_argument("--data_fn", type=Path, default="")
    parser.add_argument("--store_medians_fn", type=Path, default=None)
    parser.add_argument("--item_medians_fn", type=Path, default=None)
    parser.add_argument("--store_assign_fn", type=Path, default=None)
    parser.add_argument("--item_assign_fn", type=Path, default=None)
    parser.add_argument("--model_dir", type=Path, default="models")
    parser.add_argument("--metrics_fn", type=Path, default="")
    parser.add_argument(
        "--sample",
        type=str2bool,
        default=False,
        help="Sample N random (store,item) pairs instead of taking first N",
    )
    parser.add_argument("--log_fn", type=Path, default="")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    # Models
    parser.add_argument(
        "--models",
        type=str,
        default="EXPONENTIAL_SMOOTHING,AUTO_ARIMA,THETA,KALMAN",
        help=(
            "Comma-separated model list, e.g. "
            "'EXPONENTIAL_SMOOTHING,AUTO_ARIMA,THETA,KALMAN,"
            "LIGHTGBM,RANDOM_FOREST,LINEAR_REGRESSION,NBEATS,TFT,TSMIXER,TCN,BLOCK_RNN,TIDE'"
        ),
    )

    # Training / splitting
    parser.add_argument("--split_point", type=float, default=0.8)
    parser.add_argument("--min_train_data_points", type=int, default=35)
    parser.add_argument(
        "--N", type=int, default=0, help="Limit to first N (store,item) pairs"
    )

    # DL hyperparameters (ignored for classical / tree models)
    parser.add_argument(
        "--batch_size", type=int, default=800, help="DL batch size"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=15, help="DL number of epochs"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="DL dropout rate"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="DL early-stopping patience (epochs)",
    )
    parser.add_argument(
        "--xl_design",
        type=str2bool,
        default=False,
        help="Use XL design for models (DL + tree + linear_regression)",
    )

    # Covariates
    parser.add_argument(
        "--past_covs",
        type=str2bool,
        default=True,
        help="Use past covariates when supported",
    )
    parser.add_argument(
        "--future_covs",
        type=str2bool,
        default=True,
        help="Use future covariates when supported",
    )

    # GPU
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="Specific GPU id to use; if -1, round-robin over all GPUs",
    )

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    # Parse model list
    model_types = parse_models_arg(args.models)

    # Setup logging
    logger = setup_logging(args.log_fn, args.log_level)
    logger.info(f"Training models: {[m.value for m in model_types]}")
    logger.info(f"Data fn: {args.data_fn}")
    logger.info(f"Metrics fn: {args.metrics_fn}")
    logger.info(f"Model dir: {args.model_dir}")
    logger.info(f"Log fn: {args.log_fn}")
    logger.info(f"Split point: {args.split_point}")
    logger.info(f"Min train data points: {args.min_train_data_points}")
    logger.info(f"Limit to first N combinations: {args.N}")
    logger.info(f"XL design: {args.xl_design}")
    logger.info(f"Past covs: {args.past_covs}")
    logger.info(f"Future covs: {args.future_covs}")
    logger.info(f"Batch size (DL): {args.batch_size}")
    logger.info(f"n_epochs (DL): {args.n_epochs}")
    logger.info(f"Dropout (DL): {args.dropout}")
    logger.info(f"Patience (DL): {args.patience}")

    # Assign GPU
    gpu_id = get_first_free_gpu()

    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        logger.info(
            f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}"
        )
    else:
        logger.info("Running on CPU")

    # Load data
    df = load_raw_data(args.data_fn)

    # Optional: convert city/state to ids if available
    if "city" in df.columns and "city_id" not in df.columns:
        df["city_id"] = df["city"].astype("category").cat.codes
    if "state" in df.columns and "state_id" not in df.columns:
        df["state_id"] = df["state"].astype("category").cat.codes

    # Drop raw categorical columns if present to avoid confusion
    drop_cols = [
        c
        for c in ["class", "cluster", "family", "city", "state"]
        if c in df.columns
    ]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Loaded data head:")
    logger.info(df.head())

    # Log which covariates are available globally
    available_past_covs = [c for c in PAST_COV_COLS if c in df.columns]
    logger.info(f"Available past covariate columns: {available_past_covs}")
    available_future_covs = [c for c in FUTURE_COV_COLS if c in df.columns]
    logger.info(f"Available future covariate columns: {available_future_covs}")
    available_static_covs = [c for c in STATIC_COV_COLS if c in df.columns]
    logger.info(f"Available static covariate columns: {available_static_covs}")

    # Unique (store, item) pairs
    unique_combinations = df[["store", "item"]].drop_duplicates()
    if args.N > 0:
        if args.sample:
            logger.info(f"Sampling {args.N} random combinations")
            unique_combinations = unique_combinations.sample(
                args.N, random_state=args.seed
            )
        else:
            logger.info(f"Limiting to first {args.N} combinations")
            unique_combinations = unique_combinations.head(args.N)

    logger.info(
        f"Starting processing of {len(unique_combinations)} (store,item) pairs..."
    )
    df.query(
        "store in @unique_combinations['store'] and item in @unique_combinations['item']",
        inplace=True,
    )
    logger.info(f"Filtered data shape: {df.shape}")

    # Metrics dataframe
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
    train(
        df=df,
        model_types=model_types,
        split_point=args.split_point,
        min_train_data_points=args.min_train_data_points,
        model_dir=args.model_dir,
        patience=args.patience,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        dropout=args.dropout,
        past_covs=args.past_covs,
        future_covs=args.future_covs,
        xl_design=args.xl_design,
        metrics_df=metrics_df,
        gpu_id=gpu_id,
        store_medians_fn=args.store_medians_fn,
        item_medians_fn=args.item_medians_fn,
        store_assign_fn=args.store_assign_fn,
        item_assign_fn=args.item_assign_fn,
    )

    # Final save
    metrics_fn = Path(args.metrics_fn).resolve()
    save_csv_or_parquet(metrics_df, metrics_fn)
    logger.info("Benchmark complete.")

    # Summary
    if not metrics_df.empty:
        logger.info("Summary of results:")
        for model_name in metrics_df["Model"].unique():
            model_metrics = metrics_df[metrics_df["Model"] == model_name]
            successful_runs = model_metrics.dropna(
                subset=[
                    "SMAPE",
                    "RMSSE",
                    "MARRE",
                ]
            ).shape[0]
            total_runs = len(model_metrics)
            logger.info(
                f"  {model_name}: {successful_runs}/{total_runs} successful runs"
            )

            if successful_runs > 0:
                clean = model_metrics.dropna(
                    subset=["SMAPE", "RMSSE", "MARRE"]
                )
                if len(clean) > 0:
                    smape_mean = clean["SMAPE"].mean()
                    rmsse_mean = clean["RMSSE"].mean()
                    marre_mean = clean["MARRE"].mean()
                    logger.info(
                        f"    Mean SMAPE: {smape_mean:.4f}, "
                        f"Mean RMSSE: {rmsse_mean:.4f}, "
                        f"Mean MARRE: {marre_mean:.4f}"
                    )
    else:
        logger.warning("No successful model runs completed!")


if __name__ == "__main__":
    main()
