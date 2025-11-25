#!/usr/bin/env python3
"""
Training script for the Growth Rate Forecasting model.

Supports training multiple models per (store, item) with past and future covariates.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, List

import traceback

import pandas as pd
from tqdm import tqdm

from darts.utils.callbacks import TFMProgressBar

import torch
from torchmetrics import SymmetricMeanAbsolutePercentageError, MetricCollection
from pytorch_lightning.callbacks import EarlyStopping

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
    prepare_store_item_series,
    get_train_val_data_with_covariates,
    eval_model_with_covariates,
)

logger = get_logger(__name__)


def generate_torch_kwargs(
    gpu_id: Optional[int], working_dir: Path, patience: int = 8
) -> Dict:
    """Return trainer kwargs + torch_metrics for a specific GPU."""
    # Increase patience slightly as models with covariates might take longer to converge
    early_stopper = EarlyStopping(
        monitor="train_smape",
        min_delta=0.001,
        patience=patience,
        verbose=True,
        mode="min",
    )
    callbacks = [
        early_stopper,
        TFMProgressBar(enable_train_bar_only=True),
    ]

    if gpu_id is not None and torch.cuda.is_available():
        accelerator = "gpu"
        devices = [gpu_id]
        logger.debug(f"Using GPU {gpu_id}")
    else:
        accelerator = "auto"
        devices = "auto"

    metrics = MetricCollection(
        {"smape": SymmetricMeanAbsolutePercentageError()}
    )

    return {
        "pl_trainer_kwargs": {
            "accelerator": accelerator,
            "devices": devices,
            "callbacks": callbacks,
            "default_root_dir": str(working_dir),
        },
        "torch_metrics": metrics,
    }


# ---------------------------------------------------------------------
# Train all models for a given (store, item)
# ---------------------------------------------------------------------


def process_store_item(
    store: int,
    item: int,
    gpu_id: Optional[int],
    df: pd.DataFrame,
    args,
    model_types: List[ModelType],
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Train ALL requested models for a single store-item pair with covariates."""
    try:
        # Prepare DataFrame with target and all covariates
        ts_df = prepare_store_item_series(df, store, item)
        if ts_df.empty:
            return metrics_df

        # Get Train/Val splits for Target, Past Covs, and Future Covs
        data_dict = get_train_val_data_with_covariates(
            ts_df, store, item, args.split_point, args.min_train_data_points
        )

        if data_dict is None:
            return metrics_df

        # Determine covariate dimensions from the data
        # (n_components returns the number of columns in the TimeSeries)
        p_dim = (
            data_dict["train_past"].n_components
            if data_dict["train_past"]
            else 0
        )
        f_dim = (
            data_dict["train_future"].n_components
            if data_dict["train_future"]
            else 0
        )
        logger.info(
            f"S{store}/I{item}: Data prepared. Train len: {len(data_dict['train_target'])}, "
            f"Val len: {len(data_dict['val_target'])}, Past Dim: {p_dim}, Future Dim: {f_dim}"
        )
        # Train each model requested
        for mtype in model_types:
            model_dir = (
                args.model_dir.resolve()
                / mtype.value
                / f"store_{store}_item_{item}"
            )
            model_dir.mkdir(parents=True, exist_ok=True)
            torch_kwargs = generate_torch_kwargs(
                gpu_id, model_dir, args.patience
            )

            # Pass dimensions to factory
            model = create_model(
                mtype,
                args.batch_size,
                torch_kwargs,
                args.n_epochs,
                args.dropout,
                args.xl_design,
            )

            logger.info(
                f"[GPU {gpu_id}] Training {mtype.value} for store={store}, item={item} "
                f"(Past Dim: {p_dim}, Future Dim: {f_dim})"
            )

            # Pass the full data dictionary containing splits to evaluation
            metrics_df = eval_model_with_covariates(
                mtype.value,
                model,
                store,
                item,
                data_dict,
                metrics_df,
                args.no_past_covs,
                args.no_future_covs,
            )

        return metrics_df

    except Exception as e:
        logger.error(f"ERROR for store={store}, item={item}: {e}")
        logger.error(traceback.format_exc())
        return metrics_df


# ---------------------------------------------------------------------
# Argparse + main
# ---------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Favorita forecasting benchmark with covariates"
    )

    parser.add_argument("--data_fn", type=Path, default="")
    parser.add_argument("--model_dir", type=Path, default="")
    parser.add_argument(
        "--models",
        type=str,
        default="NBEATS",
        help="Comma-separated model list: 'NBEATS,TFT,TSMIXER,TCN,BLOCK_RNN,TIDE'",
    )
    parser.add_argument(
        "--batch_size", type=int, default=800, help="Override batch_size"
    )
    # Reduced default workers to prevent excessive memory usage with complex dataloaders
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Override num_workers"
    )
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--metrics_fn", type=Path, default="")
    parser.add_argument("--split_point", type=float, default=0.8)
    parser.add_argument("--min_train_data_points", type=int, default=35)
    parser.add_argument(
        "--N", type=int, default=0, help="Limit to first N combinations"
    )
    parser.add_argument("--log_fn", type=Path, default="")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--n_epochs", type=int, default=15)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--xl_design", type=str2bool, default=False)
    parser.add_argument("--no_past_covs", type=str2bool, default=False)
    parser.add_argument("--no_future_covs", type=str2bool, default=False)
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse model list
    model_types = parse_models_arg(args.models)

    # Setup logging
    logger = setup_logging(args.log_fn, args.log_level)
    logger.info(f"Training models: {[m.value for m in model_types]}")

    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Number of workers: {args.num_workers}")
    logger.info(f"Dropout: {args.dropout}")
    logger.info(f"Patience: {args.patience}")
    logger.info(f"XL design: {args.xl_design}")
    logger.info(f"No past covs: {args.no_past_covs}")
    logger.info(f"No future covs: {args.no_future_covs}")
    logger.info(f"N epochs: {args.n_epochs}")
    logger.info(f"Split point: {args.split_point}")
    logger.info(f"Min train data points: {args.min_train_data_points}")
    logger.info(f"Limit to first N combinations: {args.N}")
    logger.info(f"Data fn: {args.data_fn}")
    logger.info(f"Model dir: {args.model_dir}")
    logger.info(f"Metrics fn: {args.metrics_fn}")
    logger.info(f"Log fn: {args.log_fn}")

    # GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        num_gpus = 0
        logger.warning("No GPUs detected — running on CPU")

    df = load_raw_data(args.data_fn)
    logger.info("Loaded data head:")
    logger.info(df.head())

    unique_combinations = df[["store", "item"]].drop_duplicates()
    if args.N > 0:
        logger.info(f"Limiting to first {args.N} combinations")
        unique_combinations = unique_combinations.head(args.N)

    # Initialize metrics dataframe
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

    logger.info(f"Starting processing of {len(unique_combinations)} pairs...")

    for idx, (store, item) in tqdm(
        enumerate(
            unique_combinations[["store", "item"]].itertuples(
                index=False, name=None
            )
        ),
        total=len(unique_combinations),
        desc="Store/Item Pairs",
    ):

        # GPU assignment (round-robin)
        if num_gpus == 0:
            gpu_id = None
        elif args.gpu >= 0:
            gpu_id = args.gpu
        else:
            gpu_id = idx % num_gpus

        metrics_df = process_store_item(
            store, item, gpu_id, df, args, model_types, metrics_df
        )

        # Periodic save
        if (idx + 1) % 10 == 0:
            metrics_fn = Path(args.metrics_fn).resolve()
            save_csv_or_parquet(metrics_df, metrics_fn)

    metrics_fn = Path(args.metrics_fn).resolve()
    save_csv_or_parquet(metrics_df, metrics_fn)

    logger.info("Benchmark complete.")

    # Log summary statistics
    if not metrics_df.empty:
        logger.info("Summary of results:")
        for model in metrics_df["Model"].unique():
            model_metrics = metrics_df[metrics_df["Model"] == model]
            successful_runs = model_metrics.dropna(subset=["SMAPE"]).shape[0]
            total_runs = len(model_metrics)
            logger.info(
                f"  {model}: {successful_runs}/{total_runs} successful runs"
            )

            if successful_runs > 0:
                clean_metrics = model_metrics.dropna(subset=["SMAPE", "RMSSE"])
                if len(clean_metrics) > 0:
                    smape_mean = clean_metrics["SMAPE"].mean()
                    rmsse_mean = clean_metrics["RMSSE"].mean()
                    logger.info(
                        f"    Mean SMAPE: {smape_mean:.4f}, "
                        f"Mean RMSSE: {rmsse_mean:.4f}"
                    )
    else:
        logger.warning("No successful model runs completed!")


if __name__ == "__main__":
    main()
