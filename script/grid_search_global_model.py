#!/usr/bin/env python3
"""
Unified Optuna HPO script for Favorita / Growth Rate Forecasting.

Reuses the global covariate-aware pipeline:
- build_global_train_val_lists()
- make_optuna_objective_global()
- eval_global_model_with_covariates()
"""

import sys
import argparse
from pathlib import Path
from typing import List

import traceback

import pandas as pd

# Optional DL / GPU imports (used only for deep-learning models)
import torch
from pytorch_lightning import seed_everything
import optuna  # NEW

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
    make_optuna_objective_global,
    build_global_train_val_lists,
    FUTURE_COV_COLS,
    PAST_COV_COLS,
    STATIC_COV_COLS,
)
from src.model_utils import get_first_free_gpu

logger = get_logger(__name__)


def search(
    df: pd.DataFrame,
    model_types: List[ModelType],
    split_point: float,
    min_train_data_points: int,
    past_covs: bool,
    future_covs: bool,
    xl_design: bool,
    store_medians_fn: Path | None,
    item_medians_fn: Path | None,
    store_assign_fn: Path | None,
    item_assign_fn: Path | None,
    n_trials: int,
    timeout: int | None,
) -> pd.DataFrame:
    """
    Run Optuna search for each global-capable model type.

    Returns
    -------
    pd.DataFrame
        One row per model_type with columns:
        - Model
        - BestObjective
        - param_<name> for each tuned hyperparameter
    """
    results_rows: list[dict] = []

    try:
        logger.info("Processing all (store, item) combinations for HPO")
        (
            _,
            meta_list,
        ) = build_global_train_val_lists(
            df,
            split_point=split_point,
            min_train_data_points=min_train_data_points,
            store_medians_fn=store_medians_fn,
            item_medians_fn=item_medians_fn,
            store_assign_fn=store_assign_fn,
            item_assign_fn=item_assign_fn,
        )

        # Split requested models into global-capable vs local-only
        global_model_types = [m for m in model_types if m.supports_global]
        local_only_model_types = [m for m in model_types if m.supports_local]

        if local_only_model_types:
            logger.warning(
                "The following models are local-only in Darts and will be "
                "skipped in the GLOBAL HPO pipeline: "
                f"{[m.value for m in local_only_model_types]}"
            )

        # Run Optuna for each model requested
        for mtype in global_model_types:
            logger.info(
                f"=== Starting Optuna search for model {mtype.value} ==="
            )

            objective = make_optuna_objective_global(
                model_type=mtype,
                series_meta=meta_list,
                past_covs=past_covs,
                future_covs=future_covs,
                xl_design=xl_design,
            )

            study = optuna.create_study(direction="minimize")
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout if timeout and timeout > 0 else None,
            )

            # Study statistics (mirroring the example slide)
            pruned_trials = [
                t
                for t in study.trials
                if t.state == optuna.trial.TrialState.PRUNED
            ]
            complete_trials = [
                t
                for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]

            logger.info("Study statistics for %s:", mtype.value)
            logger.info("  Number of finished trials: %d", len(study.trials))
            logger.info("  Number of pruned trials:   %d", len(pruned_trials))
            logger.info(
                "  Number of complete trials: %d", len(complete_trials)
            )

            best_trial = study.best_trial
            logger.info("Best trial for %s:", mtype.value)
            logger.info("  Value:  %.6f", best_trial.value)
            logger.info("  Params:")
            for key, value in best_trial.params.items():
                logger.info("    %s: %s", key, value)

            # Collect results into a row
            row: dict = {
                "Model": mtype.value,
                "BestObjective": float(best_trial.value),
            }
            for key, value in best_trial.params.items():
                row[f"param_{key}"] = value
            results_rows.append(row)

    except Exception as e:
        logger.error(f"ERROR during Optuna search: {e}")
        logger.error(traceback.format_exc())

    if results_rows:
        return pd.DataFrame(results_rows)
    else:
        return pd.DataFrame(columns=["Model", "BestObjective"])


# ---------------------------------------------------------------------
# Argparse + main
# ---------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Optuna HPO for Favorita / Growth Rate forecasting "
            "using the unified global covariate-aware pipeline."
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
        default="NBEATS,TSMIXER,TFT,TCN,BLOCK_RNN,TIDE",
        help=(
            "Comma-separated model list, e.g. "
            "'NBEATS,TFT,TSMIXER,TCN,BLOCK_RNN,TIDE'"
        ),
    )

    # Training / splitting
    parser.add_argument("--split_point", type=float, default=0.8)
    parser.add_argument("--min_train_data_points", type=int, default=35)
    parser.add_argument(
        "--N", type=int, default=0, help="Limit to first N (store,item) pairs"
    )

    # DL hyperparameters (they are tuned by Optuna, so these are only defaults)
    parser.add_argument(
        "--batch_size", type=int, default=800, help="DL batch size (baseline)"
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=15,
        help="DL number of epochs (baseline)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="DL dropout rate (baseline)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="DL early-stopping patience (baseline)",
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
    parser.add_argument(
        "--results_fn",
        type=Path,
        default=False,
        help="Path to save results (relative to project root)",
    )
    # GPU
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="Specific GPU id to use; if -1, round-robin over all GPUs",
    )

    # HPO config
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of Optuna trials per model",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Timeout in seconds for Optuna per model (0 means no timeout)",
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
    logger.info(f"Running Optuna for models: {[m.value for m in model_types]}")
    logger.info(f"Data fn: {args.data_fn}")
    logger.info(f"Results fn: {args.results_fn}")
    logger.info(f"Store medians fn: {args.store_medians_fn}")
    logger.info(f"Item medians fn: {args.item_medians_fn}")
    logger.info(f"Model dir: {args.model_dir}")
    logger.info(f"Log fn: {args.log_fn}")
    logger.info(f"Split point: {args.split_point}")
    logger.info(f"Min train data points: {args.min_train_data_points}")
    logger.info(f"Limit to first N combinations: {args.N}")
    logger.info(f"XL design: {args.xl_design}")
    logger.info(f"Past covs: {args.past_covs}")
    logger.info(f"Future covs: {args.future_covs}")
    logger.info(f"Batch size (baseline DL): {args.batch_size}")
    logger.info(f"n_epochs (baseline DL): {args.n_epochs}")
    logger.info(f"Dropout (baseline DL): {args.dropout}")
    logger.info(f"Patience (baseline DL): {args.patience}")
    logger.info(f"Optuna n_trials: {args.n_trials}")
    logger.info(f"Optuna timeout: {args.timeout}")

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

    metrics_df = search(
        df=df,
        model_types=model_types,
        split_point=args.split_point,
        min_train_data_points=args.min_train_data_points,
        past_covs=args.past_covs,
        future_covs=args.future_covs,
        xl_design=args.xl_design,
        store_medians_fn=args.store_medians_fn,
        item_medians_fn=args.item_medians_fn,
        store_assign_fn=args.store_assign_fn,
        item_assign_fn=args.item_assign_fn,
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    # Final save of HPO results
    results_fn = Path(args.results_fn).resolve()
    save_csv_or_parquet(metrics_df, results_fn)
    logger.info("Optuna HPO complete.")

    # Summary of best objective values
    if metrics_df is not None and not metrics_df.empty:
        logger.info("Summary of best objectives per model:")
        for _, row in metrics_df.iterrows():
            model_name = row["Model"]
            best_obj = row["BestObjective"]
            logger.info("  %s: BestObjective = %.4f", model_name, best_obj)
    else:
        logger.warning("No successful Optuna runs completed!")


if __name__ == "__main__":
    main()
