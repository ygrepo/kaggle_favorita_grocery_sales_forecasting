#!/usr/bin/env python3
"""
Training script for the Favorita Grocery Sales Forecasting model.

This script handles the complete training pipeline including:
- Data loading and preprocessing
- Logging
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.metrics import rmse, rmsse, mae, mase, mape, ope, smape
from tqdm import tqdm
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler
from darts.datasets import EnergyDataset
from darts.metrics import r2_score
from darts.models import NBEATSModel
from darts.utils.callbacks import TFMProgressBar
from torchmetrics import SymmetricMeanAbsolutePercentageError

import optuna
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import Callback, EarlyStopping


# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.utils import (
    setup_logging,
    get_logger,
    save_csv_or_parquet,
)
from src.data_utils import load_raw_data
from src.time_series_utils import get_train_val_data
from src.time_series_utils import (
    prepare_store_item_series,
    get_train_val_data,
    calculate_metrics,
    eval_model,
)

logger = get_logger(__name__)


class PatchedPruningCallback(
    optuna.integration.PyTorchLightningPruningCallback, Callback
):
    pass


def generate_torch_kwargs(working_dir: Path) -> dict:
    # throughout training we'll monitor the validation loss for both pruning and early stopping
    pruner = PatchedPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping(
        "val_loss", min_delta=0.001, patience=3, verbose=True
    )
    callbacks = [
        pruner,
        early_stopper,
        TFMProgressBar(enable_train_bar_only=True),
    ]
    # detect if a GPU is available
    if torch.cuda.is_available():
        num_workers = 2
    else:
        num_workers = 0

    # run torch models on CPU, and disable progress bars for all model stages except training.
    return {
        "pl_trainer_kwargs": {
            "accelerator": "auto",
            "callbacks": callbacks,
            "work_dir": working_dir,
            "num_workers": num_workers,
        }
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create features for Favorita Grocery Sales Forecasting model"
    )
    parser.add_argument(
        "--data_fn",
        type=Path,
        default="",
        help="Path to data file (relative to project root)",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default="",
        help="Path to model directory (relative to project root)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="NBEATS",
        help="Model to train",
    )
    parser.add_argument(
        "--metrics_fn",
        type=Path,
        default="",
        help="Path to data file (relative to project root)",
    )
    parser.add_argument(
        "--split_point",
        type=float,
        default=0.8,
        help="Proportion of data to use for training",
    )
    parser.add_argument(
        "--min_train_data_points",
        type=int,
        default=15,
        help="Minimum number of data points to train on",
    )
    parser.add_argument(
        "--log_fn",
        type=Path,
        default="",
        help="Path to save script outputs (relative to project root)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    data_fn = Path(args.data_fn).resolve()
    output_metrics_fn = Path(args.metrics_fn).resolve()
    log_fn = args.log_fn.resolve()

    # Set up logging
    logger = setup_logging(log_fn, args.log_level)
    logger.info(f"Log fn: {log_fn}")

    try:
        # Log configuration
        logger.info(
            "Starting time series model benchmarking with configuration:"
        )
        logger.info(f"  Data fn: {data_fn}")
        logger.info(f"  Model dir: {args.model_dir}")
        logger.info(f"  Metrics fn: {output_metrics_fn}")
        logger.info(f"  Log fn: {log_fn}")
        logger.info(f"  Split point: {args.split_point}")
        logger.info(f"  Min train data points: {args.min_train_data_points}")

        # Load raw data
        logger.info("Loading raw data...")
        df = load_raw_data(data_fn)
        # Get unique store-item pairs
        logger.info("Finding unique store-item combinations...")
        unique_combinations = df[["store", "item"]].drop_duplicates()
        # Initialize metrics dataframe
        logger.info("Running models...")
        metrics_df = pd.DataFrame(
            columns=[
                "Model",
                "Store",
                "Item",
                "RMSE",
                "MAE",
                "SMAPE",
                "OPE",
            ]
        )

        for _, row in tqdm(
            unique_combinations.iterrows(), total=len(unique_combinations)
        ):
            store = row["store"]
            item = row["item"]
            # Prepare time series
            ts_df = prepare_store_item_series(df, store, item)
            if ts_df.empty:
                logger.warning(f"No data for store {store}, item {item}")
                continue

            ts, train_ts, val_ts = get_train_val_data(
                ts_df,
                store,
                item,
                args.split_point,
                args.min_train_data_points,
            )

            # "Look at 30 days, predict the next 1 day."
            model_name = args.model
            model_dir = (
                args.model_dir.resolve() / f"{model_name}_{store}_{item}"
            )
            model = NBEATSModel(
                input_chunk_length=30,
                output_chunk_length=1,
                generic_architecture=True,
                num_stacks=10,
                num_blocks=1,
                num_layers=4,
                layer_widths=512,
                n_epochs=100,
                nr_epochs_val_period=1,
                batch_size=800,
                random_state=42,
                model_name=model_name,
                save_checkpoints=True,
                force_reset=True,
                **generate_torch_kwargs(model_dir),
            )

            # Train your model on the 584-day training set
            model.fit(train_ts)
            model = NBEATSModel.load_from_checkpoint(
                model_name=model_name, best=True
            )
            forecast = model.historical_forecasts(
                ts,
                start=val_ts.start_time(),
                forecast_horizon=7,
                stride=7,
                last_points_only=False,
                retrain=False,
                verbose=True,
            )
            forecast = concatenate(forecast)

            metrics = calculate_metrics(train_ts, val_ts, forecast)

            new_row = pd.DataFrame(
                [
                    {
                        "Model": model_name,
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

        logger.info(f"Saving results to {output_metrics_fn}")
        save_csv_or_parquet(metrics_df, output_metrics_fn)
        logger.info("Benchmarking completed successfully!")

    except Exception as e:
        logger.error(f"Error in benchmarking: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
