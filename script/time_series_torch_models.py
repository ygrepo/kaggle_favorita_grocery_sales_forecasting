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
from tqdm import tqdm

from darts import TimeSeries
from darts.models import NBEATSModel
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
)
from src.data_utils import load_raw_data
from src.time_series_utils import (
    prepare_store_item_series,
    get_train_val_data,
    calculate_metrics,
)

logger = get_logger(__name__)


def generate_torch_kwargs(gpu_id: int, working_dir: Path) -> dict:
    """
    Build pl_trainer_kwargs and torch_metrics for Darts' TorchForecastingModel.

    Note: we set a specific GPU device per (store, item) via `devices=[gpu_id]`.
    """
    # Early stopping; you may want to change monitor to "val_loss" if needed.
    early_stopper = EarlyStopping(
        monitor="train_smape",
        min_delta=0.001,
        patience=6,
        verbose=True,
        mode="min",  # SMAPE should be minimized
    )
    callbacks = [
        early_stopper,
        TFMProgressBar(enable_train_bar_only=True),
    ]

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        accelerator = "gpu"
        devices = [gpu_id]
        logger.debug(f"Using GPU {gpu_id} for this model.")
    else:
        accelerator = "auto"  # fallback to 'cpu' or 'auto'
        devices = "auto"
        logger.debug("No CUDA device detected; using CPU/auto.")

    metrics = MetricCollection(
        {
            "smape": SymmetricMeanAbsolutePercentageError(),
        }
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


def process_store_item(
    store: int, item: int, gpu_id: int, df: pd.DataFrame, args
):
    """
    Process a single (store, item) pair on a given GPU *sequentially*.
    """
    try:
        # Prepare time series
        ts_df = prepare_store_item_series(df, store, item)
        if ts_df.empty:
            logger.warning(f"No data for store {store}, item {item}")
            return None

        ts, train_ts, val_ts = get_train_val_data(
            ts_df,
            store,
            item,
            args.split_point,
            args.min_train_data_points,
        )

        if train_ts is None or val_ts is None or len(val_ts) == 0:
            logger.warning(
                f"Skipping (store {store}, item {item}) due to insufficient data."
            )
            return None

        model_name = args.model
        model_dir = args.model_dir.resolve() / f"{model_name}_{store}_{item}"

        # Build trainer kwargs targeting the chosen GPU
        model_kwargs = generate_torch_kwargs(gpu_id, model_dir)

        model = NBEATSModel(
            input_chunk_length=30,
            output_chunk_length=1,
            generic_architecture=True,
            num_stacks=10,
            num_blocks=1,
            num_layers=4,
            layer_widths=512,
            n_epochs=10,
            nr_epochs_val_period=1,
            batch_size=800,
            random_state=42,
            model_name=model_name,
            save_checkpoints=True,
            force_reset=True,
            **model_kwargs,
        )

        # Train your model
        logger.info(
            f"Fitting NBEATS for store {store}, item {item} on GPU {gpu_id}..."
        )
        model.fit(train_ts)

        # Log some info about the data
        logger.debug(f"Store {store}, Item {item}:")
        logger.debug(f"  Train length: {len(train_ts)}")
        logger.debug(f"  Val length: {len(val_ts)}")
        logger.debug(f"  Val start: {val_ts.start_time()}")
        logger.debug(f"  Val end: {val_ts.end_time()}")

        # One-shot forecast over the validation window
        forecast = model.predict(len(val_ts))

        metrics = calculate_metrics(train_ts, val_ts, forecast)

        logger.debug(f"  Forecast length: {len(forecast)}")
        logger.debug(f"  Forecast start: {forecast.start_time()}")
        logger.debug(f"  Forecast end: {forecast.end_time()}")

        # Check for alignment issues
        if len(forecast) != len(val_ts):
            logger.warning(
                f"Length mismatch for (store {store}, item {item}): "
                f"forecast={len(forecast)}, val={len(val_ts)}"
            )

        # Return the result as a dictionary
        return {
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

    except Exception as e:
        logger.error(
            f"ERROR processing (store {store}, item {item}) on GPU {gpu_id}: {e}"
        )
        import traceback

        logger.error(traceback.format_exc())
        return None


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
        "--gpu",
        type=int,
        default=0,
        help=(
            "Optional: If set, force all models onto this single GPU ID. "
            "If left as 0 and multiple GPUs are available, GPUs are used in round-robin."
        ),
    )
    parser.add_argument(
        "--metrics_fn",
        type=Path,
        default="",
        help="Path to metrics output file (relative to project root)",
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
    args = parse_args()
    data_fn = Path(args.data_fn).resolve()
    output_metrics_fn = Path(args.metrics_fn).resolve()
    log_fn = args.log_fn.resolve()

    # Set up logging
    logger = setup_logging(log_fn, args.log_level)
    logger.info(f"Log fn: {log_fn}")

    # List all visible devices
    num_gpus = 0
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("No GPUs found! Models will run on CPU/auto.")

    try:
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
        unique_combinations = unique_combinations.head(10)
        logger.info(f"Found {len(unique_combinations)} unique combinations")

        # Prepare metrics collection
        results = []

        if num_gpus == 0:
            # CPU-only: just iterate sequentially
            logger.info("Running sequentially on CPU/auto.")
            gpu_ids = [None] * len(
                unique_combinations
            )  # not used, but kept for API
        else:
            logger.info(
                f"Using {num_gpus} GPU(s). Assigning (store, item) to GPUs in round-robin."
            )

        # Sequential loop, GPU chosen per (store, item)
        for idx, row in tqdm(
            enumerate(unique_combinations.itertuples(index=False)),
            total=len(unique_combinations),
        ):
            store = row.store
            item = row.item

            if num_gpus == 0:
                gpu_id = 0  # ignored in generate_torch_kwargs (will fall back to auto/CPU)
            else:
                # If args.gpu is set explicitly, use that for all; else round-robin
                if args.gpu is not None and args.gpu >= 0:
                    gpu_id = args.gpu
                else:
                    gpu_id = idx % num_gpus

            res = process_store_item(store, item, gpu_id, df, args)
            if res is not None:
                results.append(res)

        logger.info(
            "All (store, item) combinations processed. Consolidating results..."
        )

        if not results:
            logger.warning(
                "No results were generated. Check logs for per-series errors."
            )
            metrics_df = pd.DataFrame(
                columns=[
                    "Model",
                    "Store",
                    "Item",
                    "RMSE",
                    "RMSSE",
                    "MAE",
                    "MASE",
                    "SMAPE",
                    "OPE",
                ]
            )
        else:
            metrics_df = pd.DataFrame(results)

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
