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
from darts import TimeSeries, concatenate
from tqdm import tqdm
from darts.models import NBEATSModel
from darts.utils.callbacks import TFMProgressBar

from torchmetrics import SymmetricMeanAbsolutePercentageError, MetricCollection
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping
import multiprocessing
from functools import partial

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


# Change the function signature to accept gpu_id
def generate_torch_kwargs(gpu_id: int, working_dir: Path) -> dict:
    # throughout training we'll monitor the validation loss for early stopping
    early_stopper = EarlyStopping(
        "val_smape",
        min_delta=0.001,
        patience=3,
        verbose=True,
    )
    callbacks = [
        early_stopper,
        TFMProgressBar(enable_train_bar_only=True),
    ]

    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = [gpu_id]
        logger.debug(f"Worker using GPU: {gpu_id}")
    else:
        accelerator = "auto"  # fallback to 'cpu' or 'auto'
        devices = "auto"
        logger.debug(f"Worker using GPU: {gpu_id}")

    metrics = MetricCollection(
        {
            "smape": SymmetricMeanAbsolutePercentageError(),
        }
    )

    # run torch models on CPU, and disable progress bars for all model stages except training.
    return {
        "pl_trainer_kwargs": {
            "accelerator": accelerator,
            "devices": devices,  # This will now be [0], or [1], etc.
            "callbacks": callbacks,
            "default_root_dir": str(working_dir),
        },
        "torch_metrics": metrics,
    }


def process_store_item(task_data, df, args):
    """
    Worker function to process a single store-item combination on a specific GPU.
    """
    store, item, gpu_id = task_data

    try:
        # Prepare time series
        ts_df = prepare_store_item_series(df, store, item)
        if ts_df.empty:
            logger.warning(f"No data for store {store}, item {item}")
            return None  # Signal to skip

        ts, train_ts, val_ts = get_train_val_data(
            ts_df,
            store,
            item,
            args.split_point,
            args.min_train_data_points,
        )

        model_name = args.model
        model_dir = args.model_dir.resolve() / f"{model_name}_{store}_{item}"

        # Pass the assigned gpu_id to the kwargs generator
        model_kwargs = generate_torch_kwargs(gpu_id, model_dir)

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
            **model_kwargs,
        )

        # Train your model
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
            verbose=False,  # Set to False to avoid flooding logs
        )
        forecast = concatenate(forecast)

        metrics = calculate_metrics(train_ts, val_ts, forecast)

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
        # Log errors from inside the worker
        logger.error(
            f"!! ERROR processing (store {store}, item {item}) on GPU {gpu_id}: {e}"
        )
        import traceback

        logger.error(traceback.format_exc())
        return None  # Signal failure


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
        help="The specific GPU ID to use for this script instance.",
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

        # --- START: NEW MULTIPROCESSING LOGIC ---

        # Determine number of GPUs to use
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            logger.warning("No GPUs found! Running sequentially on CPU.")
            num_gpus = 1  # We'll run 1 worker process on the CPU

        logger.info(f"Found {num_gpus} GPUs. Creating task list...")

        # Create a list of tasks. Each task is a tuple: (store, item, gpu_id)
        # We assign a-gpu_id in a round-robin fashion.
        tasks = []
        for i, row in enumerate(unique_combinations.itertuples(index=False)):
            tasks.append((row.store, row.item, i % num_gpus))

        logger.info(
            f"Distributing {len(tasks)} tasks across {num_gpus} workers."
        )

        # We must use 'spawn' for safety with CUDA
        multiprocessing.set_start_method("spawn", force=True)

        # Use functools.partial to "bake in" the static df and args
        # The pool will only send the changing 'task_data' to the worker
        worker_func = partial(process_store_item, df=df, args=args)

        results = []

        # Create the pool and run the tasks
        with multiprocessing.Pool(processes=num_gpus) as pool:
            # Use pool.imap_unordered for efficiency and tqdm for progress
            for result in tqdm(
                pool.imap_unordered(worker_func, tasks), total=len(tasks)
            ):
                if result is not None:  # Collect non-error results
                    results.append(result)

        logger.info("All tasks complete. Consolidating results...")

        if not results:
            logger.warning(
                "No results were generated. Check worker logs for errors."
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
