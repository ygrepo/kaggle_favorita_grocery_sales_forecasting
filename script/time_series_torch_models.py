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
from enum import Enum
from typing import Dict, Optional

try:
    import traceback
except ImportError:
    pass

import pandas as pd
from tqdm import tqdm

from darts.models import NBEATSModel, TFTModel, TSMixerModel
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


# ---------------------------------------------------------------------
# Enum + Factory
# ---------------------------------------------------------------------


class ModelType(str, Enum):
    NBEATS = "NBEATS"
    TFT = "TFT"
    TSMIXER = "TSMIXER"


def generate_torch_kwargs(gpu_id: Optional[int], working_dir: Path) -> Dict:
    """
    Build pl_trainer_kwargs and torch_metrics for Darts' TorchForecastingModel.

    If gpu_id is None or CUDA is unavailable, falls back to CPU/auto.
    """
    early_stopper = EarlyStopping(
        monitor="train_smape",  # consider "val_loss" if you want validation-based stopping
        min_delta=0.001,
        patience=6,
        verbose=True,
        mode="min",
    )
    callbacks = [
        early_stopper,
        TFMProgressBar(enable_train_bar_only=True),
    ]

    if (
        gpu_id is not None
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 0
    ):
        accelerator = "gpu"
        devices = [gpu_id]
        logger.debug(f"Using GPU {gpu_id} for this model.")
    else:
        accelerator = "auto"
        devices = "auto"
        logger.debug("No specific CUDA device; using CPU/auto.")

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


def create_model(
    model_type: ModelType,
    model_name: str,
    torch_kwargs: Dict,
) -> NBEATSModel:
    """
    Factory to create a Darts TorchForecastingModel (NBEATS, TFT, TSMixer)
    with a shared set of base hyperparameters and model-specific extras.
    """

    # Base kwargs shared by all three models
    base_kwargs = dict(
        input_chunk_length=30,
        output_chunk_length=1,
        n_epochs=10,
        batch_size=800,
        random_state=42,
        model_name=model_name,
        save_checkpoints=True,
        force_reset=True,
        **torch_kwargs,
    )

    if model_type == ModelType.NBEATS:
        return NBEATSModel(
            generic_architecture=True,
            num_stacks=10,
            num_blocks=1,
            num_layers=4,
            layer_widths=512,
            **base_kwargs,
        )

    elif model_type == ModelType.TFT:
        # You can tune these hyperparameters further
        return TFTModel(
            hidden_size=64,
            lstm_layers=1,
            num_attention_heads=4,
            dropout=0.1,
            add_relative_index=True,
            **base_kwargs,
        )

    elif model_type == ModelType.TSMIXER:
        return TSMixerModel(
            hidden_size=64,
            dropout=0.1,
            **base_kwargs,
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# ---------------------------------------------------------------------
# Core per-(store, item) processing
# ---------------------------------------------------------------------


def process_store_item(
    store: int,
    item: int,
    gpu_id: Optional[int],
    df: pd.DataFrame,
    args,
    model_type: ModelType,
):
    """
    Process a single (store, item) pair on a given GPU sequentially.
    """
    try:
        # Prepare time series
        ts_df = prepare_store_item_series(df, store, item)
        if ts_df.empty:
            logger.warning(f"No data for store {store}, item {item}")
            return None

        _, train_ts, val_ts = get_train_val_data(
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
        torch_kwargs = generate_torch_kwargs(gpu_id, model_dir)

        # Create the model via factory
        model = create_model(
            model_type=model_type,
            model_name=model_name,
            torch_kwargs=torch_kwargs,
        )

        logger.info(
            f"Fitting {model_type.value} for store {store}, item {item} on GPU {gpu_id}..."
        )
        model.fit(train_ts)

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

        if len(forecast) != len(val_ts):
            logger.warning(
                f"Length mismatch for (store {store}, item {item}): "
                f"forecast={len(forecast)}, val={len(val_ts)}"
            )
        return {
            "Model": model_name,
            "Store": store,
            "Item": item,
            "RMSE": metrics["rmse"],
            "MAE": metrics["mae"],
            "SMAPE": metrics["smape"],
            "OPE": metrics["ope"],
            "RMSSE": metrics["rmsse"],
            "MASE": metrics["mase"],
        }

    except Exception as e:
        logger.error(
            f"ERROR processing (store {store}, item {item}) on GPU {gpu_id}: {e}"
        )
        logger.error(traceback.format_exc())
        return None


# ---------------------------------------------------------------------
# Argparse + main
# ---------------------------------------------------------------------


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
        default=ModelType.NBEATS.value,
        choices=[m.value for m in ModelType],
        help="Model to train (NBEATS, TFT, TSMIXER)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help=(
            "If >=0, force all models onto this single GPU ID. "
            "If <0 and multiple GPUs are available, GPUs are used in round-robin."
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

    # Model type from Enum
    model_type = ModelType(args.model)

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
        logger.info(f"  Model type: {model_type.value}")

        # Load raw data
        logger.info("Loading raw data...")
        df = load_raw_data(data_fn)

        # Get unique store-item pairs
        logger.info("Finding unique store-item combinations...")
        unique_combinations = df[["store", "item"]].drop_duplicates()
        unique_combinations = unique_combinations.head(10)
        logger.info(f"Found {len(unique_combinations)} unique combinations")

        results = []

        if num_gpus == 0:
            logger.info("Running sequentially on CPU/auto.")
        else:
            logger.info(
                f"Using {num_gpus} GPU(s). "
                "Assigning (store, item) to GPUs in round-robin, unless --gpu is set."
            )

        for idx, row in tqdm(
            enumerate(unique_combinations.itertuples(index=False)),
            total=len(unique_combinations),
        ):
            store = row.store
            item = row.item

            if num_gpus == 0:
                gpu_id = None
            else:
                if args.gpu is not None and args.gpu >= 0:
                    gpu_id = args.gpu
                else:
                    gpu_id = idx % num_gpus

            res = process_store_item(store, item, gpu_id, df, args, model_type)
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
                    "MAE",
                    "SMAPE",
                    "OPE",
                    "RMSSE",
                    "MASE",
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
