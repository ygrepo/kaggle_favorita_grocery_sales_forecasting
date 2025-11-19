#!/usr/bin/env python3
"""
Training script for the Favorita Grocery Sales Forecasting model.

Supports training multiple models per (store, item):
--models "NBEATS,TFT" or --models "NBEATS"
"""

import sys
import argparse
from pathlib import Path
from enum import Enum
from typing import Dict, Optional, List


import traceback

import pandas as pd
from tqdm import tqdm

from darts.models import (
    NBEATSModel,
    TFTModel,
    TSMixerModel,
    BlockRNNModel,
    TCNModel,
    TiDEModel,
)
from darts.models.forecasting.forecasting_model import ForecastingModel
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
    BLOCK_RNN = "BLOCK_RNN"
    TCN = "TCN"
    TIDE = "TIDE"


def parse_models_arg(models_string: str) -> List[ModelType]:
    """
    Convert --models "NBEATS,TFT" into [ModelType.NBEATS, ModelType.TFT].
    """
    try:
        names = [m.strip().upper() for m in models_string.split(",")]
        return [ModelType(name) for name in names]
    except Exception:
        raise ValueError(f"Invalid --models argument: {models_string}")


def generate_torch_kwargs(gpu_id: Optional[int], working_dir: Path) -> Dict:
    """Return trainer kwargs + torch_metrics for a specific GPU."""
    early_stopper = EarlyStopping(
        monitor="train_smape",
        min_delta=0.001,
        patience=6,
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


def create_model(
    model_type: ModelType,
    torch_kwargs: Dict,
) -> ForecastingModel:
    """
    Factory to create a Darts model instance.
    """

    base = dict(
        input_chunk_length=30,
        output_chunk_length=1,
        n_epochs=10,
        batch_size=800,
        random_state=42,
        save_checkpoints=True,
        force_reset=True,
        **torch_kwargs,
    )

    # -------------------------
    # NBEATS
    # -------------------------
    if model_type == ModelType.NBEATS:
        return NBEATSModel(
            generic_architecture=True,
            num_stacks=10,
            num_blocks=1,
            num_layers=4,
            layer_widths=512,
            **base,
        )

    # -------------------------
    # TFT (Temporal Fusion Transformer)
    # -------------------------
    elif model_type == ModelType.TFT:
        return TFTModel(
            hidden_size=64,
            lstm_layers=1,
            dropout=0.1,
            num_attention_heads=4,
            add_relative_index=True,
            **base,
        )

    # -------------------------
    # TS-Mixer
    # -------------------------
    elif model_type == ModelType.TSMIXER:
        return TSMixerModel(
            hidden_size=64,
            dropout=0.1,
            **base,
        )

    # -------------------------
    # Block RNN (LSTM/GRU)
    # -------------------------
    elif model_type == ModelType.BLOCK_RNN:
        return BlockRNNModel(
            model="LSTM",  # Could be "RNN" / "LSTM" / "GRU"
            hidden_dim=64,  # Good default
            n_rnn_layers=2,  # Stacked layers
            dropout=0.1,
            **base,
        )

    # -------------------------
    # Temporal Convolutional Network (TCN)
    # -------------------------
    elif model_type == ModelType.TCN:
        return TCNModel(
            kernel_size=3,  # receptive field
            num_filters=64,  # number of convolutional filters (capacity)
            dilation_base=2,  # exponential dilation growth
            weight_norm=True,  # stable training
            dropout=0.1,
            **base,
        )

    # -------------------------
    # TiDE Model (MLP-based global model)
    # -------------------------
    elif model_type == ModelType.TIDE:
        # only pass known-good kwargs for your Darts version
        return TiDEModel(
            hidden_size=64,
            dropout=0.1,
            **base,
        )

    raise ValueError(f"Unsupported model: {model_type}")


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
) -> List[dict]:
    """Train ALL requested models for a single store-item pair."""
    try:
        ts_df = prepare_store_item_series(df, store, item)
        if ts_df.empty:
            return []

        _, train_ts, val_ts = get_train_val_data(
            ts_df, store, item, args.split_point, args.min_train_data_points
        )
        if train_ts is None or val_ts is None or len(val_ts) == 0:
            return []

        rows = []

        # train each model requested
        for mtype in model_types:
            model_dir = (
                args.model_dir.resolve()
                / mtype.value
                / f"store_{store}_item_{item}"
            )
            model_dir.mkdir(parents=True, exist_ok=True)
            torch_kwargs = generate_torch_kwargs(gpu_id, model_dir)

            model = create_model(mtype, torch_kwargs)

            logger.info(
                f"[GPU {gpu_id}] Training {mtype.value} for store={store}, item={item}"
            )
            model.fit(train_ts)

            forecast = model.predict(len(val_ts))
            metrics = calculate_metrics(train_ts, val_ts, forecast)

            rows.append(
                {
                    "Model": mtype.value,
                    "Store": store,
                    "Item": item,
                    "RMSE": metrics["rmse"],
                    "MAE": metrics["mae"],
                    "SMAPE": metrics["smape"],
                    "OPE": metrics["ope"],
                    "RMSSE": metrics["rmsse"],
                    "MASE": metrics["mase"],
                }
            )

        return rows

    except Exception as e:
        logger.error(f"ERROR for store={store}, item={item}: {e}")
        logger.error(traceback.format_exc())
        return []


# ---------------------------------------------------------------------
# Argparse + main
# ---------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Favorita forecasting benchmark"
    )

    parser.add_argument("--data_fn", type=Path, default="")
    parser.add_argument("--model_dir", type=Path, default="")
    parser.add_argument(
        "--models",
        type=str,
        default="NBEATS",
        help="Comma-separated model list: 'NBEATS,TFT,TSMIXER'",
    )
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--metrics_fn", type=Path, default="")
    parser.add_argument("--split_point", type=float, default=0.8)
    parser.add_argument("--min_train_data_points", type=int, default=15)
    parser.add_argument(
        "--N", type=int, default=0, help="Limit to first N combinations"
    )
    parser.add_argument("--log_fn", type=Path, default="")
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse model list
    model_types = parse_models_arg(args.models)

    # Setup logging
    logger = setup_logging(args.log_fn, args.log_level)
    logger.info(f"Training models: {[m.value for m in model_types]}")

    # GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        num_gpus = 0
        logger.warning("No GPUs detected — running on CPU")

    df = load_raw_data(args.data_fn)
    logger.info(df.head())
    unique_combinations = df[["store", "item"]].drop_duplicates()
    if args.N > 0:
        logger.info(f"Limiting to first {args.N} combinations")
        unique_combinations = unique_combinations.head(args.N)
    results = []

    for idx, (store, item) in tqdm(
        enumerate(
            unique_combinations[["store", "item"]].itertuples(
                index=False, name=None
            )
        ),
        total=len(unique_combinations),
    ):

        # GPU assignment
        if num_gpus == 0:
            gpu_id = None
        elif args.gpu >= 0:
            gpu_id = args.gpu  # force one GPU
        else:
            gpu_id = idx % num_gpus  # round-robin GPUs

        rows = process_store_item(store, item, gpu_id, df, args, model_types)
        results.extend(rows)

    metrics_df = pd.DataFrame(results)
    save_csv_or_parquet(metrics_df, args.metrics_fn)

    logger.info("Benchmark complete.")


if __name__ == "__main__":
    main()
