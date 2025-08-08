#!/usr/bin/env python3
"""
Training script for sequence models using pytorch_forecasting

This script demonstrates how to train different sequence models (TFT, NBEATS, DeepAR, LSTM)
using the updated train_model_unified function.

Usage:
    python script/train_sequence_models.py --model_type TFT --store_cluster 17 --item_cluster 15
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model_utils import train_model_unified
from src.model import MODEL_TYPE, SEQ_MODEL_TYPES


def main():
    parser = argparse.ArgumentParser(description="Train sequence models")

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=SEQ_MODEL_TYPES,
        help="Sequence model type",
    )
    parser.add_argument(
        "--store_cluster", type=int, required=True, help="Store cluster ID"
    )
    parser.add_argument(
        "--item_cluster", type=int, required=True, help="Item cluster ID"
    )
    parser.add_argument(
        "--dataloader_dir",
        type=str,
        default="./output/data/dataloader_seq_model_2014_2015_top_53_store_2000_item/",
        help="Directory containing sequence dataloaders",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./output/models/sequence_models/",
        help="Directory to save models",
    )
    parser.add_argument(
        "--model_logger_dir",
        type=str,
        default="./output/logs/sequence_models/",
        help="Directory for model logs",
    )
    parser.add_argument(
        "--history_dir",
        type=str,
        default="./output/history/sequence_models/",
        help="Directory for training history",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--hidden_dim", type=int, default=32, help="Hidden dimension size"
    )
    parser.add_argument(
        "--attention_head_size",
        type=int,
        default=4,
        help="Attention head size (for TFT)",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Convert paths
    dataloader_dir = Path(args.dataloader_dir)
    model_dir = Path(args.model_dir)
    model_logger_dir = Path(args.model_logger_dir)
    history_dir = Path(args.history_dir)

    # Default label columns (adjust as needed)
    label_cols = ["unit_sales"]
    y_to_log_features = ["unit_sales"]

    print(
        f"Training {args.model_type} model for store_cluster={args.store_cluster}, item_cluster={args.item_cluster}"
    )
    print(f"Dataloader directory: {dataloader_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Epochs: {args.epochs}, Learning rate: {args.lr}")

    try:
        train_model_unified(
            model_dir=model_dir,
            dataloader_dir=dataloader_dir,
            model_logger_dir=model_logger_dir,
            model_type=MODEL_TYPE(args.model_type),  # Convert string to enum
            label_cols=label_cols,
            y_to_log_features=y_to_log_features,
            store_cluster=args.store_cluster,
            item_cluster=args.item_cluster,
            history_dir=history_dir,
            lr=args.lr,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            attention_head_size=args.attention_head_size,
            dropout=args.dropout,
            log_level=args.log_level,
        )
        print(f"Successfully trained {args.model_type} model!")

    except Exception as e:
        print(f"Error training model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
