#!/usr/bin/env python3
"""
Unified Training Example Script

This script demonstrates the updated train_model_unified function that:
1. Uses consistent CSV logging for both feedforward and sequence models
2. Tracks the same metrics (MAE, RMSE, percent MAV) for both model types
3. Removes separate history collection (everything goes to CSV logs)

Usage:
    python script/unified_training_example.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model_utils import train_model_unified
from src.model import MODEL_TYPE


def main():
    # Configuration
    store_cluster = 17
    item_cluster = 15
    label_cols = ["unit_sales"]
    y_to_log_features = ["unit_sales"]
    epochs = 10  # Reduced for demo
    log_level = "INFO"

    # Directories
    model_dir = Path("./output/models/unified_demo/")
    model_logger_dir = Path("./output/logs/unified_demo/")

    print("=" * 60)
    print("UNIFIED TRAINING DEMONSTRATION")
    print("=" * 60)
    print(f"Store cluster: {store_cluster}, Item cluster: {item_cluster}")
    print(f"Epochs: {epochs}")
    print(f"Model directory: {model_dir}")
    print(f"Logger directory: {model_logger_dir}")
    print()

    # 1. Train a feedforward model
    print("1. Training Feedforward Model (TwoLayerNN)")
    print("-" * 40)

    feedforward_dataloader_dir = Path(
        "./output/data/dataloader_2014_2015_top_53_store_2000_item/"
    )

    try:
        train_model_unified(
            model_dir=model_dir,
            dataloader_dir=feedforward_dataloader_dir,
            model_logger_dir=model_logger_dir,
            model_type=MODEL_TYPE.TWO_LAYER_NN,  # Unified model type
            label_cols=label_cols,
            y_to_log_features=y_to_log_features,
            store_cluster=store_cluster,
            item_cluster=item_cluster,
            lr=3e-4,
            epochs=epochs,
            h1=64,
            h2=32,
            dropout=0.4,
            log_level=log_level,
        )
        print("✅ Feedforward model training completed!")
    except Exception as e:
        print(f"❌ Feedforward model training failed: {e}")

    print()

    # 2. Train a sequence model
    print("2. Training Sequence Model (TFT)")
    print("-" * 40)

    sequence_dataloader_dir = Path(
        "./output/data/dataloader_seq_model_2014_2015_top_53_store_2000_item/"
    )

    try:
        train_model_unified(
            model_dir=model_dir,
            dataloader_dir=sequence_dataloader_dir,
            model_logger_dir=model_logger_dir,
            model_type=MODEL_TYPE.TFT,  # Unified model type
            label_cols=label_cols,
            y_to_log_features=y_to_log_features,
            store_cluster=store_cluster,
            item_cluster=item_cluster,
            lr=1e-3,
            epochs=epochs,
            hidden_dim=32,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=16,
            log_level=log_level,
        )
        print("✅ Sequence model training completed!")
    except Exception as e:
        print(f"❌ Sequence model training failed: {e}")

    print()

    # 3. Show the results
    print("3. Training Results")
    print("-" * 40)

    # List the generated CSV log files
    csv_files = list(model_logger_dir.glob("**/*.csv"))

    if csv_files:
        print("Generated CSV log files:")
        for csv_file in csv_files:
            print(f"  📄 {csv_file.relative_to(model_logger_dir)}")

        print("\nThese CSV files contain:")
        print("  • Training and validation loss per epoch")
        print("  • Learning rate changes")
        print("  • MAE and RMSE metrics")
        print("  • Percent MAV (Mean Absolute Value) metrics")
        print("  • All metrics are consistent between feedforward and sequence models")
    else:
        print("No CSV log files found.")

    print()
    print("=" * 60)
    print("KEY IMPROVEMENTS")
    print("=" * 60)
    print("✅ Consistent CSV logging for both model types")
    print("✅ Same metrics tracked: MAE, RMSE, percent MAV")
    print("✅ No separate history collection needed")
    print("✅ Unified training interface")
    print("✅ Sequence models wrapped for metric compatibility")
    print("✅ Checkpoints saved for both model types")
    print("✅ Early stopping and learning rate monitoring")


if __name__ == "__main__":
    main()
