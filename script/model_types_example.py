#!/usr/bin/env python3
"""
Model Types Example Script

This script demonstrates the new FF_MODEL_TYPES and SEQ_MODEL_TYPES lists
and shows how to use them for training different model families.

Usage:
    python script/model_types_example.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import (
    ModelType, 
    SEQUENCE_MODEL, 
    FF_MODEL_TYPES, 
    SEQ_MODEL_TYPES
)
from src.model_utils import train_model_unified


def main():
    print("=" * 60)
    print("MODEL TYPES DEMONSTRATION")
    print("=" * 60)
    
    # Configuration
    store_cluster = 17
    item_cluster = 15
    label_cols = ["unit_sales"]
    y_to_log_features = ["unit_sales"]
    epochs = 5  # Short for demo
    
    # Directories
    model_dir = Path("./output/models/model_types_demo/")
    model_logger_dir = Path("./output/logs/model_types_demo/")
    
    print("\n1. FEEDFORWARD MODEL TYPES")
    print("-" * 40)
    print("Available feedforward models:")
    for i, model_type in enumerate(FF_MODEL_TYPES, 1):
        print(f"  {i}. {model_type.value} (enum: {model_type})")
    
    print(f"\nTotal feedforward models: {len(FF_MODEL_TYPES)}")
    
    print("\n2. SEQUENCE MODEL TYPES")
    print("-" * 40)
    print("Available sequence models:")
    for i, model_type in enumerate(SEQ_MODEL_TYPES, 1):
        print(f"  {i}. {model_type}")
    
    print(f"\nTotal sequence models: {len(SEQ_MODEL_TYPES)}")
    
    print("\n3. TRAINING EXAMPLES")
    print("-" * 40)
    
    # Example 1: Train a feedforward model
    print("\nExample 1: Training a feedforward model")
    feedforward_dataloader_dir = Path("./output/data/dataloader_2014_2015_top_53_store_2000_item/")
    
    try:
        # Use the first feedforward model type
        selected_ff_model = FF_MODEL_TYPES[0]  # ModelType.SHALLOW_NN
        print(f"Selected feedforward model: {selected_ff_model.value}")
        
        train_model_unified(
            model_dir=model_dir,
            dataloader_dir=feedforward_dataloader_dir,
            model_logger_dir=model_logger_dir,
            model_type=selected_ff_model,  # ModelType enum
            model_family="feedforward",
            label_cols=label_cols,
            y_to_log_features=y_to_log_features,
            store_cluster=store_cluster,
            item_cluster=item_cluster,
            lr=3e-4,
            epochs=epochs,
            hidden_dim=64,
            dropout=0.2,
            log_level="INFO",
        )
        print(f"✅ Successfully trained {selected_ff_model.value}!")
        
    except Exception as e:
        print(f"❌ Feedforward training failed: {e}")
    
    # Example 2: Train a sequence model
    print("\nExample 2: Training a sequence model")
    sequence_dataloader_dir = Path("./output/data/dataloader_seq_model_2014_2015_top_53_store_2000_item/")
    
    try:
        # Use the first sequence model type
        selected_seq_model = SEQ_MODEL_TYPES[0]  # SEQUENCE_MODEL.TFT
        print(f"Selected sequence model: {selected_seq_model}")
        
        train_model_unified(
            model_dir=model_dir,
            dataloader_dir=sequence_dataloader_dir,
            model_logger_dir=model_logger_dir,
            model_type=selected_seq_model,  # String
            model_family="sequence",
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
            log_level="INFO",
        )
        print(f"✅ Successfully trained {selected_seq_model}!")
        
    except Exception as e:
        print(f"❌ Sequence training failed: {e}")
    
    print("\n4. USAGE PATTERNS")
    print("-" * 40)
    
    print("\n# Iterate over all feedforward models:")
    print("for model_type in FF_MODEL_TYPES:")
    print("    print(f'Training {model_type.value}...')")
    print("    train_model_unified(model_type=model_type, model_family='feedforward', ...)")
    
    print("\n# Iterate over all sequence models:")
    print("for model_type in SEQ_MODEL_TYPES:")
    print("    print(f'Training {model_type}...')")
    print("    train_model_unified(model_type=model_type, model_family='sequence', ...)")
    
    print("\n# Check if a model type is available:")
    print("if ModelType.SHALLOW_NN in FF_MODEL_TYPES:")
    print("    print('ShallowNN is available')")
    
    print("if SEQUENCE_MODEL.TFT in SEQ_MODEL_TYPES:")
    print("    print('TFT is available')")
    
    print("\n5. KEY DIFFERENCES")
    print("-" * 40)
    print("FF_MODEL_TYPES:")
    print("  - Contains ModelType enums")
    print("  - Used with model_family='feedforward'")
    print("  - Example: ModelType.SHALLOW_NN")
    
    print("\nSEQ_MODEL_TYPES:")
    print("  - Contains string constants")
    print("  - Used with model_family='sequence'")
    print("  - Example: SEQUENCE_MODEL.TFT")
    
    print("\n" + "=" * 60)
    print("MIGRATION COMPLETE!")
    print("=" * 60)
    print("✅ MODEL_TYPES → FF_MODEL_TYPES")
    print("✅ Added SEQ_MODEL_TYPES")
    print("✅ Updated all scripts and notebooks")
    print("✅ Consistent naming across codebase")


if __name__ == "__main__":
    main()
