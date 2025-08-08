#!/bin/bash

# Learning Rate Finder Script for Favorita Grocery Sales Forecasting
# This script runs the learning rate finder to help determine optimal learning rates

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Default configuration
DATALOADER_DIR="${PROJECT_ROOT}/output/data/dataloader_2014_2015_top_53_store_2000_item/"
OUTPUT_DIR="${PROJECT_ROOT}/output/lr_finder/"
LOG_DIR="${PROJECT_ROOT}/output/logs"

# Default parameters
STORE_CLUSTER=17
ITEM_CLUSTER=15
MODEL_TYPE="ShallowNN"
START_LR=1e-7
END_LR=1.0
NUM_ITER=100
LOG_LEVEL="INFO"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataloader_dir)
            DATALOADER_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --store_cluster)
            STORE_CLUSTER="$2"
            shift 2
            ;;
        --item_cluster)
            ITEM_CLUSTER="$2"
            shift 2
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --start_lr)
            START_LR="$2"
            shift 2
            ;;
        --end_lr)
            END_LR="$2"
            shift 2
            ;;
        --num_iter)
            NUM_ITER="$2"
            shift 2
            ;;
        --log_level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataloader_dir DIR    Directory containing dataloaders (default: $DATALOADER_DIR)"
            echo "  --output_dir DIR        Output directory for results (default: $OUTPUT_DIR)"
            echo "  --store_cluster NUM     Store cluster ID (default: $STORE_CLUSTER)"
            echo "  --item_cluster NUM      Item cluster ID (default: $ITEM_CLUSTER)"
            echo "  --model_type TYPE       Model type: ShallowNN, TwoLayerNN, ResidualMLP (default: $MODEL_TYPE)"
            echo "  --start_lr FLOAT        Starting learning rate (default: $START_LR)"
            echo "  --end_lr FLOAT          Ending learning rate (default: $END_LR)"
            echo "  --num_iter NUM          Number of iterations (default: $NUM_ITER)"
            echo "  --log_level LEVEL       Log level: DEBUG, INFO, WARNING, ERROR (default: $LOG_LEVEL)"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --store_cluster 17 --item_cluster 15"
            echo "  $0 --model_type TwoLayerNN --start_lr 1e-6 --end_lr 0.1"
            echo "  $0 --store_cluster 5 --item_cluster 10 --num_iter 200"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create necessary directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Set up logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/lr_finder_${STORE_CLUSTER}_${ITEM_CLUSTER}_${MODEL_TYPE}_${TIMESTAMP}.log"

echo "Starting Learning Rate Finder..." | tee "$LOG_FILE"
echo "Configuration:" | tee -a "$LOG_FILE"
echo "  Dataloader directory: $DATALOADER_DIR" | tee -a "$LOG_FILE"
echo "  Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "  Store cluster: $STORE_CLUSTER" | tee -a "$LOG_FILE"
echo "  Item cluster: $ITEM_CLUSTER" | tee -a "$LOG_FILE"
echo "  Model type: $MODEL_TYPE" | tee -a "$LOG_FILE"
echo "  LR range: $START_LR to $END_LR" | tee -a "$LOG_FILE"
echo "  Iterations: $NUM_ITER" | tee -a "$LOG_FILE"
echo "  Log level: $LOG_LEVEL" | tee -a "$LOG_FILE"
echo "  Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check if dataloader directory exists
if [ ! -d "$DATALOADER_DIR" ]; then
    echo "Error: Dataloader directory does not exist: $DATALOADER_DIR" | tee -a "$LOG_FILE"
    echo "Please run the dataloader creation script first." | tee -a "$LOG_FILE"
    exit 1
fi

# Check if specific dataloader files exist
TRAIN_FILE="${DATALOADER_DIR}/train_loader_${STORE_CLUSTER}_${ITEM_CLUSTER}.pt"
VAL_FILE="${DATALOADER_DIR}/val_loader_${STORE_CLUSTER}_${ITEM_CLUSTER}.pt"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "Error: Training dataloader not found: $TRAIN_FILE" | tee -a "$LOG_FILE"
    echo "Available dataloaders:" | tee -a "$LOG_FILE"
    ls -la "$DATALOADER_DIR"/*.pt 2>/dev/null | head -10 | tee -a "$LOG_FILE" || echo "No .pt files found" | tee -a "$LOG_FILE"
    exit 1
fi

if [ ! -f "$VAL_FILE" ]; then
    echo "Error: Validation dataloader not found: $VAL_FILE" | tee -a "$LOG_FILE"
    exit 1
fi

echo "Found required dataloader files:" | tee -a "$LOG_FILE"
echo "  Train: $TRAIN_FILE" | tee -a "$LOG_FILE"
echo "  Val: $VAL_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Show GPU info if available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:" | tee -a "$LOG_FILE"
    nvidia-smi | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
fi

# Run the learning rate finder
echo "Running Learning Rate Finder..." | tee -a "$LOG_FILE"
python "${SCRIPT_DIR}/lr_finder.py" \
    --dataloader_dir "$DATALOADER_DIR" \
    --store_cluster "$STORE_CLUSTER" \
    --item_cluster "$ITEM_CLUSTER" \
    --model_type "$MODEL_TYPE" \
    --output_dir "$OUTPUT_DIR" \
    --start_lr "$START_LR" \
    --end_lr "$END_LR" \
    --num_iter "$NUM_ITER" \
    --log_level "$LOG_LEVEL" 2>&1 | tee -a "$LOG_FILE"

# Check if the run was successful
if [ $? -eq 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "Learning Rate Finder completed successfully!" | tee -a "$LOG_FILE"
    echo "Results saved in: $OUTPUT_DIR" | tee -a "$LOG_FILE"
    echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
    
    # Show the generated files
    echo "" | tee -a "$LOG_FILE"
    echo "Generated files:" | tee -a "$LOG_FILE"
    ls -la "${OUTPUT_DIR}/"*"${STORE_CLUSTER}_${ITEM_CLUSTER}_${MODEL_TYPE}"* 2>/dev/null | tee -a "$LOG_FILE" || echo "No output files found" | tee -a "$LOG_FILE"
else
    echo "" | tee -a "$LOG_FILE"
    echo "Learning Rate Finder failed. Check the log file for details: $LOG_FILE" | tee -a "$LOG_FILE"
    exit 1
fi
