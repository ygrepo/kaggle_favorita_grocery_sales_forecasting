#!/bin/bash

# Set strict error handling
set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"


# Default configuration
DATALOADER_DIR="${PROJECT_ROOT}/output/data/dataloader_2014_2015_top_53_store_2000_item/"
MODEL_DIR="${PROJECT_ROOT}/output/model_2014_2015_top_53_store_2000_item/"
MODEL_LOGGER_DIR="${PROJECT_ROOT}/output/logs/model_logger_2014_2015_top_53_store_2000_item/"
HISTORY_DIR="${PROJECT_ROOT}/output/data/history_2014_2015_top_53_store_2000_item/"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

LOG_DIR="${PROJECT_ROOT}/output/logs"
WINDOW_SIZE=1
EPOCHS=10
LR=3e-4
NUM_WORKERS=15
PERSISTENT_WORKERS=false
ENABLE_PROGRESS_BAR=true
TRAIN_LOGGER=false
LOG_LEVEL="DEBUG"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataloader_dir) DATALOADER_DIR="$2"; shift 2 ;;
    --model_dir) MODEL_DIR="$2"; shift 2 ;;
    --model_logger_dir) MODEL_LOGGER_DIR="$2"; shift 2 ;;
    --history_dir) HISTORY_DIR="$2"; shift 2 ;;
    --window_size) WINDOW_SIZE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --persistent_workers) PERSISTENT_WORKERS="$2"; shift 2 ;;
    --enable_progress_bar) ENABLE_PROGRESS_BAR="$2"; shift 2 ;;
    --train_logger) TRAIN_LOGGER="$2"; shift 2 ;;
    --log_dir) LOG_DIR="$2"; shift 2 ;;
    --log_level) LOG_LEVEL="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# Create output directories if they don't exist
mkdir -p "$MODEL_DIR"
mkdir -p "$HISTORY_DIR"
mkdir -p "$LOG_DIR"

# Optional: add header if needed
#echo "model_name,store_cluster,item_cluster,train_mav,val_mav,best_train_avg_mae,best_val_avg_mae,best_train_avg_rmse,best_val_avg_rmse,best_train_avg_mae_percent_mav,best_val_avg_mae_percent_mav" > "$HISTORY_FN"

# Set up log file with timestamp
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

echo "Starting training at $(date)" | tee -a "$LOG_FILE"
echo "Project root: $PROJECT_ROOT" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"

# Run the training script
set +e  # Disable exit on error to handle the error message
echo "Starting training with the following configuration:" | tee -a "$LOG_FILE"
echo "  Dataloader directory: ${DATALOADER_DIR}" | tee -a "$LOG_FILE"
echo "  Model directory: ${MODEL_DIR}" | tee -a "$LOG_FILE"
echo "  Model logger directory: ${MODEL_LOGGER_DIR}" | tee -a "$LOG_FILE"
echo "  History directory: ${HISTORY_DIR}" | tee -a "$LOG_FILE"
echo "  Window size: ${WINDOW_SIZE}" | tee -a "$LOG_FILE"
echo "  Epochs: ${EPOCHS}" | tee -a "$LOG_FILE"
echo "  Learning rate: ${LR}" | tee -a "$LOG_FILE"
echo "  Num workers: ${NUM_WORKERS}" | tee -a "$LOG_FILE"
echo "  Persistent workers: ${PERSISTENT_WORKERS}" | tee -a "$LOG_FILE"
echo "  Enable progress bar: ${ENABLE_PROGRESS_BAR}" | tee -a "$LOG_FILE"
echo "  Train logger: ${TRAIN_LOGGER}" | tee -a "$LOG_FILE"
echo "  Log level: ${LOG_LEVEL}" | tee -a "$LOG_FILE"

nvidia-smi | tee -a "$LOG_FILE"


python "${SCRIPT_DIR}/training.py" \
  --dataloader_dir "$DATALOADER_DIR" \
  --model_dir "$MODEL_DIR" \
  --model_logger_dir "$MODEL_LOGGER_DIR" \
  --history_dir "$HISTORY_DIR" \
  --window_size "$WINDOW_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --num_workers "$NUM_WORKERS" \
  --persistent_workers "$PERSISTENT_WORKERS" \
  --enable_progress_bar "$ENABLE_PROGRESS_BAR" \
  --train_logger "$TRAIN_LOGGER" \
  --log_level "$LOG_LEVEL" 2>&1 | tee -a "$LOG_FILE"

# Check the exit status of the Python script
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully at $(date)" | tee -a "$LOG_FILE"
    exit 0
else
    echo "Error: Training failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
    echo "Check the log file for details: $LOG_FILE"
    exit $EXIT_CODE
fi
