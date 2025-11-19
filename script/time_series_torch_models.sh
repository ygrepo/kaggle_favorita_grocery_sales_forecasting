#!/bin/bash

# Set strict error handling
set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Default configuration
DATA_FN="${PROJECT_ROOT}/output/data/2013_2014_store_2000_item_cyc_features.parquet"
DATE=$(date +"%Y%m%d")
MODEL_DIR="${PROJECT_ROOT}/output/models"
mkdir -p "$MODEL_DIR"
#MODEL="NBEATS"
#MODEL="TFT"
MODELS="NBEATS,TFT,TSMIXER"
METRICS_DIR="${PROJECT_ROOT}/output/metrics/${MODEL}"
mkdir -p "$METRICS_DIR"
METRICS_FN="${METRICS_DIR}/${DATE}_2013_2014_store_2000_item_cyc_features_metrics.csv"
SPLIT_POINT=0.8
MIN_TRAIN_DATA_POINTS=15

LOG_DIR="${PROJECT_ROOT}/output/logs"
LOG_LEVEL="DEBUG"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_fn) DATA_FN="$2"; shift 2 ;;
    --model_dir) MODEL_DIR="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --metrics_fn) METRICS_FN="$2"; shift 2 ;;
    --metrics_dir) METRICS_DIR="$2"; shift 2 ;;
    --split_point) SPLIT_POINT="$2"; shift 2 ;;
    --min_train_data_points) MIN_TRAIN_DATA_POINTS="$2"; shift 2 ;;
    --log_dir) LOG_DIR="$2"; shift 2 ;;
    --log_level) LOG_LEVEL="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# Create output directories if they don't exist
mkdir -p "$LOG_DIR"
# Set up log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_benchmark_time_series_models.log"

echo "Starting script at $(date)" | tee -a "$LOG_FILE"

# Run the script
set +e  # Disable exit on error to handle the error message
echo "Starting script with the following configuration:" | tee -a "$LOG_FILE"
echo "Project root: $PROJECT_ROOT" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Log level: $LOG_LEVEL" | tee -a "$LOG_FILE"
echo "Data fn: $DATA_FN" | tee -a "$LOG_FILE"
echo "Metrics fn: $METRICS_FN" | tee -a "$LOG_FILE"
# Unset the restriction to see both GPUs
unset CUDA_VISIBLE_DEVICES
# OR set it to both
export CUDA_VISIBLE_DEVICES=0,1

python "${SCRIPT_DIR}/time_series_torch_models.py" \
  --data_fn "$DATA_FN" \
  --model_dir "$MODEL_DIR" \
  --models "$MODELS" \
  --metrics_fn "$METRICS_FN" \
  --split_point "$SPLIT_POINT" \
  --min_train_data_points "$MIN_TRAIN_DATA_POINTS" \
  --log_fn "$LOG_FILE" \
  --log_level "$LOG_LEVEL" 

# Check the exit status of the Python script
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "Script completed successfully at $(date)" | tee -a "$LOG_FILE"
    exit 0
else
    echo "Error: Script failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
    echo "Check the log file for details: $LOG_FILE"
    exit $EXIT_CODE
fi
