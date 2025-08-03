#!/bin/bash

# Set strict error handling
set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Default configuration
DATA_DIR="${PROJECT_ROOT}/output/data/sale_cyc_features_X_1_day_y_2014_2015_top_53_store_2000_item/"
DATALOADER_DIR="${PROJECT_ROOT}/output/data/dataloader_seq_model_2014_2015_top_53_store_2000_item/"

LOG_DIR="${PROJECT_ROOT}/output/logs"
LOG_LEVEL="DEBUG"
WINDOW_SIZE=1
MAX_ENCODER_LENGTH=30 # historical window size, e.g., 30 days
MAX_PREDICTION_LENGTH=1  # usually 1 for next-day forecasting
VAL_HORIZON=20  # Last N days for validation

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    --dataloader_dir) DATALOADER_DIR="$2"; shift 2 ;;
    --log_dir) LOG_DIR="$2"; shift 2 ;;
    --log_level) LOG_LEVEL="$2"; shift 2 ;;
    --window_size) WINDOW_SIZE="$2"; shift 2 ;;
    --max_encoder_length) MAX_ENCODER_LENGTH="$2"; shift 2 ;;
    --max_prediction_length) MAX_PREDICTION_LENGTH="$2"; shift 2 ;;
    --val_horizon) VAL_HORIZON="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# Create output directories if they don't exist
mkdir -p "$LOG_DIR"
mkdir -p "$DATALOADER_DIR"

# Set up log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/create_X_y_data_loaders_${TIMESTAMP}.log"

echo "Starting script at $(date)" | tee -a "$LOG_FILE"

# Run the script
set +e  # Disable exit on error to handle the error message
echo "Starting script with the following configuration:" | tee -a "$LOG_FILE"
echo "Project root: $PROJECT_ROOT" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Log level: $LOG_LEVEL" | tee -a "$LOG_FILE"
echo "  Data dir: ${DATA_DIR}" | tee -a "$LOG_FILE"
echo "  Dataloader dir: ${DATALOADER_DIR}" | tee -a "$LOG_FILE"
echo "  Window size: ${WINDOW_SIZE}" | tee -a "$LOG_FILE"
echo "  Max encoder length: ${MAX_ENCODER_LENGTH}" | tee -a "$LOG_FILE"
echo "  Max prediction length: ${MAX_PREDICTION_LENGTH}" | tee -a "$LOG_FILE"
echo "  Validation horizon: ${VAL_HORIZON}" | tee -a "$LOG_FILE"

python "${SCRIPT_DIR}/create_seq_model_data_loaders.py" \
  --data_dir "$DATA_DIR" \
  --dataloader_dir "$DATALOADER_DIR" \
  --log_dir "$LOG_DIR" \
  --log_level "$LOG_LEVEL" \
  --window_size "$WINDOW_SIZE" \
  --max_encoder_length "$MAX_ENCODER_LENGTH" \
  --max_prediction_length "$MAX_PREDICTION_LENGTH" \
  --val_horizon "$VAL_HORIZON" \
   2>&1 | tee -a "$LOG_FILE"


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
