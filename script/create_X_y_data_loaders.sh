#!/bin/bash

# Set strict error handling
set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Default configuration
#DATA_DIR="${PROJECT_ROOT}/output/data"
#DATA_FN="${DATA_DIR}/train_top_store_15_item_clusters_sales_cyclical_features_X_1_day_y.parquet"
#DATA_FN="${DATA_DIR}/20250711_train_top_store_2000_item_clusters_sales_cyclical_features_X_1_day_y.csv"
#DATA_DIR="${PROJECT_ROOT}/output/data/sale_cyc_features_X_1_day_y/"
#DATA_DIR="${PROJECT_ROOT}/output/data/sale_cyc_features_X_1_day_y_28_store_10_item/"
DATA_DIR="${PROJECT_ROOT}/output/data/sale_cyc_features_X_1_day_y_12_store_20_item/"

DATALOADER_DIR="${PROJECT_ROOT}/output/data/dataloader_12_store_20_item/"
SCALERS_DIR="${PROJECT_ROOT}/output/data/scalers_12_store_20_item/"

LOG_DIR="${PROJECT_ROOT}/output/logs"
LOG_LEVEL="DEBUG"
WINDOW_SIZE=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    --scalers_dir) SCALERS_DIR="$2"; shift 2 ;;
    --dataloader_dir) DATALOADER_DIR="$2"; shift 2 ;;
    --log_dir) LOG_DIR="$2"; shift 2 ;;
    --log_level) LOG_LEVEL="$2"; shift 2 ;;
    --window_size) WINDOW_SIZE="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# Create output directories if they don't exist
mkdir -p "$LOG_DIR"
mkdir -p "$SCALERS_DIR"
mkdir -p "$DATALOADER_DIR"

# Set up log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/create_X_y_data_loaders_${TIMESTAMP}.log"

echo "Starting script at $(date)" | tee -a "$LOG_FILE"
echo "Project root: $PROJECT_ROOT" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Window size: $WINDOW_SIZE" | tee -a "$LOG_FILE"
echo "Log level: $LOG_LEVEL" | tee -a "$LOG_FILE"

# Run the script
set +e  # Disable exit on error to handle the error message
echo "Starting script with the following configuration:" | tee -a "$LOG_FILE"
echo "  Data dir: ${DATA_DIR}" | tee -a "$LOG_FILE"
echo "  Scalers dir: ${SCALERS_DIR}" | tee -a "$LOG_FILE"
echo "  Dataloader dir: ${DATALOADER_DIR}" | tee -a "$LOG_FILE"
echo "  Window size: ${WINDOW_SIZE}" | tee -a "$LOG_FILE"
echo "  Log level: ${LOG_LEVEL}" | tee -a "$LOG_FILE"

python "${SCRIPT_DIR}/create_X_y_data_loaders.py" \
  --data_dir "$DATA_DIR" \
  --scalers_dir "$SCALERS_DIR" \
  --dataloader_dir "$DATALOADER_DIR" \
  --log_dir "$LOG_DIR" \
  --log_level "$LOG_LEVEL" \
  --window_size "$WINDOW_SIZE" \
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
