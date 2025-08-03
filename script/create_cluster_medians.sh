#!/bin/bash

# Set strict error handling
set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Default configuration
DATA_DIR="${PROJECT_ROOT}/output/data"
DATA_FN="${OUTPUT_DATA_DIR}/train_2014_2015_top_53_store_2000_item_cluster.parquet"
OUTPUT_FN="${DATA_DIR}/train_2014_2015_top_53_store_2000_item_cluster_medians.parquet"
ITEM_FN="${DATA_DIR}/train_2014_2015_top_53_store_2000_item_item_cluster_medians.parquet"
STORE_FN="${DATA_DIR}/train_2014_2015_top_53_store_2000_item_store_cluster_medians.parquet"

LOG_DIR="${PROJECT_ROOT}/output/logs"
LOG_LEVEL="DEBUG"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --data_fn) DATA_FN="$2"; shift 2 ;;
    --item_fn) ITEM_FN="$2"; shift 2 ;;
    --store_fn) STORE_FN="$2"; shift 2 ;;
    --log_dir) LOG_DIR="$2"; shift 2 ;;
    --log_level) LOG_LEVEL="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# Create output directories if they don't exist
mkdir -p "$LOG_DIR"
mkdir -p "$DATA_DIR"

# Set up log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/create_cluster_medians_${TIMESTAMP}.log"

echo "Starting script at $(date)" | tee -a "$LOG_FILE"
echo "Project root: $PROJECT_ROOT" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Log level: $LOG_LEVEL" | tee -a "$LOG_FILE"

# Run the training script
set +e  # Disable exit on error to handle the error message
echo "Starting training with the following configuration:" | tee -a "$LOG_FILE"
echo "  Data fn: ${DATA_FN}" | tee -a "$LOG_FILE"
echo "  Item fn: ${ITEM_FN}" | tee -a "$LOG_FILE"
echo "  Store fn: ${STORE_FN}" | tee -a "$LOG_FILE"
echo "  Output fn: ${OUTPUT_FN}" | tee -a "$LOG_FILE"
echo "  Log dir: ${LOG_DIR}" | tee -a "$LOG_FILE"
echo "  Log level: ${LOG_LEVEL}" | tee -a "$LOG_FILE"

python "${SCRIPT_DIR}/create_cluster_medians.py" \
  --data_fn "$DATA_FN" \
  --item_fn "$ITEM_FN" \
  --store_fn "$STORE_FN" \
  --output_fn "$OUTPUT_FN" \
  --log_dir "$LOG_DIR" \
  --log_level "$LOG_LEVEL" \
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
