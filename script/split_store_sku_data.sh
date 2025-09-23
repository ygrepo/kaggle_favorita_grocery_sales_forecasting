#!/bin/bash

# Set strict error handling
set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Default configuration
DATA_DIR="${PROJECT_ROOT}/output/data/growth_rate_2014_January_top_53_store_2000_item"
DATA_FN="${DATA_DIR}/growth_rate_2014_January_top_53_store_2000_item_cluster_bt_medians.parquet"
OUTPUT_DIR="${DATA_DIR}/data_split"
# Create output directories if they don't exist
mkdir -p "$OUTPUT_DIR"
LOG_DIR="${PROJECT_ROOT}/output/logs"
mkdir -p "$LOG_DIR"
LOG_LEVEL="DEBUG"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/growth_rate_2014_January_top_53_store_2000_item_split_store_sku_data_${TIMESTAMP}.log"


# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_fn)
      shift
      DATA_FN="$1"
      ;;
    --output_dir)
      shift
      OUTPUT_DIR="$1"
      ;;
    --log_dir)
      shift
      LOG_DIR="$1"
      ;;
    --log_file)
      shift
      LOG_FILE="$1"
      ;;
    --log_level)
      shift
      LOG_LEVEL="$1"
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
  shift
done


# Set up log file with timestamp
echo "Starting script at $(date)" | tee -a "$LOG_FILE"
echo "Project root: $PROJECT_ROOT" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Data fn: $DATA_FN" | tee -a "$LOG_FILE"
echo "Output dir: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Log dir: $LOG_DIR" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Log level: $LOG_LEVEL" | tee -a "$LOG_FILE"

# Run the training script
set +e  # Disable exit on error to handle the error message
echo "Starting script with the following configuration:" | tee -a "$LOG_FILE"
echo "  Data fn: ${DATA_FN}" | tee -a "$LOG_FILE"
echo "  Output dir: ${OUTPUT_DIR}" | tee -a "$LOG_FILE"


python "${SCRIPT_DIR}/split_store_sku_data.py" \
  --data_fn "$DATA_FN" \
  --output_dir "$OUTPUT_DIR" \
  --log_file "$LOG_FILE" \
  --log_level "$LOG_LEVEL" \
   2>&1 | tee -a "$LOG_FILE"

# Check the exit status of the Python script
exit_status=$?
echo "Script finished with exit status: $exit_status" | tee -a "$LOG_FILE"
exit $exit_status
