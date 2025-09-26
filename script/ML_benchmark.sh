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
OUTPUT_DATA_DIR="${PROJECT_ROOT}/output/data"
DATA_FN="${OUTPUT_DATA_DIR}/2014_January_top_53_store_2000_item_growth_rate_clustered_medians_cyc_X_y_features.parquet"
MODEL_DIR="${PROJECT_ROOT}/output/models"
mkdir -p "$MODEL_DIR"

LOG_DIR="${PROJECT_ROOT}/output/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_ML_benchmark.log"
LOG_LEVEL="INFO"


# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_fn) DATA_FN="$2"; shift 2 ;;
    --model_dir) MODEL_DIR="$2"; shift 2 ;;
    --log_fn) LOG_FILE="$2"; shift 2 ;;
    --log_level) LOG_LEVEL="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# Create output directories if they don't exist
mkdir -p "$LOG_DIR"

# Set up log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/ML_benchmark_${TIMESTAMP}.log"


# Run the script
set +e  # Disable exit on error to handle the error message
echo "Starting script with the following configuration:" | tee -a "$LOG_FILE"
echo "  Data fn: ${DATA_FN}" | tee -a "$LOG_FILE"
echo "  Log level: ${LOG_LEVEL}" | tee -a "$LOG_FILE"

python "${SCRIPT_DIR}/ML_benchmark.py" \
  --data_fn "$DATA_FN" \
  --model_dir "$MODEL_DIR" \
  --log_dir "$LOG_DIR" \
  --log_fn "LOG_FILE" \
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
