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

#DATA_FN="${OUTPUT_DATA_DIR}/train_2014_January_top_53_store_2000_item.parquet"
#DATA_FN="${OUTPUT_DATA_DIR}/train_2014_January_top_53_store_2000_item.parquet"
DATA_FN="${OUTPUT_DATA_DIR}/train_2014_January_12_store_20_item_cluster.parquet"
#DATA_FN="${OUTPUT_DATA_DIR}/train_2014_2015_top_53_store_2000_item.parquet"
OUTPUT_GROWTH_RATE_DIR="${OUTPUT_DATA_DIR}/growth_rate_2014_January_12_store_20_item"
#OUTPUT_GROWTH_RATE_DIR="${OUTPUT_DATA_DIR}/growth_rate_2014_January_top_53_store_2000_item"
#OUTPUT_GROWTH_RATE_DIR="${OUTPUT_DATA_DIR}/growth_rate_2014_January_top_53_store_2000_item"
#OUTPUT_FN="${OUTPUT_GROWTH_RATE_DIR}/train_2014_January_top_53_store_2000_item_growth_rate.parquet"
OUTPUT_FN="${OUTPUT_GROWTH_RATE_DIR}/growth_rate_2014_January_12_store_20_item.parquet"
#OUTPUT_FN="${OUTPUT_GROWTH_RATE_DIR}/growth_rate_2014_January_top_53_store_2000_item.parquet"
mkdir -p "$OUTPUT_GROWTH_RATE_DIR"
#OUTPUT_FN="${OUTPUT_DATA_DIR}/train_2014_2015_top_53_store_2000_item_growth_rate.parquet"

N_JOBS=-1
BATCH_SIZE=100

LOG_DIR="${PROJECT_ROOT}/output/logs"
LOG_FILE="${LOG_DIR}/growth_rate_$(date +"%Y%m%d_%H%M%S").log"
LOG_LEVEL="INFO"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    --data_fn) DATA_FN="$2"; shift 2 ;;
    --output_fn) OUTPUT_FN="$2"; shift 2 ;;
    --output_data_dir) OUTPUT_DATA_DIR="$2"; shift 2 ;;
    --n_jobs) N_JOBS="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --log_fn) LOG_FILE="$2"; shift 2 ;;
    --log_level) LOG_LEVEL="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# Create output directories if they don't exist
mkdir -p "$LOG_DIR"


# Set up log file with timestamp
echo "Starting script at $(date)" | tee -a "$LOG_FILE"
echo "Project root: $PROJECT_ROOT" | tee -a "$LOG_FILE"
echo "Data fn: $DATA_FN" | tee -a "$LOG_FILE"
echo "Output fn: $OUTPUT_FN" | tee -a "$LOG_FILE"
echo "Log level: $LOG_LEVEL" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"

# Run the script
set +e
python "${SCRIPT_DIR}/growth_rate.py" \
  --data_fn "$DATA_FN" \
  --output_fn "$OUTPUT_FN" \
  --n_jobs "$N_JOBS" \
  --batch_size "$BATCH_SIZE" \
  --log_level "$LOG_LEVEL" \
  --log_fn "$LOG_FILE"
exit_code=$?
set -e

if [[ ${exit_code} -eq 0 ]]; then
    echo "OK: finished at $(date)" | tee -a "${LOG_FILE}"
else
    echo "Python script failed with exit code $exit_code" >&2
    exit ${exit_code}
fi
