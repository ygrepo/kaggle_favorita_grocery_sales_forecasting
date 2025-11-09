#!/bin/bash

# Set strict error handling
set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Default configuration
DATA_DIR="${PROJECT_ROOT}/data"
DATA_FN="${DATA_DIR}/train.csv"
WEIGHTS_FN="${DATA_DIR}/items.csv"
START_DATE="2013-01-01"
START_YEAR=$(echo $START_DATE | cut -d'-' -f1)
END_DATE="2014-01-31"
END_YEAR=$(echo $END_DATE | cut -d'-' -f1)

OUTPUT_DATA_DIR="${PROJECT_ROOT}/output/data"
mkdir -p "$OUTPUT_DATA_DIR"
OUTPUT_FN="${OUTPUT_DATA_DIR}/${START_YEAR}_${END_YEAR}_store_2000_item.parquet"

LOG_DIR="${PROJECT_ROOT}/output/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/preprocess_$(date +"%Y%m%d_%H%M%S").log"


LOG_LEVEL="DEBUG"
ITEM_TOP_N=700
ITEM_MED_N=300
ITEM_BOTTOM_N=700
# STORE_TOP_N=4
# STORE_MED_N=2
# STORE_BOTTOM_N=4
# ITEM_TOP_N=6
# ITEM_MED_N=4
# ITEM_BOTTOM_N=6
NROWS=0

ITEM_FN="${OUTPUT_DATA_DIR}/items.csv"
STORE_FN="${OUTPUT_DATA_DIR}/stores.csv"
GROUP_STORE_COLUMN="store"
GROUP_ITEM_COLUMN="item"
VALUE_COLUMN="unit_sales"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    --item_fn) ITEM_FN="$2"; shift 2 ;;
    --store_fn) STORE_FN="$2"; shift 2 ;;
    --nrows) NROWS="$2"; shift 2 ;;
    --start_date) START_DATE="$2"; shift 2 ;;
    --end_date) END_DATE="$2"; shift 2 ;;
    --output_fn) OUTPUT_FN="$2"; shift 2 ;;
    --log_dir) LOG_DIR="$2"; shift 2 ;;
    --log_level) LOG_LEVEL="$2"; shift 2 ;;
    --item_top_n) ITEM_TOP_N="$2"; shift 2 ;;
    --item_med_n) ITEM_MED_N="$2"; shift 2 ;;
    --item_bottom_n) ITEM_BOTTOM_N="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# Create output directories if they don't exist
mkdir -p "$LOG_DIR"


# Set up log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/data_preprocessing_${TIMESTAMP}.log"
echo "Starting script at $(date)" | tee -a "$LOG_FILE"
echo "Project root: $PROJECT_ROOT" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Log level: $LOG_LEVEL" | tee -a "$LOG_FILE"
echo "Data fn: $DATA_FN" | tee -a "$LOG_FILE"
echo "Item fn: $ITEM_FN" | tee -a "$LOG_FILE"
echo "Store fn: $STORE_FN" | tee -a "$LOG_FILE"
echo "Output fn: $OUTPUT_FN" | tee -a "$LOG_FILE"
echo "Nrows: $NROWS" | tee -a "$LOG_FILE"
echo "Start date: $START_DATE" | tee -a "$LOG_FILE"
echo "End date: $END_DATE" | tee -a "$LOG_FILE"
echo "Top items n: $ITEM_TOP_N" | tee -a "$LOG_FILE"
echo "Median items n: $ITEM_MED_N" | tee -a "$LOG_FILE"
echo "Bottom items n: $ITEM_BOTTOM_N" | tee -a "$LOG_FILE"

python "${SCRIPT_DIR}/preprocess_data.py" \
  --data_fn "$DATA_FN" \
  --output_fn "$OUTPUT_FN" \
  --item_fn "$ITEM_FN" \
  --store_fn "$STORE_FN" \
  --log_level "$LOG_LEVEL" \
  --log_fn "$LOG_FILE" \
  --nrows "$NROWS" \
  --start_date "$START_DATE" \
  --end_date "$END_DATE" \
  --item_top_n "$ITEM_TOP_N" \
  --item_med_n "$ITEM_MED_N" \
  --item_bottom_n "$ITEM_BOTTOM_N"
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
