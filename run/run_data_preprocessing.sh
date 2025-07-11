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
DATA_FN="${DATA_DIR}/20250707_train.csv"
OUTPUT_DATA_DIR="${PROJECT_ROOT}/output/data"
#DATA_FN="${OUTPUT_DATA_DIR}/20250711_train_44_1503844.csv"
OUTPUT_FN="${OUTPUT_DATA_DIR}/20250711_train_top_store_100_item.csv"
LOG_DIR="${PROJECT_ROOT}/output/logs"
LOG_LEVEL="DEBUG"
ITEM_TOP_N=25
ITEM_MED_N=25
ITEM_BOTTOM_N=25
ITEM_FN="${OUTPUT_DATA_DIR}/20250711_train_top_100_item_sale.csv"
STORE_FN="${OUTPUT_DATA_DIR}/20250711_train_top_store_sale.csv"
GROUP_STORE_COLUMN="store"
GROUP_ITEM_COLUMN="item"
VALUE_COLUMN="unit_sales"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --data-fn) DATA_FN="$2"; shift 2 ;;
    --output-data-dir) OUTPUT_DATA_DIR="$2"; shift 2 ;;
    --output-fn) OUTPUT_FN="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --log-level) LOG_LEVEL="$2"; shift 2 ;;
    --item-top-n) ITEM_TOP_N="$2"; shift 2 ;;
    --item-med-n) ITEM_MED_N="$2"; shift 2 ;;
    --item-bottom-n) ITEM_BOTTOM_N="$2"; shift 2 ;;
    --item-fn) ITEM_FN="$2"; shift 2 ;;
    --store-fn) STORE_FN="$2"; shift 2 ;;
    --group-store-column) GROUP_STORE_COLUMN="$2"; shift 2 ;;
    --group-item-column) GROUP_ITEM_COLUMN="$2"; shift 2 ;;
    --value-column) VALUE_COLUMN="$2"; shift 2 ;;
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
echo "Output fn: $OUTPUT_FN" | tee -a "$LOG_FILE"
echo "Log dir: $LOG_DIR" | tee -a "$LOG_FILE"
echo "Top items n: $ITEM_TOP_N" | tee -a "$LOG_FILE"
echo "Median items n: $ITEM_MED_N" | tee -a "$LOG_FILE"
echo "Bottom items n: $ITEM_BOTTOM_N" | tee -a "$LOG_FILE"
echo "Group Store column: $GROUP_STORE_COLUMN" | tee -a "$LOG_FILE"
echo "Group Item column: $GROUP_ITEM_COLUMN" | tee -a "$LOG_FILE"
echo "Value column: $VALUE_COLUMN" | tee -a "$LOG_FILE"

python "${SCRIPT_DIR}/run_data_preprocessing.py" \
  --data-fn "$DATA_FN" \
  --output-fn "$OUTPUT_FN" \
  --log-dir "$LOG_DIR" \
  --log-level "$LOG_LEVEL" \
  --store-fn "$STORE_FN" \
  --item-top-n "$ITEM_TOP_N" \
  --item-med-n "$ITEM_MED_N" \
  --item-bottom-n "$ITEM_BOTTOM_N" \
  --item-fn "$ITEM_FN" \
  --group-store-column "$GROUP_STORE_COLUMN" \
  --group-item-column "$GROUP_ITEM_COLUMN" \
  --value-column "$VALUE_COLUMN" \
   2>&1 | tee -a "$LOG_FILE"

# Check the exit status of the Python script
DATA_PREPROCESSING_EXIT_CODE=${PIPESTATUS[0]}

if [ $DATA_PREPROCESSING_EXIT_CODE -eq 0 ]; then
    echo "Script completed successfully at $(date)" | tee -a "$LOG_FILE"
    exit 0
else
    echo "Error: Script failed with exit code $DATA_PREPROCESSING_EXIT_CODE" | tee -a "$LOG_FILE"
    echo "Check the log file for details: $LOG_FILE"
    exit $DATA_PREPROCESSING_EXIT_CODE
fi
