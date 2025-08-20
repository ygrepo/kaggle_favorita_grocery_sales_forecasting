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

DATA_FN="${OUTPUT_DATA_DIR}/train_2014_2015_top_53_store_2000_item.parquet"
STORE_ITEM_MATRIX_FN=""
ITEM_FN="${OUTPUT_DATA_DIR}/train_2014_2015_top_53_store_2000_item_item_cluster.csv"
STORE_FN="${OUTPUT_DATA_DIR}/train_2014_2015_top_53_store_2000_item_store_cluster.csv"

MAV_OUTPUT_FN="${OUTPUT_DATA_DIR}/train_2014_2015_top_53_store_2000_item_cluster_mav"
mkdir -p "$MAV_OUTPUT_FN"
ONLY_BEST_MODEL="True"
ONLY_TOP_N_CLUSTERS="2"
OUTPUT_FN="${OUTPUT_DATA_DIR}/train_2014_2015_top_53_store_2000_item_cluster.parquet"

ROW_RANGE="5:10"
COL_RANGE="5:20"
# ROW_RANGE="10:20"
# COL_RANGE="10:20"
LOG_DIR="${PROJECT_ROOT}/output/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_data_clustering.log"
LOG_LEVEL="DEBUG"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    --data_fn) DATA_FN="$2"; shift 2 ;;
    --output_data_dir) OUTPUT_DATA_DIR="$2"; shift 2 ;;
    --only_best_model) ONLY_BEST_MODEL="$2"; shift 2 ;;
    --only_top_n_clusters) ONLY_TOP_N_CLUSTERS="$2"; shift 2 ;;
    --store_item_matrix_fn) STORE_ITEM_MATRIX_FN="$2"; shift 2 ;;
    --mav_output_fn) MAV_OUTPUT_FN="$2"; shift 2 ;;
    --output_fn) OUTPUT_FN="$2"; shift 2 ;;
    --item_fn) ITEM_FN="$2"; shift 2 ;;
    --store_fn) STORE_FN="$2"; shift 2 ;;
    --row_range) ROW_RANGE="$2"; shift 2 ;;
    --col_range) COL_RANGE="$2"; shift 2 ;;
    --log_dir) LOG_DIR="$2"; shift 2 ;;
    --log_file) LOG_FILE="$2"; shift 2 ;;
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
echo "Store item matrix fn: $STORE_ITEM_MATRIX_FN" | tee -a "$LOG_FILE"
echo "Item fn: $ITEM_FN" | tee -a "$LOG_FILE"
echo "Store fn: $STORE_FN" | tee -a "$LOG_FILE"
echo "MAV output fn: $MAV_OUTPUT_FN" | tee -a "$LOG_FILE"
echo "Only best model: $ONLY_BEST_MODEL" | tee -a "$LOG_FILE"
echo "Only top n clusters: $ONLY_TOP_N_CLUSTERS" | tee -a "$LOG_FILE"
echo "Output fn: $OUTPUT_FN" | tee -a "$LOG_FILE"
echo "Log level: $LOG_LEVEL" | tee -a "$LOG_FILE"
echo "Row range: $ROW_RANGE" | tee -a "$LOG_FILE"
echo "Col range: $COL_RANGE" | tee -a "$LOG_FILE"
echo "Log dir: $LOG_DIR" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"

python "${SCRIPT_DIR}/cluster.py" \
  --data_fn "$DATA_FN" \
  --store_item_matrix_fn "$STORE_ITEM_MATRIX_FN" \
  --mav_output_fn "$MAV_OUTPUT_FN" \
  --output_fn "$OUTPUT_FN" \
  --item_fn "$ITEM_FN" \
  --store_fn "$STORE_FN" \
  --row_range "$ROW_RANGE" \
  --col_range "$COL_RANGE" \
  --only_best_model "$ONLY_BEST_MODEL" \
  --only_top_n_clusters "$ONLY_TOP_N_CLUSTERS" \
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
