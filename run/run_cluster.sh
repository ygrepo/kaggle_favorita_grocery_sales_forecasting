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

# DATA_FN="${OUTPUT_DATA_DIR}/20250707_train_top_51_store_100_item.csv"
# STORE_ITEM_MATRIX_FN="${OUTPUT_DATA_DIR}/20250707_train_top_51_store_100_item_matrix.csv"
# CLUSTER_OUTPUT_FN="${OUTPUT_DATA_DIR}/20250707_train_top_51_store_100_item_sales_cluster.csv"
# OUTPUT_FN="${OUTPUT_DATA_DIR}/20250707_train_top_51_store_100_item_cluster.csv"

#DATA_FN="${OUTPUT_DATA_DIR}/20250711_train_top_store_100_item.csv" 
#DATA_FN="${OUTPUT_DATA_DIR}/20250711_train_top_store_20_item.csv" 
DATA_FN="${OUTPUT_DATA_DIR}/20150209_10_train_top_store_20_item_cluster.csv"

# STORE_ITEM_MATRIX_FN="${OUTPUT_DATA_DIR}/20250711_train_top_20_item_matrix.csv"
# CLUSTER_OUTPUT_FN="${OUTPUT_DATA_DIR}/20250711_train_top_20_item_cluster_result.csv"
# OUTPUT_FN="${OUTPUT_DATA_DIR}/20250711_train_top_20_item_cluster.csv"

STORE_ITEM_MATRIX_FN="${OUTPUT_DATA_DIR}/20150209_10_train_top_20_item_matrix.csv"
CLUSTER_OUTPUT_FN="${OUTPUT_DATA_DIR}/20150209_10_train_top_20_item_cluster_result.csv"
OUTPUT_FN="${OUTPUT_DATA_DIR}/20150209_10_train_top_20_item_cluster.csv"


ROW_RANGE="10:20"
COL_RANGE="10:20"
LOG_DIR="${PROJECT_ROOT}/output/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_data_clustering.log"
LOG_LEVEL="DEBUG"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --data-fn) DATA_FN="$2"; shift 2 ;;
    --output-data-dir) OUTPUT_DATA_DIR="$2"; shift 2 ;;
    --store-item-matrix-fn) STORE_ITEM_MATRIX_FN="$2"; shift 2 ;;
    --cluster-output-fn) CLUSTER_OUTPUT_FN="$2"; shift 2 ;;
    --output-fn) OUTPUT_FN="$2"; shift 2 ;;
    --row-range) ROW_RANGE="$2"; shift 2 ;;
    --col-range) COL_RANGE="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    --log-level) LOG_LEVEL="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# Create output directories if they don't exist
mkdir -p "$LOG_DIR"


# Set up log file with timestamp
echo "Starting script at $(date)" | tee -a "$LOG_FILE"
echo "Project root: $PROJECT_ROOT" | tee -a "$LOG_FILE"
echo "Data dir: $DATA_DIR" | tee -a "$LOG_FILE"
echo "Output data dir: $OUTPUT_DATA_DIR" | tee -a "$LOG_FILE"
echo "Data fn: $DATA_FN" | tee -a "$LOG_FILE"
echo "Store item matrix fn: $STORE_ITEM_MATRIX_FN" | tee -a "$LOG_FILE"
echo "Cluster output fn: $CLUSTER_OUTPUT_FN" | tee -a "$LOG_FILE"
echo "Output fn: $OUTPUT_FN" | tee -a "$LOG_FILE"
echo "Log level: $LOG_LEVEL" | tee -a "$LOG_FILE"
echo "Row range: $ROW_RANGE" | tee -a "$LOG_FILE"
echo "Col range: $COL_RANGE" | tee -a "$LOG_FILE"
echo "Log dir: $LOG_DIR" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"

python "${SCRIPT_DIR}/run_clustering.py" \
  --data-fn "$DATA_FN" \
  --store-item-matrix-fn "$STORE_ITEM_MATRIX_FN" \
  --cluster-output-fn "$CLUSTER_OUTPUT_FN" \
  --output-fn "$OUTPUT_FN" \
  --row-range "$ROW_RANGE" \
  --col-range "$COL_RANGE" \
  --log-dir "$LOG_DIR" \
  --log-level "$LOG_LEVEL" \
   2>&1 | tee -a "$LOG_FILE"

# Check the exit status of the Python script
CLUSTERING_EXIT_CODE=${PIPESTATUS[0]}

if [ $CLUSTERING_EXIT_CODE -eq 0 ]; then
    echo "Script completed successfully at $(date)" | tee -a "$LOG_FILE"
    exit 0
else
    echo "Error: Script failed with exit code $CLUSTERING_EXIT_CODE" | tee -a "$LOG_FILE"
    echo "Check the log file for details: $LOG_FILE"
    exit $CLUSTERING_EXIT_CODE
fi
