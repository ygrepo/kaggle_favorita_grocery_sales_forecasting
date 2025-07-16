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

#OUTPUT_FN="${DATA_DIR}/20250711_train_top_20_item_clusters_sales_cyclical_features_1_days_X_y.csv"
OUTPUT_FN="${DATA_DIR}/train_top_store_15_item_clusters_sales_cyclical_features.parquet"

#SALES_FN="${DATA_DIR}/20250711_train_top_20_item_sale_cluster.csv"
# OUTPUT_FN="${DATA_DIR}/20250711_train_top_51_store_100_item_clusters_sales_cyclical_features_1_days_X_y.csv"
SALES_FN="${DATA_DIR}/train_top_store_15_item_sale_cluster.parquet"
CYC_FN="${DATA_DIR}/train_top_store_15_item_cyc_cluster.parquet"
# SALES_FN="${DATA_DIR}/20150209_10_train_top_20_item_sale_cluster.csv"
# CYC_FN="${DATA_DIR}/20150209_10_train_top_20_item_cyc_cluster.csv"
#CYC_FN="${DATA_DIR}/20250711_train_top_20_item_cyc_cluster.csv"
LOG_DIR="${PROJECT_ROOT}/output/logs"
LOG_LEVEL="DEBUG"
WINDOW_SIZE=1
ADD_Y_TARGETS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-fn) OUTPUT_FN="$2"; shift 2 ;;
    --sales-fn) SALES_FN="$2"; shift 2 ;;
    --cyc-fn) CYC_FN="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --log-level) LOG_LEVEL="$2"; shift 2 ;;
    --window-size) WINDOW_SIZE="$2"; shift 2 ;;
    --add-y-targets) ADD_Y_TARGETS="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# Create output directories if they don't exist
mkdir -p "$LOG_DIR"

# Set up log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/create_features_${TIMESTAMP}.log"

echo "Starting script at $(date)" | tee -a "$LOG_FILE"
echo "Project root: $PROJECT_ROOT" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Window size: $WINDOW_SIZE" | tee -a "$LOG_FILE"
echo "Add y targets: $ADD_Y_TARGETS" | tee -a "$LOG_FILE"
echo "Log level: $LOG_LEVEL" | tee -a "$LOG_FILE"

# Run the training script
set +e  # Disable exit on error to handle the error message
echo "Starting training with the following configuration:" | tee -a "$LOG_FILE"
echo "  Output fn: ${OUTPUT_FN}" | tee -a "$LOG_FILE"
echo "  Sales fn: ${SALES_FN}" | tee -a "$LOG_FILE"
echo "  Cyc fn: ${CYC_FN}" | tee -a "$LOG_FILE"
echo "  Window size: ${WINDOW_SIZE}" | tee -a "$LOG_FILE"
echo "  Add y targets: ${ADD_Y_TARGETS}" | tee -a "$LOG_FILE"
echo "  Log level: ${LOG_LEVEL}" | tee -a "$LOG_FILE"

python "${SCRIPT_DIR}/run_create_features.py" \
  --output-fn "$OUTPUT_FN" \
  --sales-fn "$SALES_FN" \
  --cyc-fn "$CYC_FN" \
  --log-dir "$LOG_DIR" \
  --log-level "$LOG_LEVEL" \
  --window-size "$WINDOW_SIZE" \
  --add-y-targets "$ADD_Y_TARGETS" \
   2>&1 | tee -a "$LOG_FILE"

# Check the exit status of the Python script
TRAINING_EXIT_CODE=${PIPESTATUS[0]}

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Script completed successfully at $(date)" | tee -a "$LOG_FILE"
    exit 0
else
    echo "Error: Script failed with exit code $TRAINING_EXIT_CODE" | tee -a "$LOG_FILE"
    echo "Check the log file for details: $LOG_FILE"
    exit $TRAINING_EXIT_CODE
fi
