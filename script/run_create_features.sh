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

#OUTPUT_FN="${DATA_DIR}/20250711_train_top_20_item_clusters_sales_cyclical_features_1_days_X_y.csv"
#OUTPUT_FN="${DATA_DIR}/train_top_store_15_item_clusters_sales_cyclical_features.parquet"

#SALES_FN="${DATA_DIR}/20250711_train_top_20_item_sale_cluster.csv"
# OUTPUT_FN="${DATA_DIR}/20250711_train_top_51_store_100_item_clusters_sales_cyclical_features_1_days_X_y.csv"
#SALES_FN="${DATA_DIR}/train_top_store_15_item_sale_cluster.parquet"
#CYC_FN="${DATA_DIR}/train_top_store_15_item_cyc_cluster.parquet"
#SALES_FN="${DATA_DIR}/train_top_store_15_item_sale_cluster.parquet"
#CYC_FN="${DATA_DIR}/train_2014_2015_top_53_store_2000_item_cyc_cluster.parquet"

# SALES_FN="${DATA_DIR}/20150209_10_train_top_20_item_sale_cluster.csv"
# CYC_FN="${DATA_DIR}/20150209_10_train_top_20_item_cyc_cluster.csv"
#CYC_FN="${DATA_DIR}/20250711_train_top_20_item_cyc_cluster.csv"

# SALE_DIR="${PROJECT_ROOT}/output/data/sales_data/"
# CYC_DIR="${PROJECT_ROOT}/output/data/cyc_data/"
SALE_DIR="${PROJECT_ROOT}/output/data/sales_data_28_store_10_item/"
CYC_DIR="${PROJECT_ROOT}/output/data/cyc_data_28_store_10_item/"
OUTPUT_DIR="${PROJECT_ROOT}/output/data/sale_cyc_data_28_store_10_item/"

LOG_DIR="${PROJECT_ROOT}/output/logs"
LOG_LEVEL="DEBUG"
WINDOW_SIZE=1
ADD_Y_TARGETS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --sales_dir) SALES_DIR="$2"; shift 2 ;;
    --cyc_dir) CYC_DIR="$2"; shift 2 ;;
    --log_dir) LOG_DIR="$2"; shift 2 ;;
    --log_level) LOG_LEVEL="$2"; shift 2 ;;
    --window_size) WINDOW_SIZE="$2"; shift 2 ;;
    --add_y_targets) ADD_Y_TARGETS="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# Create output directories if they don't exist
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

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
echo "  Sales dir: ${SALE_DIR}" | tee -a "$LOG_FILE"
echo "  Cyc dir: ${CYC_DIR}" | tee -a "$LOG_FILE"
echo "  Output dir: ${OUTPUT_DIR}" | tee -a "$LOG_FILE"
echo "  Window size: ${WINDOW_SIZE}" | tee -a "$LOG_FILE"
echo "  Add y targets: ${ADD_Y_TARGETS}" | tee -a "$LOG_FILE"
echo "  Log level: ${LOG_LEVEL}" | tee -a "$LOG_FILE"

python "${SCRIPT_DIR}/run_create_features.py" \
  --sales_dir "$SALE_DIR" \
  --cyc_dir "$CYC_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --log_dir "$LOG_DIR" \
  --log_level "$LOG_LEVEL" \
  --window_size "$WINDOW_SIZE" \
  --add_y_targets "$ADD_Y_TARGETS" \
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
