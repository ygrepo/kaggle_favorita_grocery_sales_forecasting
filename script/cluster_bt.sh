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

ROW_RANGE="5:20"
COL_RANGE="5:20"
# ROW_RANGE="10:20"
# COL_RANGE="10:20"

ALPHA="1e-2"
BETA="0.6"
BLOCK_L1="0.0"
B_INNER="15"
MAX_ITER="200"  
TOL="1e-5"  
MAX_PVE_DROP="0.01"  
MIN_SIL="-0.05"  
MIN_KEEP="6"  
TOP_K="10"  

GROWTH_RATE_FN="${OUTPUT_DATA_DIR}/train_2014_2015_top_53_store_2000_item_growth_rate.parquet"
TOP_RANK_FN="${OUTPUT_DATA_DIR}/train_2014_2015_top_53_store_2000_item_cluster_bt_top_rank.csv"
SUMMARY_FN="${OUTPUT_DATA_DIR}/train_2014_2015_top_53_store_2000_item_cluster_bt_summary.csv"
FIGURE_FN="${OUTPUT_DATA_DIR}/train_2014_2015_top_53_store_2000_item_cluster_bt_figure.png"
OUTPUT_FN="${OUTPUT_DATA_DIR}/train_2014_2015_top_53_store_2000_item_cluster_bt.parquet"

LOG_DIR="${PROJECT_ROOT}/output/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_data_clustering.log"
LOG_LEVEL="DEBUG"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    --data_fn) DATA_FN="$2"; shift 2 ;;
    --row_range) ROW_RANGE="$2"; shift 2 ;;
    --col_range) COL_RANGE="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --beta) BETA="$2"; shift 2 ;;
    --block_l1) BLOCK_L1="$2"; shift 2 ;;
    --b_inner) B_INNER="$2"; shift 2 ;;
    --max_iter) MAX_ITER="$2"; shift 2 ;;
    --tol) TOL="$2"; shift 2 ;;
    --max_pve_drop) MAX_PVE_DROP="$2"; shift 2 ;;
    --min_sil) MIN_SIL="$2"; shift 2 ;;
    --min_keep) MIN_KEEP="$2"; shift 2 ;;
    --top_k) TOP_K="$2"; shift 2 ;;
    --growth_rate_fn) GROWTH_RATE_FN="$2"; shift 2 ;;
    --top_rank_fn) TOP_RANK_FN="$2"; shift 2 ;;
    --summary_fn) SUMMARY_FN="$2"; shift 2 ;;
    --figure_fn) FIGURE_FN="$2"; shift 2 ;;
    --output_fn) OUTPUT_FN="$2"; shift 2 ;;
    --output_data_dir) OUTPUT_DATA_DIR="$2"; shift 2 ;;
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
echo "Row range: $ROW_RANGE" | tee -a "$LOG_FILE"
echo "Col range: $COL_RANGE" | tee -a "$LOG_FILE"
echo "Alpha: $ALPHA" | tee -a "$LOG_FILE"
echo "Beta: $BETA" | tee -a "$LOG_FILE"
echo "Block l1: $BLOCK_L1" | tee -a "$LOG_FILE"
echo "B inner: $B_INNER" | tee -a "$LOG_FILE"
echo "Max iter: $MAX_ITER" | tee -a "$LOG_FILE"
echo "Tolerance: $TOL" | tee -a "$LOG_FILE"
echo "Max PVE drop: $MAX_PVE_DROP" | tee -a "$LOG_FILE"
echo "Min Silhouette: $MIN_SIL" | tee -a "$LOG_FILE"
echo "Min keep: $MIN_KEEP" | tee -a "$LOG_FILE"
echo "Top k: $TOP_K" | tee -a "$LOG_FILE"
echo "Growth rate fn: $GROWTH_RATE_FN" | tee -a "$LOG_FILE"
echo "Top rank fn: $TOP_RANK_FN" | tee -a "$LOG_FILE"
echo "Summary fn: $SUMMARY_FN" | tee -a "$LOG_FILE"
echo "Figure fn: $FIGURE_FN" | tee -a "$LOG_FILE"
echo "Output fn: $OUTPUT_FN" | tee -a "$LOG_FILE"
echo "Log level: $LOG_LEVEL" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"


python "${SCRIPT_DIR}/cluster_bt.py" \
  --data_fn "$DATA_FN" \
  --row_range "$ROW_RANGE" \
  --col_range "$COL_RANGE" \
  --alpha "$ALPHA" \
  --beta "$BETA" \
  --block_l1 "$BLOCK_L1" \
  --b_inner "$B_INNER" \
  --max_iter "$MAX_ITER" \
  --tol "$TOL" \
  --max_pve_drop "$MAX_PVE_DROP" \
  --min_sil "$MIN_SIL" \
  --min_keep "$MIN_KEEP" \
  --top_k "$TOP_K" \
  --growth_rate_fn "$GROWTH_RATE_FN" \
  --top_rank_fn "$TOP_RANK_FN" \
  --summary_fn "$SUMMARY_FN" \
  --figure_fn "$FIGURE_FN" \
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
