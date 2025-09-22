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

DATA_GROWTH_RATE_DIR="${OUTPUT_DATA_DIR}/growth_rate_2014_January_top_53_store_2000_item"
DATA_FN="${DATA_GROWTH_RATE_DIR}/growth_rate_2014_January_top_53_store_2000_item.parquet"
#DATA_FN="${OUTPUT_DATA_DIR}/train_2014_2015_top_53_store_2000_item_growth_rate.parquet"

ROW_RANGE=90
COL_RANGE=230
# ROW_RANGE=90,120,140
# COL_RANGE=230,240,250

ALPHA="0.5"
BETA="0.001"
BLOCK_L1="0"
B_INNER="35"
MAX_ITER="200"  
TOL="1E-6"  
MAX_PVE_DROP="0.01"  
MIN_SIL="-0.05"  
MIN_KEEP="10"  
TOP_K="10"
K_ROW=0
K_COL=0
KEEP_STRATEGY=""

TOP_RANK_FN="${DATA_GROWTH_RATE_DIR}/growth_rate_2014_January_top_53_store_2000_item_cluster_bt_top_rank.csv"
SUMMARY_FN="${DATA_GROWTH_RATE_DIR}/growth_rate_2014_January_top_53_store_2000_item_cluster_bt_summary.csv"
FIGURE_FN="${DATA_GROWTH_RATE_DIR}/growth_rate_2014_January_top_53_store_2000_item_cluster_bt_figure.tiff"
OUTPUT_FN="${DATA_GROWTH_RATE_DIR}/growth_rate_2014_January_top_53_store_2000_item_cluster_bt.parquet"
# TOP_RANK_FN="${OUTPUT_DATA_DIR}/train_2014_2015_top_53_store_2000_item_cluster_bt_top_rank.csv"
# SUMMARY_FN="${OUTPUT_DATA_DIR}/train_2014_2015_top_53_store_2000_item_cluster_bt_summary.csv"
# FIGURE_FN="${OUTPUT_DATA_DIR}/train_2014_2015_top_53_store_2000_item_cluster_bt_figure.png"
# OUTPUT_FN="${OUTPUT_DATA_DIR}/train_2014_2015_top_53_store_2000_item_cluster_bt.parquet"

LOG_DIR="${PROJECT_ROOT}/output/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_data_clustering.log"
LOG_LEVEL="INFO"


N_JOBS=-1
BATCH_SIZE=1

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
    --k_row) K_ROW="$2"; shift 2 ;;
    --k_col) K_COL="$2"; shift 2 ;;
    --max_iter) MAX_ITER="$2"; shift 2 ;;
    --tol) TOL="$2"; shift 2 ;;
    --max_pve_drop) MAX_PVE_DROP="$2"; shift 2 ;;
    --min_sil) MIN_SIL="$2"; shift 2 ;;
    --min_keep) MIN_KEEP="$2"; shift 2 ;;
    --top_k) TOP_K="$2"; shift 2 ;;
    --top_rank_fn) TOP_RANK_FN="$2"; shift 2 ;;
    --keep_strategy) KEEP_STRATEGY="$2"; shift 2 ;;
    --n_jobs) N_JOBS="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --summary_fn) SUMMARY_FN="$2"; shift 2 ;;
    --figure_fn) FIGURE_FN="$2"; shift 2 ;;
    --output_fn) OUTPUT_FN="$2"; shift 2 ;;
    --output_data_dir) OUTPUT_DATA_DIR="$2"; shift 2 ;;
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
echo "Row range: $ROW_RANGE" | tee -a "$LOG_FILE"
echo "Col range: $COL_RANGE" | tee -a "$LOG_FILE"
echo "Alpha: $ALPHA" | tee -a "$LOG_FILE"
echo "Beta: $BETA" | tee -a "$LOG_FILE"
echo "Block l1: $BLOCK_L1" | tee -a "$LOG_FILE"
echo "B inner: $B_INNER" | tee -a "$LOG_FILE"
echo "Max iter: $MAX_ITER" | tee -a "$LOG_FILE"
echo "Tolerance: $TOL" | tee -a "$LOG_FILE"
echo "K row: $K_ROW" | tee -a "$LOG_FILE"
echo "K col: $K_COL" | tee -a "$LOG_FILE"
echo "Keep strategy: $KEEP_STRATEGY" | tee -a "$LOG_FILE"
echo "Max PVE drop: $MAX_PVE_DROP" | tee -a "$LOG_FILE"
echo "Min Silhouette: $MIN_SIL" | tee -a "$LOG_FILE"
echo "Min keep: $MIN_KEEP" | tee -a "$LOG_FILE"
echo "Top k: $TOP_K" | tee -a "$LOG_FILE"
echo "Top rank fn: $TOP_RANK_FN" | tee -a "$LOG_FILE"
echo "N jobs: $N_JOBS" | tee -a "$LOG_FILE"
echo "Batch size: $BATCH_SIZE" | tee -a "$LOG_FILE"
echo "Summary fn: $SUMMARY_FN" | tee -a "$LOG_FILE"
echo "Figure fn: $FIGURE_FN" | tee -a "$LOG_FILE"
echo "Output fn: $OUTPUT_FN" | tee -a "$LOG_FILE"
echo "Log level: $LOG_LEVEL" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"


# Run the script
set +e
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
  --k_row "$K_ROW" \
  --k_col "$K_COL" \
  --max_pve_drop "$MAX_PVE_DROP" \
  --min_sil "$MIN_SIL" \
  --min_keep "$MIN_KEEP" \
  --top_k "$TOP_K" \
  --top_rank_fn "$TOP_RANK_FN" \
  --keep_strategy "$KEEP_STRATEGY" \
  --n_jobs "$N_JOBS" \
  --batch_size "$BATCH_SIZE" \
  --summary_fn "$SUMMARY_FN" \
  --figure_fn "$FIGURE_FN" \
  --output_fn "$OUTPUT_FN" \
  --log_fn "$LOG_FILE" \
  --log_level "$LOG_LEVEL" 
exit_code=$?
set -e

if [[ ${exit_code} -eq 0 ]]; then
    echo "OK: finished at $(date)" | tee -a "${LOG_FILE}"
else
    echo "Python script failed with exit code $exit_code" >&2
    exit ${exit_code}
fi
