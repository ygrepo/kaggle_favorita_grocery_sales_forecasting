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
#DATA_FN="${OUTPUT_DATA_DIR}/2014_January_top_53_store_2000_item_growth_rate_input_clustered.parquet"
#DATA_FN="${OUTPUT_DATA_DIR}/top_gc_median_df.parquet"
DATA_FN="${OUTPUT_DATA_DIR}/2014_January_top_53_store_2000_item_growth_rate_imputed_features.parquet"

#TOP_RANK_FN="${OUTPUT_DATA_DIR}/top_gc_median_df_top_rank.csv"
#SUMMARY_FN="${OUTPUT_DATA_DIR}/top_gc_median_df_summary.csv"
#BLOCK_ID_FN="${OUTPUT_DATA_DIR}/top_gc_median_df_block_id.npy"
#OUTPUT_FN="${OUTPUT_DATA_DIR}/top_gc_median_df.csv"
#MODEL_FN="${OUTPUT_DATA_DIR}/top_gc_median_df_model.pickle"



TOP_RANK_FN="${OUTPUT_DATA_DIR}/2014_January_top_53_store_2000_item_growth_rate_clustered_top_rank.csv"
SUMMARY_FN="${OUTPUT_DATA_DIR}/2014_January_top_53_store_2000_item_growth_rate_clustered_summary.csv"
OUTPUT_FN="${OUTPUT_DATA_DIR}/2014_January_top_53_store_2000_item_growth_rate_clustered.parquet"
MODEL_FN="${OUTPUT_DATA_DIR}/2014_January_top_53_store_2000_item_growth_rate_clustered_model.pickle"

ROW_RANGE=10
COL_RANGE=80

ALPHA=1e-2 
BETA=1E-4  # Aggressively reduce sparsity
BLOCK_L1=0
MAX_ITER=500  
TOL=1E-6    # Keep relaxed tolerance
PATIENCE=20
FEATURE_WEIGHTS="1.0,0.1,0.1,1.0,1.0,1.0,1.0,1,1"
#FEATURE_WEIGHTS="1.0,0,0,0,0,0,0,0,0"

K_ROW=0
K_COL=0
B_INNER=35
MAX_PVE_DROP=0.01 
TOP_K=100
EMPTY_CLUSTER_PENALTY=1.0
MIN_CLUSTER_SIZE=2
MULTIFEATURE=True
NMF_RANK=100
FEATURES="gr_median,gr_std,gr_iqr,frac_up,frac_sideways,frac_down,up_to_down_ratio,ac_lag1,ac_lag4"

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
    --top_k) TOP_K="$2"; shift 2 ;;
    --top_rank_fn) TOP_RANK_FN="$2"; shift 2 ;;
    --n_jobs) N_JOBS="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --summary_fn) SUMMARY_FN="$2"; shift 2 ;;
    --model_fn) MODEL_FN="$2"; shift 2 ;;
    --output_data_dir) OUTPUT_DATA_DIR="$2"; shift 2 ;;
    --Empty_cluster_penalty) EMPTY_CLUSTER_PENALTY="$2"; shift 2 ;;
    --min_cluster_size) MIN_CLUSTER_SIZE="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --multifeature) MULTIFEATURE="$2"; shift 2 ;;
    --nmf_rank) NMF_RANK="$2"; shift 2 ;;
    --features) FEATURES="$2"; shift 2 ;;
    --feature_weights) FEATURE_WEIGHTS="$2"; shift 2 ;;
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
echo "Max PVE drop: $MAX_PVE_DROP" | tee -a "$LOG_FILE"
echo "Top k: $TOP_K" | tee -a "$LOG_FILE"
echo "Top rank fn: $TOP_RANK_FN" | tee -a "$LOG_FILE"
echo "Empty cluster penalty: $EMPTY_CLUSTER_PENALTY" | tee -a "$LOG_FILE"
echo "Min cluster size: $MIN_CLUSTER_SIZE" | tee -a "$LOG_FILE"
echo "N jobs: $N_JOBS" | tee -a "$LOG_FILE"
echo "Batch size: $BATCH_SIZE" | tee -a "$LOG_FILE"
echo "Summary fn: $SUMMARY_FN" | tee -a "$LOG_FILE"
echo "Model fn: $MODEL_FN" | tee -a "$LOG_FILE"
echo "Patience: $PATIENCE" | tee -a "$LOG_FILE"
echo "Multifeature: $MULTIFEATURE" | tee -a "$LOG_FILE"
echo "Features: $FEATURES" | tee -a "$LOG_FILE"
echo "Feature weights: $FEATURE_WEIGHTS" | tee -a "$LOG_FILE"
echo "NMF rank: $NMF_RANK" | tee -a "$LOG_FILE"
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
  --top_k "$TOP_K" \
  --top_rank_fn "$TOP_RANK_FN" \
  --n_jobs "$N_JOBS" \
  --batch_size "$BATCH_SIZE" \
  --summary_fn "$SUMMARY_FN" \
  --model_fn "$MODEL_FN" \
  --patience "$PATIENCE" \
  --multifeature "$MULTIFEATURE" \
  --features "$FEATURES" \
  --feature_weights "$FEATURE_WEIGHTS" \
  --nmf_rank "$NMF_RANK" \
  --log_fn "$LOG_FILE" \
  --empty_cluster_penalty "$EMPTY_CLUSTER_PENALTY" \
  --min_cluster_size "$MIN_CLUSTER_SIZE" \
  --log_level "$LOG_LEVEL" 
exit_code=$?
set -e

if [[ ${exit_code} -eq 0 ]]; then
    echo "OK: finished at $(date)" | tee -a "${LOG_FILE}"
else
    echo "Python script failed with exit code $exit_code" >&2
    exit ${exit_code}
fi
