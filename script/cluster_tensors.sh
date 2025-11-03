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
DATA_FN="${OUTPUT_DATA_DIR}/2014_January_top_53_store_2000_item_growth_rate_imputed_features.parquet"
OUTPUT_PATH="${OUTPUT_DATA_DIR}/2014_January_top_53_store_2000_item_growth_rate_clustered.csv"

RANK_LIST="10,20,30,40,50,60,70,80,90,100"
STORE_RANKS="30,40"
SKU_RANKS="100,200,300"
FEATURE_RANKS="2,7,8"
#FEATURES="gr_median"
FEATURES="gr_median,gr_std,gr_iqr,frac_up,frac_sideways,frac_down,up_to_down_ratio,ac_lag1,ac_lag4"
MAX_ITER=500
TOL=1e-5
METHOD="tucker"

LOG_DIR="${PROJECT_ROOT}/output/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_data_clustering.log"
LOG_LEVEL="INFO"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="$2"; shift 2 ;;
    --output_path) OUTPUT_PATH="$2"; shift 2 ;;
    --data_fn) DATA_FN="$2"; shift 2 ;;
    --rank_list) RANK_LIST="$2"; shift 2 ;;
    --store_ranks) STORE_RANKS="$2"; shift 2 ;;
    --sku_ranks) SKU_RANKS="$2"; shift 2 ;;
    --feature_ranks) FEATURE_RANKS="$2"; shift 2 ;;
    --max_iter) MAX_ITER="$2"; shift 2 ;;
    --tol) TOL="$2"; shift 2 ;;
    --features) FEATURES="$2"; shift 2 ;;
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
echo "Rank list: $RANK_LIST" | tee -a "$LOG_FILE"
echo "Store ranks: $STORE_RANKS" | tee -a "$LOG_FILE"
echo "Item ranks: $SKU_RANKS" | tee -a "$LOG_FILE"
echo "Feature ranks: $FEATURE_RANKS" | tee -a "$LOG_FILE"
echo "Max iter: $MAX_ITER" | tee -a "$LOG_FILE"
echo "Tolerance: $TOL" | tee -a "$LOG_FILE"
echo "Features: $FEATURES" | tee -a "$LOG_FILE"
echo "Output path: $OUTPUT_PATH" | tee -a "$LOG_FILE"
echo "Log level: $LOG_LEVEL" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"


# Run the script
set +e
python "${SCRIPT_DIR}/cluster_tensors.py" \
  --method "$METHOD" \
  --output_path "$OUTPUT_PATH" \
  --data_fn "$DATA_FN" \
  --rank_list "$RANK_LIST" \
  --store_ranks "$STORE_RANKS" \
  --sku_ranks "$SKU_RANKS" \
  --feature_ranks "$FEATURE_RANKS" \
  --max_iter "$MAX_ITER" \
  --tol "$TOL" \
  --features "$FEATURES" \
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
