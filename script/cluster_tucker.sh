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

RANKS=12, 12, 40
FEATURES="gr_median,gr_std,gr_iqr,frac_up,frac_sideways,frac_down,up_to_down_ratio,ac_lag1,ac_lag4"
MAX_ITER=500
TOL=1e-5

LOG_DIR="${PROJECT_ROOT}/output/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_data_clustering.log"
LOG_LEVEL="INFO"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_fn) DATA_FN="$2"; shift 2 ;;
    --ranks) RANKS="$2"; shift 2 ;;
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
echo "Ranks: $RANKS" | tee -a "$LOG_FILE"
echo "Max iter: $MAX_ITER" | tee -a "$LOG_FILE"
echo "Tolerance: $TOL" | tee -a "$LOG_FILE"
echo "Features: $FEATURES" | tee -a "$LOG_FILE"
echo "Log level: $LOG_LEVEL" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"


# Run the script
set +e
python "${SCRIPT_DIR}/cluster_tucker.py" \
  --data_fn "$DATA_FN" \
  --ranks "$RANKS" \
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
