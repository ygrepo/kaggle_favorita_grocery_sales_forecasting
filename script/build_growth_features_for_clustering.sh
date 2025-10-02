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

DATA_FN="${OUTPUT_DATA_DIR}/2014_January_top_53_store_2000_item_weekly_growth_rate.parquet"

OUTPUT_CLUSTER_FN="${OUTPUT_DATA_DIR}/2014_January_top_53_store_2000_item_growth_rate_input_clustered.parquet"
OUTPUT_FEATURES_FN="${OUTPUT_DATA_DIR}/2014_January_top_53_store_2000_item_growth_rate_features.parquet"

TAU=0.05
INCLUDE_PCA_SMOOTHED=False
PCA_COMPONENTS=4
SMOOTH_WINDOW=4
KEYS="store_item"


LOG_DIR="${PROJECT_ROOT}/output/logs"
LOG_FILE="${LOG_DIR}/growth_rate_$(date +"%Y%m%d_%H%M%S").log"
LOG_LEVEL="DEBUG"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    --data_fn) DATA_FN="$2"; shift 2 ;;
    --output_cluster_fn) OUTPUT_CLUSTER_FN="$2"; shift 2 ;;
    --output_features_fn) OUTPUT_FEATURES_FN="$2"; shift 2 ;;
    --output_data_dir) OUTPUT_DATA_DIR="$2"; shift 2 ;;
    --tau) TAU="$2"; shift 2 ;;
    --include_pca_smoothed) INCLUDE_PCA_SMOOTHED="$2"; shift 2 ;;
    --pca_components) PCA_COMPONENTS="$2"; shift 2 ;;
    --smooth_window) SMOOTH_WINDOW="$2"; shift 2 ;;
    --keys) KEYS="$2"; shift 2 ;;
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
echo "Output cluster fn: $OUTPUT_CLUSTER_FN" | tee -a "$LOG_FILE"
echo "Output features fn: $OUTPUT_FEATURES_FN" | tee -a "$LOG_FILE"
echo "Tau: $TAU" | tee -a "$LOG_FILE"
echo "Include PCA smoothed: $INCLUDE_PCA_SMOOTHED" | tee -a "$LOG_FILE"
echo "PCA components: $PCA_COMPONENTS" | tee -a "$LOG_FILE"
echo "Smooth window: $SMOOTH_WINDOW" | tee -a "$LOG_FILE"
echo "Keys: $KEYS" | tee -a "$LOG_FILE"
echo "Log level: $LOG_LEVEL" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"

# Run the script
set +e
python "${SCRIPT_DIR}/build_growth_features_for_clustering.py" \
  --data_fn "$DATA_FN" \
  --output_cluster_fn "$OUTPUT_CLUSTER_FN" \
  --output_features_fn "$OUTPUT_FEATURES_FN" \
  --tau "$TAU" \
  --include_pca_smoothed "$INCLUDE_PCA_SMOOTHED" \
  --pca_components "$PCA_COMPONENTS" \
  --smooth_window "$SMOOTH_WINDOW" \
  --keys "$KEYS" \
  --log_level "$LOG_LEVEL" \
  --log_fn "$LOG_FILE"
exit_code=$?
set -e

if [[ ${exit_code} -eq 0 ]]; then
    echo "OK: finished at $(date)" | tee -a "${LOG_FILE}"
else
    echo "Python script failed with exit code $exit_code" >&2
    exit ${exit_code}
fi
