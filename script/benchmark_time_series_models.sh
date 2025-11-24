#!/bin/bash

set -euo pipefail

# export OMP_NUM_THREADS=32
# export MKL_NUM_THREADS=32
# export NUMEXPR_NUM_THREADS=32

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

DATA_FN="${PROJECT_ROOT}/output/data/2013_2014_store_2000_item_cyc_features.parquet"
DATE=$(date +"%Y%m%d")

METRICS_DIR="${PROJECT_ROOT}/output/metrics"
mkdir -p "$METRICS_DIR"
METRICS_FN="${METRICS_DIR}/${DATE}_2013_2014_store_2000_item_cyc_features_ml_metrics.csv"

SPLIT_POINT=0.8
MIN_TRAIN_DATA_POINTS=15
MODELS="EXPONENTIAL_SMOOTHING,AUTO_ARIMA,THETA,KALMAN"
N=10
LOG_DIR="${PROJECT_ROOT}/output/logs"
LOG_LEVEL="DEBUG"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_fn) DATA_FN="$2"; shift 2 ;;
    --metrics_fn) METRICS_FN="$2"; shift 2 ;;
    --split_point) SPLIT_POINT="$2"; shift 2 ;;
    --min_train_data_points) MIN_TRAIN_DATA_POINTS="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --N) N="$2"; shift 2 ;;
    --log_dir) LOG_DIR="$2"; shift 2 ;;
    --log_level) LOG_LEVEL="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_benchmark_time_series_models.log"

echo "Starting script at $(date)" | tee -a "$LOG_FILE"
echo "Project root: $PROJECT_ROOT" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Log level: $LOG_LEVEL" | tee -a "$LOG_FILE"
echo "Data fn: $DATA_FN" | tee -a "$LOG_FILE"
echo "Split point: $SPLIT_POINT" | tee -a "$LOG_FILE"
echo "Min train data points: $MIN_TRAIN_DATA_POINTS" | tee -a "$LOG_FILE"
echo "Models: $MODELS" | tee -a "$LOG_FILE"
echo "N: $N" | tee -a "$LOG_FILE"
echo "Metrics fn: $METRICS_FN" | tee -a "$LOG_FILE"

set +e
python "${SCRIPT_DIR}/benchmark_time_series_models.py" \
  --data_fn "$DATA_FN" \
  --metrics_fn "$METRICS_FN" \
  --split_point "$SPLIT_POINT" \
  --min_train_data_points "$MIN_TRAIN_DATA_POINTS" \
  --models "$MODELS" \
  --N "$N" \
  --log_fn "$LOG_FILE" \
  --log_level "$LOG_LEVEL"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Script completed successfully at $(date)" | tee -a "$LOG_FILE"
    exit 0
else
    echo "Error: Script failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
    echo "Check the log file for details: $LOG_FILE"
    exit $EXIT_CODE
fi