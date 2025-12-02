#!/bin/bash

# Set strict error handling
set -euo pipefail

export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32


# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Default configuration
DATA_FN="${PROJECT_ROOT}/output/data/2013_2014_store_2000_item_cyc_features.parquet"
STORE_MEDIAN_FN="${PROJECT_ROOT}/output/data/2013_2014_store_2000_item_cyc_features_with_store_medians.parquet"
STORE_ASSIGN_FN="${PROJECT_ROOT}/output/data/20251124_store_assignments.csv"
ITEM_MEDIAN_FN="${PROJECT_ROOT}/output/data/2013_2014_store_2000_item_cyc_features_with_item_medians.parquet"
ITEM_ASSIGN_FN="${PROJECT_ROOT}/output/data/20251124_item_assignments.csv"

DATE=$(date +"%Y%m%d")

MODEL_DIR="${PROJECT_ROOT}/output/models"
mkdir -p "$MODEL_DIR"

# Multiple models by default
#MODELS="TFT"
#MODELS="EXPONENTIAL_SMOOTHING,AUTO_ARIMA,THETA,KALMAN,XGBOOST,RANDOM_FOREST,LINEAR_REGRESSION"
MODELS="LIGHTGBM,NBEATS,TFT,TSMIXER,TCN,BLOCK_RNN,TIDE"

METRICS_DIR=""
METRICS_FN=""

SPLIT_POINT=0.8
MIN_TRAIN_DATA_POINTS=15
N=100
SAMPLE="True"
N_EPOCHS=300
BATCH_SIZE=64
NUM_WORKERS=8
DROPOUT=0.5
PATIENCE=10
PAST_COVS="True"
FUTURE_COVS="True"
XL_DESIGN="False"

LOG_DIR="${PROJECT_ROOT}/output/logs"
LOG_LEVEL="DEBUG"

# ------------------------------
# Parse command line arguments
# ------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_fn) DATA_FN="$2"; shift 2 ;;
    --store_median_fn) STORE_MEDIAN_FN="$2"; shift 2 ;;
    --store_assign_fn) STORE_ASSIGN_FN="$2"; shift 2 ;;
    --item_median_fn) ITEM_MEDIAN_FN="$2"; shift 2 ;;
    --item_assign_fn) ITEM_ASSIGN_FN="$2"; shift 2 ;;
    --model_dir) MODEL_DIR="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --metrics_fn) METRICS_FN="$2"; shift 2 ;;
    --metrics_dir) METRICS_DIR="$2"; shift 2 ;;
    --split_point) SPLIT_POINT="$2"; shift 2 ;;
    --min_train_data_points) MIN_TRAIN_DATA_POINTS="$2"; shift 2 ;;
    --N) N="$2"; shift 2 ;;
    --sample) SAMPLE="$2"; shift 2 ;;
    --n_epochs) N_EPOCHS="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --dropout) DROPOUT="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --past_covs) PAST_COVS="$2"; shift 2 ;;
    --future_covs) FUTURE_COVS="$2"; shift 2 ;;
    --xl_design) XL_DESIGN="$2"; shift 2 ;;
    --log_dir) LOG_DIR="$2"; shift 2 ;;
    --log_level) LOG_LEVEL="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# ------------------------------
# Finalize directories now that args are parsed
# ------------------------------

# Ensure base model/log dirs exist (MODEL_DIR may have been overridden)
mkdir -p "$MODEL_DIR"
mkdir -p "$LOG_DIR"

# Normalize models name for use in paths: "NBEATS,TFT" → "NBEATS_TFT"
MODELS_SAFE_NAME=$(echo "$MODELS" | tr ',' '_')

# If metrics_dir wasn't explicitly set, derive it from MODELS
if [[ -z "$METRICS_DIR" ]]; then
  METRICS_DIR="${PROJECT_ROOT}/output/metrics/${MODELS_SAFE_NAME}"
fi
mkdir -p "$METRICS_DIR"

# If metrics_fn wasn't explicitly set, derive it from METRICS_DIR
if [[ -z "$METRICS_FN" ]]; then
  METRICS_FN="${METRICS_DIR}/${DATE}_2013_2014_store_2000_item_cyc_features_all_dl_past_future_covs_metrics.csv"
fi

# Create separate MODEL_DIRS for each model type
IFS=',' read -ra MODEL_LIST <<< "$MODELS"
for MODEL_NAME in "${MODEL_LIST[@]}"; do
    MODEL_NAME_TRIMMED=$(echo "$MODEL_NAME" | xargs)  # trim spaces
    MODEL_SUBDIR="${MODEL_DIR}/${MODEL_NAME_TRIMMED}"
    mkdir -p "$MODEL_SUBDIR"
    echo "Created model subdirectory: $MODEL_SUBDIR"
done

# ------------------------------
# Logging setup
# ------------------------------

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_train_forecasting_models.log"

echo "Starting script at $(date)" | tee -a "$LOG_FILE"
echo "Starting script with the following configuration:" | tee -a "$LOG_FILE"
echo "Project root: $PROJECT_ROOT" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Log level: $LOG_LEVEL" | tee -a "$LOG_FILE"
echo "Data fn: $DATA_FN" | tee -a "$LOG_FILE"
echo "Store median fn: $STORE_MEDIAN_FN" | tee -a "$LOG_FILE"
echo "Store assign fn: $STORE_ASSIGN_FN" | tee -a "$LOG_FILE"
echo "Item median fn: $ITEM_MEDIAN_FN" | tee -a "$LOG_FILE"
echo "Item assign fn: $ITEM_ASSIGN_FN" | tee -a "$LOG_FILE"
echo "Models: $MODELS" | tee -a "$LOG_FILE"
echo "Batch size: $BATCH_SIZE" | tee -a "$LOG_FILE"
echo "Model dir (parent): $MODEL_DIR" | tee -a "$LOG_FILE"
echo "N epochs: $N_EPOCHS" | tee -a "$LOG_FILE"
echo "N: $N" | tee -a "$LOG_FILE"
echo "Sample: $SAMPLE" | tee -a "$LOG_FILE"
echo "Dropout: $DROPOUT" | tee -a "$LOG_FILE"
echo "Patience: $PATIENCE" | tee -a "$LOG_FILE"
echo "XL design: $XL_DESIGN" | tee -a "$LOG_FILE"
echo "Past covs: $PAST_COVS" | tee -a "$LOG_FILE"
echo "Future covs: $FUTURE_COVS" | tee -a "$LOG_FILE"
echo "Metrics dir: $METRICS_DIR" | tee -a "$LOG_FILE"
echo "Metrics fn: $METRICS_FN" | tee -a "$LOG_FILE"

# ------------------------------
# GPU setup
# ------------------------------

# Explicitly expose GPUs 0 and 1
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1

# ------------------------------
# Run Python script
# ------------------------------

set +e  # allow error handling below

python "${SCRIPT_DIR}/train_forecasting_models.py" \
  --data_fn "$DATA_FN" \
  --store_medians_fn "$STORE_MEDIAN_FN" \
  --store_assign_fn "$STORE_ASSIGN_FN" \
  --item_medians_fn "$ITEM_MEDIAN_FN" \
  --item_assign_fn "$ITEM_ASSIGN_FN" \
  --model_dir "$MODEL_DIR" \
  --models "$MODELS" \
  --metrics_fn "$METRICS_FN" \
  --split_point "$SPLIT_POINT" \
  --min_train_data_points "$MIN_TRAIN_DATA_POINTS" \
  --N "$N" \
  --sample "$SAMPLE" \
  --log_fn "$LOG_FILE" \
  --log_level "$LOG_LEVEL" \
  --n_epochs "$N_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --dropout "$DROPOUT" \
  --patience "$PATIENCE" \
  --past_covs "$PAST_COVS" \
  --future_covs "$FUTURE_COVS" \
  --xl_design "$XL_DESIGN"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "Script completed successfully at $(date)" | tee -a "$LOG_FILE"
    exit 0
else
    echo "Error: Script failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
    echo "Check the log file for details: $LOG_FILE"
    exit $EXIT_CODE
fi
