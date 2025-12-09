#!/bin/bash

# Grid search hyperparameter optimization for time series forecasting models
# Reuses code from train_forecasting_models.py and train_forecasting_store_sku_models.py

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default configuration
DATA_FN="${PROJECT_ROOT}/output/data/2013_2014_store_2000_item_cyc_features.parquet"


MODEL_DIR="${PROJECT_ROOT}/models"
RESULTS_DIR="${PROJECT_ROOT}/output/grid_search"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/output/logs"
mkdir -p "$LOG_DIR"
LOG_FN="${LOG_DIR}/${TIMESTAMP}_grid_search.log"

# Grid search parameters
MODELS="TIDE"  # Change to TCN, TFT, etc. for other models
RESULTS_FN="${RESULTS_DIR}/${TIMESTAMP}_${MODELS}_grid_search_results.csv"
N=10  # Number of (store, item) pairs to process
SPLIT_POINT=0.8
MIN_TRAIN_DATA_POINTS=15
PAST_COVS="True"
FUTURE_COVS="True"
XL_DESIGN="False"
LOG_LEVEL="INFO"
N_TRIALS=30
TIMEOUT=3600
SEED=42

echo "Starting grid search for model: $MODELS"
echo "Results will be saved to: $RESULTS_FN"
echo "Log file: $LOG_FN"

python3 script/grid_search_global_model.py \
    --data_fn "$DATA_FN" \
    --model_dir "$MODEL_DIR" \
    --results_fn "$RESULTS_FN" \
    --log_fn "$LOG_FN" \
    --log_level "$LOG_LEVEL" \
    --models "$MODELS" \
    --N "$N" \
    --split_point "$SPLIT_POINT" \
    --min_train_data_points "$MIN_TRAIN_DATA_POINTS" \
    --past_covs "$PAST_COVS" \
    --future_covs "$FUTURE_COVS" \
    --xl_design "$XL_DESIGN" \
    --n_trials "$N_TRIALS" \
    --timeout "$TIMEOUT" \
    --seed "$SEED"

echo "Grid search completed!"

