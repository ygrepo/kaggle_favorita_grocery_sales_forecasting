#!/bin/bash

# Grid search hyperparameter optimization for time series forecasting models
# Reuses code from train_forecasting_models.py and train_forecasting_store_sku_models.py

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
DATA_FN="${PROJECT_ROOT}/data/favorita_processed.parquet"
STORE_MEDIANS_FN="${PROJECT_ROOT}/data/store_medians.parquet"
ITEM_MEDIANS_FN="${PROJECT_ROOT}/data/item_medians.parquet"
STORE_ASSIGN_FN="${PROJECT_ROOT}/data/store_cluster_assignment.parquet"
ITEM_ASSIGN_FN="${PROJECT_ROOT}/data/item_cluster_assignment.parquet"

MODEL_DIR="${PROJECT_ROOT}/models"
RESULTS_DIR="${PROJECT_ROOT}/output/grid_search"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FN="${RESULTS_DIR}/${TIMESTAMP}_grid_search_results.csv"
LOG_DIR="${PROJECT_ROOT}/output/logs"
mkdir -p "$LOG_DIR"
LOG_FN="${LOG_DIR}/${TIMESTAMP}_grid_search.log"

# Grid search parameters
MODELS="NBEATS"  # Change to TCN, TFT, etc. for other models
N=5  # Number of (store, item) pairs to process
SPLIT_POINT=0.8
MIN_TRAIN_DATA_POINTS=15
BATCH_SIZE=64
N_EPOCHS=500
DROPOUT=0.2
PAST_COVS="True"
FUTURE_COVS="True"
XL_DESIGN="False"
LOG_LEVEL="INFO"
N_TRIALS=10
TIMEOUT=0
SEED=42

echo "Starting grid search for model: $MODELS"
echo "Results will be saved to: $RESULTS_FN"
echo "Log file: $LOG_FN"

python3 script/grid_search_global_model.py \
    --data_fn "$DATA_FN" \
    --store_medians_fn "$STORE_MEDIANS_FN" \
    --item_medians_fn "$ITEM_MEDIANS_FN" \
    --store_assign_fn "$STORE_ASSIGN_FN" \
    --item_assign_fn "$ITEM_ASSIGN_FN" \
    --model_dir "$MODEL_DIR" \
    --results_fn "$RESULTS_FN" \
    --log_fn "$LOG_FN" \
    --log_level "$LOG_LEVEL" \
    --models "$MODELS" \
    --N "$N" \
    --split_point "$SPLIT_POINT" \
    --min_train_data_points "$MIN_TRAIN_DATA_POINTS" \
    --batch_size "$BATCH_SIZE" \
    --n_epochs "$N_EPOCHS" \
    --dropout "$DROPOUT" \
    --past_covs "$PAST_COVS" \
    --future_covs "$FUTURE_COVS" \
    --xl_design "$XL_DESIGN" \
    --n_trials "$N_TRIALS" \
    --timeout "$TIMEOUT" \
    --seed "$SEED"

echo "Grid search completed!"

