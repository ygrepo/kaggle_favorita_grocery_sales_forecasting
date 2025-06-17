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
DATA_FN="${DATA_DIR}/20250611_train_top_10_store_10_item_sales_cyclical_features_16_days_X_y.xlsx"
WEIGHTS_FN="${DATA_DIR}/top_10_item_weights.xlsx"
LOG_DIR="${PROJECT_ROOT}/output/logs"
MODEL_DIR="${PROJECT_ROOT}/output/models"
WINDOW_SIZE=16
BATCH_SIZE=32
LEARNING_RATE=0.001
EPOCHS=2
SEED=42
LOG_LEVEL="INFO"
VENV_PATH="${PROJECT_ROOT}/.venv"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-fn) DATA_FN="$2"; shift 2 ;;
    --weights-fn) WEIGHTS_FN="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --window-size) WINDOW_SIZE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --log-level) LOG_LEVEL="$2"; shift 2 ;;
    --venv) VENV_PATH="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# Create output directories if they don't exist
mkdir -p "$MODEL_DIR"
mkdir -p "$LOG_DIR"

# Set up log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

echo "Starting training at $(date)" | tee -a "$LOG_FILE"
echo "Project root: $PROJECT_ROOT" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE" | tee -a "$LOG_FILE"

# # Activate virtual environment if it exists
# if [ -f "${VENV_PATH}/bin/activate" ]; then
#     echo "Activating virtual environment at ${VENV_PATH}" | tee -a "$LOG_FILE"
#     source "${VENV_PATH}/bin/activate"
# else
#     echo "No virtual environment found at ${VENV_PATH}, using system Python" | tee -a "$LOG_FILE"
# fi

# # Check Python version
# PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
# echo "Using Python ${PYTHON_VERSION}" | tee -a "$LOG_FILE"

# # Install project in development mode
# echo "Installing project in development mode..." | tee -a "$LOG_FILE"
# pip install -e . 2>&1 | tee -a "$LOG_FILE"

# Run the training script
set +e  # Disable exit on error to handle the error message
echo "Starting training with the following configuration:" | tee -a "$LOG_FILE"
echo "  Data fn: ${DATA_FN}" | tee -a "$LOG_FILE"
echo "  Weights fn: ${WEIGHTS_FN}" | tee -a "$LOG_FILE"
echo "  Model directory: ${MODEL_DIR}" | tee -a "$LOG_FILE"
echo "  Window size: ${WINDOW_SIZE}" | tee -a "$LOG_FILE"
echo "  Batch size: ${BATCH_SIZE}" | tee -a "$LOG_FILE"
echo "  Learning rate: ${LEARNING_RATE}" | tee -a "$LOG_FILE"
echo "  Epochs: ${EPOCHS}" | tee -a "$LOG_FILE"
echo "  Random seed: ${SEED}" | tee -a "$LOG_FILE"

python "${SCRIPT_DIR}/run_training.py" \
  --data-fn "$DATA_FN" \
  --weights-fn "$WEIGHTS_FN" \
  --model-dir "$MODEL_DIR" \
  --window-size "$WINDOW_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --log-level "$LOG_LEVEL" 2>&1 | tee -a "$LOG_FILE"

# Check the exit status of the Python script
TRAINING_EXIT_CODE=${PIPESTATUS[0]}

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully at $(date)" | tee -a "$LOG_FILE"
    exit 0
else
    echo "Error: Training failed with exit code $TRAINING_EXIT_CODE" | tee -a "$LOG_FILE"
    echo "Check the log file for details: $LOG_FILE"
    exit $TRAINING_EXIT_CODE
fi
