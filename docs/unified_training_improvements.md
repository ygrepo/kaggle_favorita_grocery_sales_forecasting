# Unified Training Improvements

This document outlines the improvements made to the training system to provide consistent metrics tracking and logging for both feedforward and sequence models.

## 🎯 **Key Improvements**

### 1. **Consistent Metric Tracking**
Both feedforward and sequence models now track the same metrics:
- **Training/Validation Loss**
- **Mean Absolute Error (MAE)** in original units
- **Root Mean Squared Error (RMSE)** in original units  
- **Percent MAV** (Mean Absolute Value) - normalized performance metric

### 2. **Unified CSV Logging**
- **Removed separate history collection** - everything goes to CSV logs
- **Same CSV logger** used for both model types
- **Consistent file naming**: `{model_name}_{store_cluster}_{item_cluster}/version_X/metrics.csv`

### 3. **Sequence Model Wrapper**
Created `SequenceModelWrapper` class that:
- **Wraps pytorch_forecasting models** for consistent interface
- **Extracts predictions and targets** for metric computation
- **Handles tensor shape mismatches** gracefully
- **Provides same logging interface** as feedforward models

## 🔧 **Technical Changes**

### New Classes Added

#### `SequenceModelWrapper` (in `src/model.py`)
```python
class SequenceModelWrapper(pl.LightningModule):
    """Lightning wrapper for sequence models with consistent metric tracking"""
```

**Features:**
- Wraps any pytorch_forecasting model
- Tracks MAE, RMSE, and percent MAV metrics
- Handles different batch formats (dict, tuple, tensor)
- Graceful error handling for metric computation
- Delegates optimizer configuration to underlying model

### Updated Functions

#### `train_model_unified` (in `src/model_utils.py`)
**Changes:**
- **Sequence models** now wrapped with `SequenceModelWrapper`
- **CSV logger** added for sequence models (same as feedforward)
- **MAV computation** added for sequence models
- **Checkpoints and callbacks** consistent between model types
- **History collection removed** - all data goes to CSV logs

## 📊 **Metrics Comparison**

| Metric | Feedforward Models | Sequence Models | Status |
|--------|-------------------|-----------------|---------|
| Training Loss | ✅ | ✅ | **Consistent** |
| Validation Loss | ✅ | ✅ | **Consistent** |
| MAE (original units) | ✅ | ✅ | **Consistent** |
| RMSE (original units) | ✅ | ✅ | **Consistent** |
| Percent MAV | ✅ | ✅ | **Consistent** |
| Learning Rate | ✅ | ✅ | **Consistent** |
| CSV Logging | ✅ | ✅ | **Consistent** |
| Checkpoints | ✅ | ✅ | **Consistent** |

## 🚀 **Usage Examples**

### Feedforward Model (unchanged)
```python
train_model_unified(
    model_type=ModelType.TWO_LAYER_NN,  # Enum
    model_family="feedforward",
    # ... other parameters
)
```

### Sequence Model (improved)
```python
train_model_unified(
    model_type=SEQUENCE_MODEL.TFT,  # String
    model_family="sequence", 
    # ... other parameters
)
```

## 📁 **Output Structure**

Both model types now produce the same output structure:

```
output/
├── models/unified_demo/
│   └── checkpoints/
│       ├── 17_15_TwoLayerNN/
│       │   ├── 17_15_TwoLayerNN.ckpt
│       │   └── last.ckpt
│       └── 17_15_TFT/
│           ├── 17_15_TFT.ckpt
│           └── last.ckpt
└── logs/unified_demo/
    ├── 17_15_TwoLayerNN_17_15/
    │   └── version_0/
    │       └── metrics.csv
    └── 17_15_TFT_17_15/
        └── version_0/
            └── metrics.csv
```

## 📈 **CSV Log Contents**

Each `metrics.csv` file contains:
```csv
epoch,train_loss,val_loss,train_percent_mav,val_percent_mav,lr-Adam
0,1.234,1.456,15.2,18.7,0.0003
1,1.123,1.345,14.1,17.2,0.0003
2,1.012,1.234,13.5,16.8,0.0003
...
```

## 🔄 **Migration Guide**

### Before (separate systems)
- Feedforward models: CSV logs + manual history collection
- Sequence models: Basic logging only
- Different metrics tracked
- Inconsistent output formats

### After (unified system)
- Both model types: Consistent CSV logs
- Same metrics tracked for both
- No manual history collection needed
- Unified output structure
- Same analysis tools work for both

## 🎉 **Benefits**

1. **Simplified Analysis**: Same CSV format for all models
2. **Consistent Metrics**: Fair comparison between model types
3. **Reduced Code Duplication**: Single logging system
4. **Better Monitoring**: All models tracked the same way
5. **Easier Debugging**: Consistent error handling and logging
6. **Future-Proof**: Easy to add new sequence models

## 🔍 **Example Analysis**

With consistent CSV logging, you can now easily compare models:

```python
import pandas as pd

# Load logs from both model types
ff_logs = pd.read_csv("logs/17_15_TwoLayerNN_17_15/version_0/metrics.csv")
seq_logs = pd.read_csv("logs/17_15_TFT_17_15/version_0/metrics.csv")

# Compare final validation performance
print(f"Feedforward final val_loss: {ff_logs['val_loss'].iloc[-1]:.4f}")
print(f"Sequence final val_loss: {seq_logs['val_loss'].iloc[-1]:.4f}")

# Compare percent MAV (normalized performance)
print(f"Feedforward final val_percent_mav: {ff_logs['val_percent_mav'].iloc[-1]:.2f}%")
print(f"Sequence final val_percent_mav: {seq_logs['val_percent_mav'].iloc[-1]:.2f}%")
```

This unified system makes it much easier to compare and analyze different model architectures on the same dataset!
