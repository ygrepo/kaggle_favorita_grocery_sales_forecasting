# Model Types Refactoring

This document outlines the refactoring of model type constants to provide clearer separation between feedforward and sequence models.

## 🎯 **Changes Made**

### **1. Renamed MODEL_TYPES to FF_MODEL_TYPES**

**Before:**
```python
MODEL_TYPES = list(ModelType)
```

**After:**
```python
FF_MODEL_TYPES = list(ModelType)
SEQ_MODEL_TYPES = [
    SEQUENCE_MODEL.TFT,
    SEQUENCE_MODEL.NBEATS,
    SEQUENCE_MODEL.DEEPAR,
    SEQUENCE_MODEL.LSTM,
]
```

### **2. Updated All References**

| File | Change |
|------|--------|
| `src/model.py` | ✅ Added `FF_MODEL_TYPES` and `SEQ_MODEL_TYPES` |
| `src/model_utils.py` | ✅ Updated imports |
| `script/training.py` | ✅ `MODEL_TYPES` → `FF_MODEL_TYPES` |
| `script/train_sequence_models.py` | ✅ Updated to use `SEQ_MODEL_TYPES` |
| `script/unified_training_example.py` | ✅ Updated imports |
| `notebook/lr_finder.ipynb` | ✅ Updated to use `FF_MODEL_TYPES` |
| `notebook/unified_training_example.ipynb` | ✅ Updated imports |

## 📊 **Model Type Lists**

### **FF_MODEL_TYPES (Feedforward Models)**
```python
FF_MODEL_TYPES = [
    ModelType.SHALLOW_NN,      # "ShallowNN"
    ModelType.TWO_LAYER_NN,    # "TwoLayerNN" 
    ModelType.RESIDUAL_MLP,    # "ResidualMLP"
]
```

**Usage:**
```python
# Iterate over all feedforward models
for model_type in FF_MODEL_TYPES:
    train_model_unified(
        model_type=model_type,  # ModelType enum
        model_family="feedforward",
        # ... other parameters
    )
```

### **SEQ_MODEL_TYPES (Sequence Models)**
```python
SEQ_MODEL_TYPES = [
    SEQUENCE_MODEL.TFT,        # "TFT"
    SEQUENCE_MODEL.NBEATS,     # "NBEATS"
    SEQUENCE_MODEL.DEEPAR,     # "DEEPAR"
    SEQUENCE_MODEL.LSTM,       # "LSTM"
]
```

**Usage:**
```python
# Iterate over all sequence models
for model_type in SEQ_MODEL_TYPES:
    train_model_unified(
        model_type=model_type,  # String
        model_family="sequence",
        # ... other parameters
    )
```

## 🔧 **Key Differences**

| Aspect | FF_MODEL_TYPES | SEQ_MODEL_TYPES |
|--------|----------------|-----------------|
| **Type** | `List[ModelType]` (enums) | `List[str]` (strings) |
| **Usage** | `model_family="feedforward"` | `model_family="sequence"` |
| **Example** | `ModelType.SHALLOW_NN` | `SEQUENCE_MODEL.TFT` |
| **Count** | 3 models | 4 models |

## 🚀 **Benefits**

### **1. Clear Separation**
- **Feedforward models** have their own list
- **Sequence models** have their own list
- **No confusion** about which models belong to which family

### **2. Easy Iteration**
```python
# Train all feedforward models
for model_type in FF_MODEL_TYPES:
    train_feedforward_model(model_type)

# Train all sequence models  
for model_type in SEQ_MODEL_TYPES:
    train_sequence_model(model_type)
```

### **3. Type Safety**
- **FF_MODEL_TYPES** contains `ModelType` enums
- **SEQ_MODEL_TYPES** contains string constants
- **Clear distinction** in function signatures

### **4. Extensibility**
```python
# Easy to add new models
FF_MODEL_TYPES.append(ModelType.NEW_FF_MODEL)
SEQ_MODEL_TYPES.append(SEQUENCE_MODEL.NEW_SEQ_MODEL)
```

## 📁 **Files Updated**

### **Core Files**
- ✅ `src/model.py` - Added new constants
- ✅ `src/model_utils.py` - Updated imports

### **Scripts**
- ✅ `script/training.py` - Uses `FF_MODEL_TYPES`
- ✅ `script/train_sequence_models.py` - Uses `SEQ_MODEL_TYPES`
- ✅ `script/unified_training_example.py` - Updated imports
- ✅ `script/model_types_example.py` - New demo script

### **Notebooks**
- ✅ `notebook/lr_finder.ipynb` - Uses `FF_MODEL_TYPES`
- ✅ `notebook/unified_training_example.ipynb` - Updated imports

## 🔄 **Migration Guide**

### **For Feedforward Models**
```python
# Old way
from src.model import MODEL_TYPES
for model_type in MODEL_TYPES:
    # train model

# New way
from src.model import FF_MODEL_TYPES
for model_type in FF_MODEL_TYPES:
    # train model
```

### **For Sequence Models**
```python
# Old way
models = [SEQUENCE_MODEL.TFT, SEQUENCE_MODEL.NBEATS, ...]
for model_type in models:
    # train model

# New way
from src.model import SEQ_MODEL_TYPES
for model_type in SEQ_MODEL_TYPES:
    # train model
```

## 🎉 **Summary**

This refactoring provides:

1. **Clear naming**: `FF_MODEL_TYPES` vs `SEQ_MODEL_TYPES`
2. **Better organization**: Separate lists for different model families
3. **Type safety**: Enums vs strings are clearly distinguished
4. **Easier maintenance**: Adding new models is straightforward
5. **Consistent usage**: All scripts and notebooks updated

The codebase now has a clear separation between feedforward and sequence model types, making it easier to work with both model families independently or together.
