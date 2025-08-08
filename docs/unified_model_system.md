# Unified Model System

This document outlines the new unified model system with a single `MODEL_TYPE` enum and unified `model_factory` function.

## 🎯 **Key Changes**

### **1. Single MODEL_TYPE Enum**

**Before:**
```python
class ModelType(Enum):  # Only feedforward
    SHALLOW_NN = "ShallowNN"
    TWO_LAYER_NN = "TwoLayerNN"
    RESIDUAL_MLP = "ResidualMLP"

class SEQUENCE_MODEL:  # Separate class
    TFT = "TFT"
    NBEATS = "NBEATS"
    DEEPAR = "DEEPAR"
    LSTM = "LSTM"
```

**After:**
```python
class MODEL_TYPE(str, Enum):
    # Feedforward models
    SHALLOW_NN = "ShallowNN"
    TWO_LAYER_NN = "TwoLayerNN"
    RESIDUAL_MLP = "ResidualMLP"
    
    # Sequence models
    TFT = "TFT"
    NBEATS = "NBEATS"
    DEEPAR = "DEEPAR"
    LSTM = "LSTM"
```

### **2. Unified model_factory Function**

**Before:**
```python
def model_factory(...)  # Only feedforward
def sequence_model_factory(...)  # Only sequence
```

**After:**
```python
def model_factory(
    model_type: MODEL_TYPE,
    # Feedforward parameters
    input_dim: int = None,
    hidden_dim: int = 128,
    h1: int = 64,
    h2: int = 32,
    depth: int = 3,
    output_dim: int = 1,
    dropout: float = 0.0,
    # Sequence model parameters
    training_dataset = None,
    learning_rate: float = 1e-3,
    attention_head_size: int = 4,
    hidden_continuous_size: int = 16,
    **kwargs
) -> nn.Module:
    """Unified factory for both feedforward and sequence models."""
```

### **3. Auto-Detection in train_model_unified**

**Before:**
```python
def train_model_unified(
    model_type: ModelType,
    model_family: str,  # Required parameter
    ...
)
```

**After:**
```python
def train_model_unified(
    model_type: MODEL_TYPE,  # Auto-detects family
    ...
):
    # Auto-detect model family
    if is_feedforward_model(model_type):
        model_family = "feedforward"
    elif is_sequence_model(model_type):
        model_family = "sequence"
```

## 🚀 **Usage Examples**

### **Feedforward Models**
```python
from src.model import MODEL_TYPE
from src.model_utils import train_model_unified

# Train a feedforward model
train_model_unified(
    model_type=MODEL_TYPE.SHALLOW_NN,  # Auto-detected as feedforward
    model_dir=model_dir,
    dataloader_dir=feedforward_dataloader_dir,
    # ... other parameters
)
```

### **Sequence Models**
```python
# Train a sequence model
train_model_unified(
    model_type=MODEL_TYPE.TFT,  # Auto-detected as sequence
    model_dir=model_dir,
    dataloader_dir=sequence_dataloader_dir,
    # ... other parameters
)
```

### **Using the Unified Factory Directly**
```python
from src.model import model_factory, MODEL_TYPE

# Create a feedforward model
ff_model = model_factory(
    model_type=MODEL_TYPE.SHALLOW_NN,
    input_dim=100,
    hidden_dim=64,
    output_dim=3,
    dropout=0.2
)

# Create a sequence model
seq_model = model_factory(
    model_type=MODEL_TYPE.TFT,
    training_dataset=dataset,
    learning_rate=1e-3,
    hidden_dim=32,
    attention_head_size=4
)
```

## 📊 **Model Type Lists (Convenience)**

```python
# Still available for convenience
FF_MODEL_TYPES = [
    MODEL_TYPE.SHALLOW_NN,
    MODEL_TYPE.TWO_LAYER_NN,
    MODEL_TYPE.RESIDUAL_MLP,
]

SEQ_MODEL_TYPES = [
    MODEL_TYPE.TFT,
    MODEL_TYPE.NBEATS,
    MODEL_TYPE.DEEPAR,
    MODEL_TYPE.LSTM,
]

# Helper functions
def is_feedforward_model(model_type: MODEL_TYPE) -> bool:
    return model_type in FF_MODEL_TYPES

def is_sequence_model(model_type: MODEL_TYPE) -> bool:
    return model_type in SEQ_MODEL_TYPES
```

## 🔧 **Benefits**

### **1. Simplicity**
- **One enum** for all model types
- **One factory** for all models
- **Auto-detection** eliminates manual specification

### **2. Consistency**
- **Same interface** for all models
- **Unified parameter handling**
- **Consistent naming** across the codebase

### **3. Extensibility**
```python
# Easy to add new models
class MODEL_TYPE(str, Enum):
    # ... existing models
    NEW_FF_MODEL = "NewFFModel"
    NEW_SEQ_MODEL = "NewSeqModel"

# Update lists
FF_MODEL_TYPES.append(MODEL_TYPE.NEW_FF_MODEL)
SEQ_MODEL_TYPES.append(MODEL_TYPE.NEW_SEQ_MODEL)

# Add to factory
def model_factory(model_type: MODEL_TYPE, ...):
    # ... existing cases
    elif model_type == MODEL_TYPE.NEW_FF_MODEL:
        return NewFFModel(...)
    elif model_type == MODEL_TYPE.NEW_SEQ_MODEL:
        return NewSeqModel(...)
```

### **4. Type Safety**
- **Single source of truth** for model types
- **IDE autocomplete** for all models
- **Compile-time checking** of model types

## 📁 **Files Updated**

### **Core Files**
- ✅ `src/model.py` - Unified MODEL_TYPE and model_factory
- ✅ `src/model_utils.py` - Auto-detection in train_model_unified

### **Scripts**
- ✅ `script/training.py` - Uses MODEL_TYPE
- ✅ `script/train_sequence_models.py` - Uses MODEL_TYPE
- ✅ `script/unified_training_example.py` - Uses MODEL_TYPE

### **Notebooks**
- ✅ `notebook/lr_finder.ipynb` - Uses MODEL_TYPE
- ✅ `notebook/unified_training_example.ipynb` - Uses MODEL_TYPE

## 🎉 **Summary**

The unified model system provides:

1. **Single MODEL_TYPE enum** for all models
2. **Unified model_factory** function
3. **Auto-detection** of model families
4. **Simplified API** with fewer parameters
5. **Better maintainability** and extensibility

This creates a clean, consistent interface for working with both feedforward and sequence models while maintaining all existing functionality.
