# Sequence Model Integration Fixes

This document outlines the fixes made to properly integrate sequence models from pytorch_forecasting into the unified model system.

## 🐛 **Issues Encountered**

### **1. AttributeError: 'TimeSeriesDataSet' object has no attribute 'dataset'**

**Problem:**
```python
# Incorrect - trying to access nested dataset
training_dataset = train_loader.dataset.dataset  # ❌ AttributeError
```

**Root Cause:**
The TimeSeriesDataSet from pytorch_forecasting is the training dataset itself, not a wrapper around another dataset.

**Fix:**
```python
# Correct - TimeSeriesDataSet is the training dataset
training_dataset = train_loader.dataset  # ✅ This is the TimeSeriesDataSet
```

### **2. TypeError: 'int' object is not iterable**

**Problem:**
```python
# TFT model expected output_size as list for multiple targets
return TemporalFusionTransformer.from_dataset(
    training_dataset,
    output_size=output_dim,  # ❌ int when list expected
    # ...
)
```

**Root Cause:**
The TFT model's `output_size` parameter caused issues when passed as an integer. The pytorch_forecasting models automatically infer the output size from the dataset.

**Fix:**
```python
# Remove output_size parameter - let model infer from dataset
return TemporalFusionTransformer.from_dataset(
    training_dataset,
    # output_size removed ✅
    learning_rate=learning_rate,
    hidden_size=hidden_dim,
    # ...
)
```

### **3. Variable Naming Inconsistency**

**Problem:**
```python
# Inconsistent variable names
if model_type in FF_MODEL_TYPES:
    lightning_model = LightningWrapper(...)
elif model_type in SEQ_MODEL_TYPES:
    model = model_factory(...)  # ❌ Different variable name

trainer.fit(lightning_model, ...)  # ❌ lightning_model not defined for sequence
```

**Fix:**
```python
# Consistent variable naming
if model_type in FF_MODEL_TYPES:
    lightning_model = LightningWrapper(...)
elif model_type in SEQ_MODEL_TYPES:
    lightning_model = model_factory(...)  # ✅ Same variable name

trainer.fit(lightning_model, ...)  # ✅ Works for both
```

## 🔧 **Fixes Applied**

### **1. Fixed TimeSeriesDataSet Access**
```python
# Before
elif model_type in SEQ_MODEL_TYPES:
    training_dataset = train_loader.dataset.dataset  # ❌

# After
elif model_type in SEQ_MODEL_TYPES:
    training_dataset = train_loader.dataset  # ✅
```

### **2. Removed output_size Parameter**
```python
# Before - All sequence models
return TemporalFusionTransformer.from_dataset(
    training_dataset,
    output_size=output_dim,  # ❌ Removed
    # ...
)

# After - All sequence models
return TemporalFusionTransformer.from_dataset(
    training_dataset,
    # output_size parameter removed ✅
    learning_rate=learning_rate,
    hidden_size=hidden_dim,
    # ...
)
```

### **3. Consistent Variable Naming**
```python
# Before
if model_type in FF_MODEL_TYPES:
    lightning_model = LightningWrapper(...)
elif model_type in SEQ_MODEL_TYPES:
    model = model_factory(...)  # ❌ Inconsistent

# After
if model_type in FF_MODEL_TYPES:
    lightning_model = LightningWrapper(...)
elif model_type in SEQ_MODEL_TYPES:
    lightning_model = model_factory(...)  # ✅ Consistent
```

### **4. Updated Return Type Annotation**
```python
# Before
def model_factory(...) -> nn.Module:  # ❌ Too restrictive

# After
def model_factory(...) -> Union[nn.Module, pl.LightningModule]:  # ✅ Accurate
```

## 🎯 **Key Insights**

### **1. pytorch_forecasting Models Are Different**
- **Feedforward models**: Raw PyTorch modules that need LightningWrapper
- **Sequence models**: Already Lightning modules with built-in training logic
- **Dataset handling**: TimeSeriesDataSet is the training dataset, not a wrapper

### **2. Parameter Inference**
- **pytorch_forecasting models** automatically infer many parameters from the dataset
- **output_size** is determined from the dataset's target columns
- **Less manual parameter passing** needed for sequence models

### **3. Unified Interface Benefits**
- **Same variable name** (`lightning_model`) for both model types
- **Same trainer call** works for both feedforward and sequence models
- **Consistent error handling** and logging

## 🚀 **Final Working Flow**

```python
def train_model_unified(model_type: MODEL_TYPE, ...):
    # Common data loading
    train_loader = torch.load(...)
    val_loader = torch.load(...)
    
    # Unified model creation
    if model_type in FF_MODEL_TYPES:
        # Feedforward: create model + wrap in Lightning
        base_model = model_factory(model_type, input_dim=..., ...)
        lightning_model = LightningWrapper(base_model, ...)
        
    elif model_type in SEQ_MODEL_TYPES:
        # Sequence: create model directly (already Lightning)
        training_dataset = train_loader.dataset  # ✅ TimeSeriesDataSet
        lightning_model = model_factory(
            model_type=model_type,
            training_dataset=training_dataset,
            # No output_size needed ✅
            learning_rate=lr,
            hidden_dim=hidden_dim,
            # ...
        )
    
    # Common training (works for both)
    trainer = pl.Trainer(...)
    trainer.fit(lightning_model, train_loader, val_loader)  # ✅
```

## ✅ **Verification**

The fixes ensure that:
1. ✅ **TimeSeriesDataSet** is accessed correctly
2. ✅ **Sequence models** are created without parameter conflicts
3. ✅ **Variable naming** is consistent across both model types
4. ✅ **Type annotations** accurately reflect return types
5. ✅ **Training works** for both feedforward and sequence models

The unified model system now properly handles both feedforward and sequence models with a clean, consistent interface! 🎉
