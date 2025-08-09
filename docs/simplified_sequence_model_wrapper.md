# Simplified Sequence Model Wrapper Approach

This document explains the simplified approach to wrapping sequence models in `LightningWrapper` while avoiding logging conflicts.

## 🐛 **Issue Encountered**

When trying to extract metrics from sequence models wrapped in `LightningWrapper`, we encountered:

```
MisconfigurationException: You are trying to `self.log()` but it is not managed by the `Trainer` control flow
```

**Root Cause:**
- Sequence models from pytorch_forecasting have their own `self.log()` calls
- When wrapped in `LightningWrapper`, these models are no longer the top-level Lightning module
- Their `self.log()` calls fail because they're not in the proper trainer context

## 🔧 **Simplified Solution**

Instead of trying to extract metrics from sequence models, we use a **delegation approach**:

### **1. Complete Delegation for Sequence Models**

```python
def training_step(self, batch, batch_idx):
    if self.is_sequence_model:
        # Delegate completely to the underlying model
        # The sequence model handles its own logging and metrics
        return self.model.training_step(batch, batch_idx)
    else:
        # Original feedforward logic
        xb, yb, wb = batch
        preds = self.model(xb)
        loss = self.loss_fn(preds, yb, wb)
        # ... custom metrics and logging

def validation_step(self, batch, batch_idx):
    if self.is_sequence_model:
        # Delegate completely to the underlying model
        return self.model.validation_step(batch, batch_idx)
    else:
        # Original feedforward logic
        # ... custom metrics and logging
```

### **2. Conditional Metric Computation**

```python
def on_train_epoch_end(self):
    # Only compute custom metrics for feedforward models
    if not self.is_sequence_model:
        avg_train_mae = self.train_mae_metric.compute().item()
        # ... compute and log custom metrics

def on_validation_epoch_end(self):
    # Only compute custom metrics for feedforward models
    if not self.is_sequence_model:
        avg_val_mae = self.val_mae_metric.compute().item()
        # ... compute and log custom metrics
```

## 🎯 **What This Achieves**

### **1. Unified Interface**
- **Both model types** use the same `LightningWrapper`
- **Same training function** (`train_model_unified`) for both
- **Same trainer setup** and configuration

### **2. Preserved Functionality**
- **Feedforward models**: Get custom MAE, RMSE, %MAV metrics
- **Sequence models**: Use their native pytorch_forecasting metrics
- **No interference** with model-specific optimizations

### **3. Consistent Training Experience**
```python
# Same interface for both model types
train_model_unified(model_type=MODEL_TYPE.SHALLOW_NN, ...)  # Feedforward
train_model_unified(model_type=MODEL_TYPE.TFT, ...)         # Sequence

# Both use LightningWrapper but with different behavior
```

## 📊 **Metrics Comparison**

### **Feedforward Models (Custom Metrics)**
- ✅ **train_loss**, **val_loss**
- ✅ **train_mae**, **val_mae** 
- ✅ **train_rmse**, **val_rmse**
- ✅ **train_percent_mav**, **val_percent_mav**

### **Sequence Models (Native Metrics)**
- ✅ **train_loss**, **val_loss** (from pytorch_forecasting)
- ✅ **Native sequence metrics** (RMSE, MAE, etc. from pytorch_forecasting)
- ✅ **Model-specific metrics** (attention weights, quantile losses, etc.)

## 🚀 **Benefits**

### **1. No Logging Conflicts**
- **Sequence models** handle their own logging in proper context
- **No MisconfigurationException** errors
- **Clean separation** of concerns

### **2. Best of Both Worlds**
- **Feedforward models**: Custom metrics optimized for the use case
- **Sequence models**: Rich native metrics from pytorch_forecasting
- **Unified training**: Same interface and trainer setup

### **3. Maintainability**
- **Simple delegation** - no complex metric extraction
- **Clear separation** between model types
- **Easy to extend** for new model types

## 🔄 **Training Flow**

```python
def train_model_unified(model_type: MODEL_TYPE, ...):
    # Common setup
    # Common data loading
    
    if model_type in FF_MODEL_TYPES:
        # Create feedforward model + wrap with custom metrics
        base_model = model_factory(...)
        lightning_model = LightningWrapper(
            base_model, ..., is_sequence_model=False
        )
        
    elif model_type in SEQ_MODEL_TYPES:
        # Create sequence model + wrap with delegation
        base_model = model_factory(...)
        lightning_model = LightningWrapper(
            base_model, ..., is_sequence_model=True
        )
    
    # Same trainer for both
    trainer.fit(lightning_model, train_loader, val_loader)
```

## 📈 **Result**

This approach provides:

1. ✅ **Unified training interface** for all model types
2. ✅ **No logging conflicts** or trainer errors
3. ✅ **Preserved model functionality** for both types
4. ✅ **Appropriate metrics** for each model family
5. ✅ **Clean, maintainable code** with clear separation

The sequence models get their rich native metrics from pytorch_forecasting, while feedforward models get the custom metrics optimized for the specific use case. Both work seamlessly within the same training framework! 🎉
