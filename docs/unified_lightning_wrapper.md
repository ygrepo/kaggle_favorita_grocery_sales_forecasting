# Unified LightningWrapper for Consistent Metrics

This document outlines the enhancement of `LightningWrapper` to support both feedforward and sequence models with consistent metrics and logging.

## 🎯 **Goal**

Enable sequence models to use the same metrics, logging, and monitoring as feedforward models by wrapping them in the enhanced `LightningWrapper`.

## 🔧 **Key Changes Made**

### **1. Enhanced LightningWrapper Constructor**

**Added `is_sequence_model` parameter:**
```python
def __init__(
    self,
    model: nn.Module,
    model_name: str,
    store: int,
    item: int,
    sales_idx: List[int],
    train_mav: float,
    val_mav: float,
    *,
    lr: float = 3e-4,
    log_level: str = "INFO",
    is_sequence_model: bool = False,  # ← New parameter
):
```

### **2. Enhanced training_step Method**

**Handles both model types:**
```python
def training_step(self, batch, batch_idx):
    if self.is_sequence_model:
        # Delegate to sequence model's training_step
        result = self.model.training_step(batch, batch_idx)
        
        # Extract loss and predictions for metrics
        if isinstance(result, dict):
            loss = result.get('loss', result.get('train_loss', None))
            # Try to get predictions for metrics
            try:
                with torch.no_grad():
                    preds = self.model(batch)
                    if isinstance(preds, dict):
                        preds = preds.get('prediction', preds.get('output', None))
            except Exception:
                preds = None
        else:
            loss = result
            preds = None
        
        # Extract targets from sequence batch format
        if hasattr(batch, 'target') or (isinstance(batch, dict) and 'target' in batch):
            yb = batch.target if hasattr(batch, 'target') else batch['target']
            wb = torch.ones_like(yb)  # Default weights
        else:
            yb = None
            wb = None
    else:
        # Original feedforward logic
        xb, yb, wb = batch
        preds = self.model(xb)
        loss = self.loss_fn(preds, yb, wb)
    
    # Common metric logging (works for both model types)
    self.log("train_loss", loss, ...)
    
    # Update metrics only if we have predictions and targets
    if preds is not None and yb is not None:
        self.train_mae_metric.update(preds, yb)
        self.train_rmse_metric.update(preds, yb)
        # ... log metrics
```

### **3. Enhanced validation_step Method**

**Similar dual handling:**
```python
def validation_step(self, batch, batch_idx):
    if self.is_sequence_model:
        # Delegate to sequence model's validation_step
        result = self.model.validation_step(batch, batch_idx)
        # ... extract loss, predictions, targets
    else:
        # Original feedforward logic
        xb, yb, wb = batch
        preds = self.model(xb)
        loss = self.loss_fn(preds, yb, wb)
    
    # Common metric logging and updates
    # ...
```

### **4. Updated train_model_unified Function**

**Now wraps ALL models in LightningWrapper:**
```python
def train_model_unified(model_type: MODEL_TYPE, ...):
    # ... common setup and data loading
    
    if model_type in FF_MODEL_TYPES:
        # Feedforward: create model + wrap
        base_model = model_factory(...)
        lightning_model = LightningWrapper(
            base_model, ..., is_sequence_model=False
        )
        
    elif model_type in SEQ_MODEL_TYPES:
        # Sequence: create model + wrap for consistent metrics
        base_model = model_factory(...)
        lightning_model = LightningWrapper(
            base_model, ..., is_sequence_model=True  # ← Key difference
        )
    
    # Same training for both
    trainer.fit(lightning_model, train_loader, val_loader)
```

## 🚀 **Benefits**

### **1. Consistent Metrics Across All Models**
- **Same MAE/RMSE metrics** for both feedforward and sequence models
- **Same logging format** and frequency
- **Same progress bar** and monitoring

### **2. Unified Training Interface**
- **Single LightningWrapper** handles both model types
- **Same trainer setup** for all models
- **Consistent callbacks** and checkpointing

### **3. Flexible Metric Computation**
- **Graceful handling** when predictions aren't available
- **Dummy MAV values** for sequence models (since they have different data formats)
- **Conditional metric updates** only when data is available

### **4. Preserved Model Functionality**
- **Sequence models** retain their native training/validation logic
- **Feedforward models** use the original optimized logic
- **No interference** with model-specific optimizations

## 📊 **Metric Handling Strategy**

### **Feedforward Models**
```python
# Standard batch format: (xb, yb, wb)
xb, yb, wb = batch
preds = self.model(xb)
loss = self.loss_fn(preds, yb, wb)

# Direct metric computation
self.train_mae_metric.update(preds, yb)
```

### **Sequence Models**
```python
# Delegate to model's native training
result = self.model.training_step(batch, batch_idx)
loss = result.get('loss', ...)

# Try to extract predictions for metrics
try:
    preds = self.model(batch)
    yb = batch.target  # Different batch format
    if preds is not None and yb is not None:
        self.train_mae_metric.update(preds, yb)
except:
    # Skip metrics if extraction fails
    pass
```

## 🎯 **Usage Examples**

### **Feedforward Model (unchanged)**
```python
train_model_unified(
    model_type=MODEL_TYPE.SHALLOW_NN,
    # ... parameters
)
# → Uses LightningWrapper with is_sequence_model=False
```

### **Sequence Model (now with consistent metrics)**
```python
train_model_unified(
    model_type=MODEL_TYPE.TFT,
    # ... parameters  
)
# → Uses LightningWrapper with is_sequence_model=True
# → Same metrics and logging as feedforward models!
```

## 📈 **Result**

Now both feedforward and sequence models have:

1. ✅ **Same metrics**: MAE, RMSE, loss tracking
2. ✅ **Same logging**: CSV logs, progress bars, checkpoints
3. ✅ **Same interface**: Single training function, consistent parameters
4. ✅ **Same monitoring**: Learning rate scheduling, early stopping
5. ✅ **Preserved functionality**: Each model type retains its optimizations

This creates a truly unified training experience with consistent metrics and monitoring across all model types! 🎉
