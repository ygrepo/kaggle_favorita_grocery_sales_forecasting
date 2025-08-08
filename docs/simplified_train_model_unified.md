# Simplified train_model_unified Function

You're absolutely right! The `train_model_unified` function can be greatly simplified by sharing common code and isolating only the differences between feedforward and sequence models.

## 🎯 **Current Issues**

The current function has:
- **Duplicated setup code** (checkpoints, callbacks, trainer, CSV logger)
- **Duplicated data loading patterns**
- **Mixed model-specific and common logic**
- **~280 lines** of mostly repetitive code

## 🚀 **Simplified Approach**

Here's how to refactor it:

### **1. Extract Helper Functions**

```python
def _load_feedforward_data(dataloader_dir, store_cluster, item_cluster, num_workers, persistent_workers, logger):
    """Load feedforward model data and return train/val loaders or None if insufficient data."""
    # Load metadata and check for sufficient data
    train_meta_fn = dataloader_dir / f"{store_cluster}_{item_cluster}_train_meta.parquet"
    val_meta_fn = dataloader_dir / f"{store_cluster}_{item_cluster}_val_meta.parquet"
    
    meta_df = pd.read_parquet(train_meta_fn)
    val_meta_df = pd.read_parquet(val_meta_fn)

    if meta_df.empty or val_meta_df.empty:
        logger.warning(f"Skipping pair ({store_cluster}, {item_cluster}) due to insufficient data.")
        return None, None

    # Load and configure dataloaders
    train_loader = torch.load(dataloader_dir / f"{store_cluster}_{item_cluster}_train_loader.pt", weights_only=False)
    val_loader = torch.load(dataloader_dir / f"{store_cluster}_{item_cluster}_val_loader.pt", weights_only=False)

    # Recreate with proper settings
    batch_size = train_loader.batch_size or 32
    train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=True)
    val_loader = DataLoader(val_loader.dataset, batch_size=batch_size, num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=True)
    
    return train_loader, val_loader

def _load_sequence_data(dataloader_dir, store_cluster, item_cluster, logger):
    """Load sequence model data and return train/val loaders and training dataset."""
    train_loader = torch.load(dataloader_dir / f"{store_cluster}_{item_cluster}_train_loader.pt", weights_only=False)
    val_loader = torch.load(dataloader_dir / f"{store_cluster}_{item_cluster}_val_loader.pt", weights_only=False)
    training_dataset = train_loader.dataset.dataset  # underlying TimeSeriesDataSet
    
    return train_loader, val_loader, training_dataset

def _get_input_dim(train_loader):
    """Get input dimension from train loader."""
    if hasattr(train_loader.dataset, "tensors"):
        return train_loader.dataset.tensors[0].shape[1]
    else:
        sample_batch = next(iter(train_loader))
        return sample_batch[0].shape[1]

def _create_common_trainer_components(model_name, store_cluster, item_cluster, checkpoints_dir, model_logger_dir, epochs, enable_progress_bar):
    """Create common trainer components (callbacks, logger, trainer)."""
    # Setup checkpoints and callbacks
    checkpoint_dir = checkpoints_dir / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1, save_last=True,
        dirpath=checkpoint_dir, filename=model_name
    )
    
    early_stop = EarlyStopping(monitor="val_loss", patience=2, mode="min", min_delta=1e-4)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # Setup CSV logger
    csv_logger_name = f"{model_name}_{store_cluster}_{item_cluster}"
    csv_logger = CSVLogger(name=csv_logger_name, save_dir=model_logger_dir)
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator=get_device(),
        deterministic=True,
        max_epochs=epochs,
        logger=csv_logger,
        enable_progress_bar=enable_progress_bar,
        callbacks=[checkpoint_callback, early_stop, lr_monitor],
    )
    
    return trainer
```

### **2. Simplified Main Function**

```python
def train_model_unified(
    model_dir: Path,
    dataloader_dir: Path,
    model_logger_dir: Path,
    model_type,  # ModelType for feedforward or str for sequence models
    model_family: str,  # "feedforward" or "sequence"
    label_cols: list[str],
    y_to_log_features: list[str],
    store_cluster: int,
    item_cluster: int,
    *,
    # ... all other parameters
) -> None:
    """Unified training for feedforward per-cluster models and sequence models."""
    
    # ========================================
    # COMMON SETUP (shared by both model types)
    # ========================================
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    pl.seed_everything(seed)
    checkpoints_dir = model_dir / "checkpoints"
    for d in [checkpoints_dir, history_dir, model_logger_dir]:
        if d is not None:
            d.mkdir(parents=True, exist_ok=True)

    model_name = f"{store_cluster}_{item_cluster}_{model_type.value if model_family == 'feedforward' else model_type}"
    
    # ========================================
    # MODEL-SPECIFIC CREATION (only differences)
    # ========================================
    if model_family == "feedforward":
        # Load data
        train_loader, val_loader = _load_feedforward_data(
            dataloader_dir, store_cluster, item_cluster, num_workers, persistent_workers, logger
        )
        if train_loader is None:  # Skip if insufficient data
            return
            
        # Create feedforward model with Lightning wrapper
        input_dim = _get_input_dim(train_loader)
        output_dim = len(label_cols)
        
        base_model = model_factory(
            ModelType(model_type.value), input_dim, hidden_dim, h1, h2, depth, output_dim, dropout
        )
        base_model.apply(init_weights)
        
        # Compute MAV and create Lightning wrapper
        col_y_index_map = {col: idx for idx, col in enumerate(label_cols)}
        y_to_log_features_idx = [col_y_index_map[c] for c in y_to_log_features]
        train_mav = compute_mav(train_loader, y_to_log_features_idx, logger)
        val_mav = compute_mav(val_loader, y_to_log_features_idx, logger)
        
        lightning_model = LightningWrapper(
            base_model, model_name, store_cluster, item_cluster,
            y_to_log_features_idx, train_mav, val_mav, lr=lr, log_level=log_level
        )
        
    elif model_family == "sequence":
        # Load data
        train_loader, val_loader, training_dataset = _load_sequence_data(
            dataloader_dir, store_cluster, item_cluster, logger
        )
        
        # Create sequence model directly (already a Lightning module)
        lightning_model = sequence_model_factory(
            model_type=model_type,
            training_dataset=training_dataset,
            learning_rate=lr,
            hidden_size=hidden_dim,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            output_size=len(label_cols),
        )
        
    else:
        raise ValueError("`model_family` must be either 'feedforward' or 'sequence'")
    
    # ========================================
    # COMMON TRAINING (shared by both model types)
    # ========================================
    trainer = _create_common_trainer_components(
        model_name, store_cluster, item_cluster, checkpoints_dir, 
        model_logger_dir, epochs, enable_progress_bar
    )
    
    # Train the model
    logger.info(f"Training {model_family} model: {model_name}")
    trainer.fit(lightning_model, train_loader, val_loader)
```

## 📊 **Benefits of This Approach**

| Aspect | Before | After |
|--------|--------|-------|
| **Lines of Code** | ~280 lines | ~80 lines |
| **Duplicated Code** | High | Minimal |
| **Maintainability** | Hard to modify | Easy to modify |
| **Readability** | Mixed concerns | Clear separation |
| **Testing** | Hard to test | Easy to test helpers |

## 🔧 **Key Improvements**

### **1. Single Responsibility**
- **Helper functions** handle specific tasks
- **Main function** orchestrates the flow
- **Clear separation** between model-specific and common code

### **2. Shared Common Code**
- **One trainer setup** for both model types
- **One CSV logger setup** for both model types
- **One callback configuration** for both model types

### **3. Isolated Differences**
- **Data loading** - only difference is file patterns and processing
- **Model creation** - feedforward needs wrapper, sequence models don't
- **Everything else is shared**

## 🎯 **The Answer to Your Question**

**Yes, we definitely need only ONE Lightning Wrapper!**

- **Feedforward models**: Use `LightningWrapper` 
- **Sequence models**: Are already Lightning modules (no wrapper needed)
- **Common training code**: Works with both because they're both Lightning modules

This approach eliminates code duplication while maintaining the flexibility to handle both model types appropriately.
