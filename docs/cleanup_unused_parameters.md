# Cleanup: Removed Unused DataLoader Parameters

This document outlines the cleanup of unused `num_workers` and `persistent_workers` parameters after simplifying the data loading approach.

## 🎯 **Issue Identified**

After simplifying `train_model_unified` to use direct `torch.load()` instead of recreating DataLoaders, the following parameters became unused:
- `num_workers: int = 15`
- `persistent_workers: bool = True`

## 🧹 **Files Cleaned Up**

### **1. src/model_utils.py**

**Function: `train_model_unified`**
```python
# Before
def train_model_unified(
    # ... other params
    num_workers: int = 15,
    persistent_workers: bool = True,
    enable_progress_bar: bool = True,
    # ...
)

# After  
def train_model_unified(
    # ... other params
    enable_progress_bar: bool = True,
    # ...
)
```

**Function: `train_all_models_for_cluster_pair`**
```python
# Before
def train_all_models_for_cluster_pair(
    # ... other params
    num_workers: int = 15,
    persistent_workers: bool = True,
    enable_progress_bar: bool = True,
    # ...
)

# After
def train_all_models_for_cluster_pair(
    # ... other params  
    enable_progress_bar: bool = True,
    # ...
)
```

**Function: `train_per_cluster_pair`**
- Updated type annotation: `ModelType` → `MODEL_TYPE`
- Removed unused parameter passing

### **2. script/training.py**

**Argument Parser**
```python
# Removed these argument definitions:
parser.add_argument("--num_workers", ...)
parser.add_argument("--persistent_workers", ...)
```

**Variable Assignments**
```python
# Before
num_workers = args.num_workers
persistent_workers = args.persistent_workers
enable_progress_bar = args.enable_progress_bar

# After
enable_progress_bar = args.enable_progress_bar
```

**Logging Statements**
```python
# Before
logger.info(f"  Num workers: {num_workers}")
logger.info(f"  Persistent workers: {persistent_workers}")
logger.info(f"  Enable progress bar: {enable_progress_bar}")

# After
logger.info(f"  Enable progress bar: {enable_progress_bar}")
```

**Function Calls**
```python
# Before
train_all_models_for_cluster_pair(
    # ... other params
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    enable_progress_bar=enable_progress_bar,
    # ...
)

# After
train_all_models_for_cluster_pair(
    # ... other params
    enable_progress_bar=enable_progress_bar,
    # ...
)
```

## 🚀 **Why These Parameters Became Unused**

### **Before (Complex DataLoader Recreation)**
```python
def train_model_unified(...):
    # Load saved dataloaders
    train_loader = torch.load(...)
    val_loader = torch.load(...)
    
    # Recreate dataloaders with custom settings
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=batch_size,
        num_workers=num_workers,        # ← Used here
        persistent_workers=persistent_workers,  # ← Used here
        pin_memory=True,
    )
    # ... same for val_loader
```

### **After (Direct DataLoader Usage)**
```python
def train_model_unified(...):
    # Load and use dataloaders directly
    train_loader = torch.load(...)  # ← Use as-is
    val_loader = torch.load(...)    # ← Use as-is
    
    # No DataLoader recreation needed!
    # num_workers and persistent_workers not needed
```

## 📊 **Benefits of Cleanup**

### **1. Simplified Function Signatures**
- **Fewer parameters** to pass around
- **Cleaner interfaces** with only relevant parameters
- **Less cognitive overhead** when calling functions

### **2. Reduced Code Complexity**
- **No unused parameter passing** through the call chain
- **Cleaner argument parsing** in scripts
- **Simplified logging** statements

### **3. Better Maintainability**
- **No dead code** to maintain
- **Clear parameter purpose** - only used parameters remain
- **Easier to understand** function interfaces

## 🔧 **Parameter Reduction Summary**

| Function | Before | After | Reduction |
|----------|--------|-------|-----------|
| `train_model_unified` | 16 parameters | 14 parameters | **2 parameters** |
| `train_all_models_for_cluster_pair` | 14 parameters | 12 parameters | **2 parameters** |
| `training.py` argument parser | 15 arguments | 13 arguments | **2 arguments** |

## 🎯 **Key Insight**

The simplification of data loading (using saved DataLoaders directly instead of recreating them) eliminated the need for DataLoader configuration parameters. This is a perfect example of how **simplifying one part of the system** (data loading) can **cascade to simplify other parts** (function signatures, argument parsing, etc.).

## ✅ **Verification**

All scripts and functions have been updated to:
1. ✅ Remove unused `num_workers` and `persistent_workers` parameters
2. ✅ Update function signatures and calls
3. ✅ Clean up argument parsing and logging
4. ✅ Maintain all existing functionality

The codebase is now cleaner and more maintainable with no unused parameters! 🎉
