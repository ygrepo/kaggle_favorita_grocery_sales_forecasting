#!/usr/bin/env python3
"""
Memory profiler for generate_growth_rate_store_sku_feature.

This script tracks memory usage to identify memory bottlenecks and leaks.

Installation:
    pip install memory_profiler psutil

Usage:
    # Method 1: Direct execution with memory tracking
    python memory_profiler_test.py
    
    # Method 2: Using mprof for detailed memory plots
    mprof run memory_profiler_test.py
    mprof plot  # Creates a memory usage plot
"""

import pandas as pd
import numpy as np
import sys
import os
import gc
import psutil
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.data_utils import generate_growth_rate_store_sku_feature, generate_aligned_windows
    from memory_profiler import profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError as e:
    print(f"Memory profiler not available: {e}")
    print("Install with: pip install memory_profiler psutil")
    MEMORY_PROFILER_AVAILABLE = False
    
    # Create dummy profile decorator
    def profile(func):
        return func


class MemoryTracker:
    """Simple memory usage tracker."""
    
    def __init__(self, name: str):
        self.name = name
        self.process = psutil.Process()
        self.start_memory = None
        
    def __enter__(self):
        gc.collect()  # Force garbage collection
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        print(f"🧠 {self.name} - Start: {self.start_memory:.1f} MB")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        delta = end_memory - self.start_memory
        print(f"🧠 {self.name} - End: {end_memory:.1f} MB (Δ{delta:+.1f} MB)")


def create_memory_test_data(n_stores=5, n_items=20, n_days=60):
    """Create test data for memory profiling."""
    print(f"📊 Creating memory test data: {n_stores} stores, {n_items} items, {n_days} days")
    
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq='D')
    
    data = []
    for store in range(1, n_stores + 1):
        for item in range(1000, 1000 + n_items):
            for date in dates:
                data.append({
                    'store': store,
                    'item': item,
                    'date': date,
                    'unit_sales': np.random.poisson(8),
                    'onpromotion': np.random.choice([0, 1], p=[0.7, 0.3]),
                    'weight': np.random.uniform(0.9, 1.1)
                })
    
    df = pd.DataFrame(data)
    print(f"📈 Generated {len(df):,} rows ({df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB)")
    return df


@profile
def memory_profiled_function(df: pd.DataFrame, window_size: int = 7) -> pd.DataFrame:
    """Memory-profiled version of the function."""
    
    # Weight preprocessing
    if "weight" in df.columns:
        w_src = df[["store", "item", "weight"]].dropna(subset=["weight"])
        if not w_src.empty:
            weight_map = w_src.groupby(["store", "item"], sort=False)["weight"].first()
        else:
            weight_map = pd.Series(dtype=float)
    else:
        weight_map = pd.Series(dtype=float)

    # Generate windows
    windows = generate_aligned_windows(df, window_size, calendar_aligned=True)
    records = []

    # Process each window
    for window_dates in windows:
        window_idx = pd.to_datetime(pd.Index(window_dates)).sort_values()
        w_df = df[df["date"].isin(window_idx)].copy()

        # Memory-intensive pivot operations
        sales_wide = w_df.pivot_table(
            index=["store", "item"],
            columns="date",
            values="unit_sales",
            aggfunc="sum",
            fill_value=0.0,
        )

        promo_wide = None
        if "onpromotion" in w_df.columns:
            promo_wide = w_df.pivot_table(
                index=["store", "item"],
                columns="date", 
                values="onpromotion",
                aggfunc="max",
                fill_value=0,
            )

        if sales_wide.empty:
            continue

        # Process each store-item combination
        for (store, item), s_sales in sales_wide.iterrows():
            s_sales = s_sales.reindex(window_idx, fill_value=0.0)

            if promo_wide is not None and (store, item) in promo_wide.index:
                s_promo = promo_wide.loc[(store, item)].reindex(window_idx, fill_value=0)
            else:
                s_promo = pd.Series(0, index=window_idx)

            row = {
                "start_date": window_idx[0],
                "store_item": f"{store}_{item}",
                "store": store,
                "item": item,
            }

            # Weight lookup
            row["weight"] = (
                weight_map.get((store, item))
                if isinstance(weight_map.index, pd.MultiIndex)
                else np.nan
            )

            # Daily calculations
            for i, d in enumerate(window_idx[:window_size], start=1):
                curr = float(s_sales.loc[d]) if pd.notna(s_sales.loc[d]) else np.nan
                row[f"sales_day_{i}"] = curr
                row[f"onpromotion_day_{i}"] = (
                    int(s_promo.loc[d]) if pd.notna(s_promo.loc[d]) else 0
                )

                # Previous day lookup
                if i == 1:
                    prev_day = pd.to_datetime(d) - pd.DateOffset(days=1)
                    prev_vals = df.loc[
                        (df["store"] == store)
                        & (df["item"] == item)
                        & (df["date"] == prev_day),
                        "unit_sales",
                    ]
                    prev = float(prev_vals.sum()) if not prev_vals.empty else np.nan
                else:
                    prev = row[f"sales_day_{i-1}"]

                # Growth rate calculation
                row[f"growth_rate_{i}"] = (
                    np.nan
                    if (not np.isfinite(prev)) or prev == 0 or pd.isna(curr)
                    else (curr - prev) / prev * 100.0
                )

            records.append(row)

        # Explicit cleanup to help with memory
        del sales_wide, promo_wide
        gc.collect()

    # Final DataFrame construction
    base_cols = ["start_date", "store_item", "store", "item", "weight"]
    sales_cols = [f"sales_day_{i}" for i in range(1, window_size + 1)]
    growth_cols = [f"growth_rate_{i}" for i in range(1, window_size + 1)]
    promo_cols = [f"onpromotion_day_{i}" for i in range(1, window_size + 1)]
    cols = base_cols + sales_cols + growth_cols + promo_cols

    out = pd.DataFrame.from_records(records)
    if out.empty:
        out = pd.DataFrame(columns=cols)
    else:
        for c in cols:
            if c not in out:
                out[c] = np.nan
        out = out[cols]

    return out


def analyze_memory_components(df: pd.DataFrame):
    """Analyze memory usage of individual components."""
    
    print(f"\n🧠 MEMORY COMPONENT ANALYSIS")
    print(f"{'='*50}")
    
    with MemoryTracker("Input DataFrame"):
        input_size = df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"   📊 Input data: {input_size:.1f} MB")
    
    with MemoryTracker("Weight preprocessing"):
        w_src = df[["store", "item", "weight"]].dropna(subset=["weight"])
        weight_map = w_src.groupby(["store", "item"], sort=False)["weight"].first()
        weight_size = weight_map.memory_usage(deep=True) / 1024 / 1024
        print(f"   📊 Weight map: {weight_size:.1f} MB")
    
    with MemoryTracker("Window generation"):
        windows = generate_aligned_windows(df, 7, calendar_aligned=True)
        print(f"   📊 Generated {len(windows)} windows")
    
    # Test pivot table memory usage
    if windows:
        window_idx = pd.to_datetime(pd.Index(windows[0])).sort_values()
        w_df = df[df["date"].isin(window_idx)].copy()
        
        with MemoryTracker("Sales pivot table"):
            sales_wide = w_df.pivot_table(
                index=["store", "item"],
                columns="date",
                values="unit_sales",
                aggfunc="sum",
                fill_value=0.0,
            )
            pivot_size = sales_wide.memory_usage(deep=True).sum() / 1024 / 1024
            print(f"   📊 Sales pivot: {pivot_size:.1f} MB ({sales_wide.shape})")
        
        with MemoryTracker("Promo pivot table"):
            promo_wide = w_df.pivot_table(
                index=["store", "item"],
                columns="date",
                values="onpromotion",
                aggfunc="max",
                fill_value=0,
            )
            promo_size = promo_wide.memory_usage(deep=True).sum() / 1024 / 1024
            print(f"   📊 Promo pivot: {promo_size:.1f} MB ({promo_wide.shape})")


def run_memory_scaling_test():
    """Test memory usage with different data sizes."""
    
    print(f"\n📈 MEMORY SCALING TEST")
    print(f"{'='*50}")
    
    sizes = [
        (2, 5, 30),   # Small: 300 rows
        (3, 10, 30),  # Medium: 900 rows  
        (5, 15, 30),  # Large: 2,250 rows
    ]
    
    for n_stores, n_items, n_days in sizes:
        print(f"\n🧪 Testing {n_stores} stores × {n_items} items × {n_days} days")
        
        with MemoryTracker(f"Dataset {n_stores}×{n_items}×{n_days}"):
            df = create_memory_test_data(n_stores, n_items, n_days)
            
            with MemoryTracker("Function execution"):
                result = memory_profiled_function(df, window_size=7)
                
            result_size = result.memory_usage(deep=True).sum() / 1024 / 1024
            print(f"   📊 Result: {len(result)} rows, {result_size:.1f} MB")
            
            # Cleanup
            del df, result
            gc.collect()


def main():
    """Main function to run memory profiling tests."""
    
    print("🧠 MEMORY PROFILING ANALYSIS")
    print("="*60)
    
    if not MEMORY_PROFILER_AVAILABLE:
        print("⚠️  Memory profiler not available - install with: pip install memory_profiler psutil")
        print("Running basic memory tracking instead...\n")
    
    # Create test data
    df = create_memory_test_data(n_stores=4, n_items=15, n_days=45)
    
    # Analyze individual components
    analyze_memory_components(df)
    
    # Run the full profiled function
    print(f"\n🔬 FULL FUNCTION MEMORY PROFILE")
    print(f"{'='*40}")
    
    with MemoryTracker("Complete function execution"):
        result = memory_profiled_function(df, window_size=7)
        
    print(f"✅ Generated {len(result)} feature rows")
    
    # Run scaling test
    run_memory_scaling_test()
    
    print(f"\n💡 MEMORY OPTIMIZATION RECOMMENDATIONS:")
    print(f"{'='*50}")
    print("1. 🔄 Pivot tables are the biggest memory consumers")
    print("2. 📊 Consider processing windows in smaller batches")
    print("3. 🧹 Explicit cleanup (del, gc.collect()) helps")
    print("4. 💾 Stream processing could reduce peak memory usage")
    print("5. 🚀 Consider using more memory-efficient data types (int32 vs int64)")


if __name__ == "__main__":
    main()
