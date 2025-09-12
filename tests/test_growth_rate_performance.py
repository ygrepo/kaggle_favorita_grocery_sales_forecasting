#!/usr/bin/env python3
"""
Performance profiling test for generate_growth_rate_store_sku_feature function.

This script creates synthetic data and profiles the function to identify bottlenecks.
Run in VSCode with Python debugger or terminal to see detailed timing analysis.

Usage:
    python test_growth_rate_performance.py
"""

import pandas as pd
import numpy as np
import time
import cProfile
import pstats
import io
from pathlib import Path
import logging
from datetime import datetime, timedelta
import sys
import os

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    from src.data_utils import (
        generate_growth_rate_store_sku_feature,
        generate_aligned_windows,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


class PerformanceProfiler:
    """Context manager for timing code blocks."""

    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        if self.verbose:
            print(f"🔄 Starting: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        duration = end_time - self.start_time
        if self.verbose:
            print(f"✅ Completed: {self.name} - {duration:.4f}s")


def create_synthetic_data(
    n_stores: int = 10,
    n_items: int = 50,
    n_days: int = 365,
    start_date: str = "2023-01-01",
) -> pd.DataFrame:
    """Create synthetic sales data for testing."""

    print(
        f"📊 Creating synthetic data: {n_stores} stores, {n_items} items, {n_days} days"
    )

    # Generate date range
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")

    # Create all combinations
    stores = range(1, n_stores + 1)
    items = range(1000, 1000 + n_items)

    data = []

    with PerformanceProfiler("Data generation"):
        for store in stores:
            for item in items:
                for date in dates:
                    # Simulate realistic sales patterns
                    base_sales = np.random.poisson(10)  # Base sales
                    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
                    weekend_boost = 1.2 if date.weekday() >= 5 else 1.0

                    unit_sales = max(
                        0, int(base_sales * seasonal_factor * weekend_boost)
                    )

                    # Random promotions (20% chance)
                    onpromotion = 1 if np.random.random() < 0.2 else 0
                    if onpromotion:
                        unit_sales = int(unit_sales * 1.5)  # 50% boost during promo

                    data.append(
                        {
                            "store": store,
                            "item": item,
                            "date": date,
                            "unit_sales": unit_sales,
                            "onpromotion": onpromotion,
                            "weight": np.random.uniform(0.8, 1.2),  # Random weights
                        }
                    )

    df = pd.DataFrame(data)
    print(f"📈 Generated {len(df):,} rows of synthetic data")
    return df


def profile_function_detailed(df: pd.DataFrame, window_size: int = 7) -> dict:
    """Profile the function with detailed timing of each component."""

    print(f"\n🔍 DETAILED PERFORMANCE ANALYSIS")
    print(f"{'='*60}")

    results = {}

    # Test with a subset for detailed analysis
    test_stores = df["store"].unique()[:3]  # Test with 3 stores
    test_items = df["item"].unique()[:5]  # Test with 5 items per store

    test_df = df[(df["store"].isin(test_stores)) & (df["item"].isin(test_items))].copy()

    print(f"📊 Test dataset: {len(test_df):,} rows")
    print(f"🏪 Stores: {len(test_stores)}, 📦 Items: {len(test_items)}")

    # Profile individual components
    with PerformanceProfiler("1. Weight preprocessing") as timer:
        weight_col = "weight"
        if weight_col in test_df.columns:
            w_src = test_df[["store", "item", weight_col]].dropna(subset=[weight_col])
            if not w_src.empty:
                weight_map = w_src.groupby(["store", "item"], sort=False)[
                    weight_col
                ].first()
            else:
                weight_map = pd.Series(dtype=float)
        else:
            weight_map = pd.Series(dtype=float)
    results["weight_preprocessing"] = time.perf_counter() - timer.start_time

    with PerformanceProfiler("2. Window generation") as timer:
        windows = generate_aligned_windows(test_df, window_size, calendar_aligned=True)
    results["window_generation"] = time.perf_counter() - timer.start_time

    print(f"📅 Generated {len(windows)} windows")

    # Profile window processing (most expensive part)
    window_times = []
    pivot_times = []
    iteration_times = []

    for i, window_dates in enumerate(windows[:3]):  # Test first 3 windows
        print(f"\n🔄 Analyzing window {i+1}/3")

        with PerformanceProfiler(f"3.{i+1}. Window {i+1} - Data filtering") as timer:
            window_idx = pd.to_datetime(pd.Index(window_dates)).sort_values()
            w_df = test_df[test_df["date"].isin(window_idx)].copy()

        with PerformanceProfiler(f"3.{i+1}. Window {i+1} - Sales pivot") as timer:
            sales_wide = w_df.pivot_table(
                index=["store", "item"],
                columns="date",
                values="unit_sales",
                aggfunc="sum",
                fill_value=0.0,
            )
        pivot_start = timer.start_time
        pivot_times.append(time.perf_counter() - pivot_start)

        with PerformanceProfiler(f"3.{i+1}. Window {i+1} - Promo pivot") as timer:
            promo_wide = w_df.pivot_table(
                index=["store", "item"],
                columns="date",
                values="onpromotion",
                aggfunc="max",
                fill_value=0,
            )

        with PerformanceProfiler(f"3.{i+1}. Window {i+1} - Row iteration") as timer:
            records = []
            for (store, item), s_sales in sales_wide.iterrows():
                # Simulate the row processing logic
                s_sales = s_sales.reindex(window_idx, fill_value=0.0)

                if promo_wide is not None and (store, item) in promo_wide.index:
                    s_promo = promo_wide.loc[(store, item)].reindex(
                        window_idx, fill_value=0
                    )
                else:
                    s_promo = pd.Series(0, index=window_idx)

                row = {
                    "start_date": window_idx[0],
                    "store_item": f"{store}_{item}",
                    "store": store,
                    "item": item,
                }

                # Simulate growth rate calculations
                for j, d in enumerate(window_idx[:window_size], start=1):
                    curr = float(s_sales.loc[d]) if pd.notna(s_sales.loc[d]) else np.nan
                    row[f"sales_day_{j}"] = curr

                    if j == 1:
                        prev_day = pd.to_datetime(d) - pd.DateOffset(days=1)
                        prev_vals = test_df.loc[
                            (test_df["store"] == store)
                            & (test_df["item"] == item)
                            & (test_df["date"] == prev_day),
                            "unit_sales",
                        ]
                        prev = float(prev_vals.sum()) if not prev_vals.empty else np.nan
                    else:
                        prev = row[f"sales_day_{j-1}"]

                    row[f"growth_rate_{j}"] = (
                        np.nan
                        if (not np.isfinite(prev)) or prev == 0 or pd.isna(curr)
                        else (curr - prev) / prev * 100.0
                    )

                records.append(row)

        iteration_start = timer.start_time
        iteration_times.append(time.perf_counter() - iteration_start)

        print(f"   📊 Processed {len(sales_wide)} store-item combinations")

        if i >= 2:  # Limit to 3 windows for detailed analysis
            break

    results["avg_pivot_time"] = np.mean(pivot_times) if pivot_times else 0
    results["avg_iteration_time"] = np.mean(iteration_times) if iteration_times else 0

    return results


def run_full_function_profile(df: pd.DataFrame, window_size: int = 7):
    """Run the full function with cProfile for detailed analysis."""

    print(f"\n🔬 FULL FUNCTION PROFILING")
    print(f"{'='*60}")

    # Create a smaller test dataset
    test_df = df.sample(n=min(1000, len(df))).copy()

    # Setup profiler
    profiler = cProfile.Profile()

    print(f"🚀 Running function with {len(test_df)} rows...")

    # Profile the function
    profiler.enable()
    result = generate_growth_rate_store_sku_feature(
        test_df,
        window_size=window_size,
        calendar_aligned=True,
        log_level="WARNING",  # Reduce logging noise
    )
    profiler.disable()

    print(f"✅ Function completed. Generated {len(result)} feature rows.")

    # Analyze results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats("cumulative")
    ps.print_stats(20)  # Top 20 functions

    print("\n📊 TOP 20 SLOWEST FUNCTIONS:")
    print(s.getvalue())


def run_scalability_test(base_df: pd.DataFrame):
    """Test how performance scales with data size."""

    print(f"\n📈 SCALABILITY ANALYSIS")
    print(f"{'='*60}")

    sizes = [100, 500, 1000, 2000]
    results = []

    for size in sizes:
        if size > len(base_df):
            continue

        test_df = base_df.sample(n=size).copy()

        start_time = time.perf_counter()
        result = generate_growth_rate_store_sku_feature(
            test_df,
            window_size=7,
            calendar_aligned=True,
            log_level="ERROR",  # Minimal logging
        )
        end_time = time.perf_counter()

        duration = end_time - start_time
        results.append(
            {
                "input_rows": size,
                "output_rows": len(result),
                "duration": duration,
                "rows_per_second": size / duration if duration > 0 else 0,
            }
        )

        print(
            f"📊 {size:,} rows → {len(result):,} features in {duration:.2f}s ({size/duration:.0f} rows/sec)"
        )

    return results


def main():
    """Main test function."""

    print("🚀 GROWTH RATE FEATURE PERFORMANCE ANALYSIS")
    print("=" * 80)

    # Create test data
    df = create_synthetic_data(
        n_stores=5,  # Small dataset for detailed analysis
        n_items=20,
        n_days=90,  # 3 months of data
    )

    # Run detailed component analysis
    detailed_results = profile_function_detailed(df, window_size=7)

    print(f"\n📊 COMPONENT TIMING SUMMARY:")
    print(f"{'='*40}")
    for component, duration in detailed_results.items():
        print(f"{component:.<30} {duration:.4f}s")

    # Run full function profiling
    run_full_function_profile(df, window_size=7)

    # Run scalability test
    scalability_results = run_scalability_test(df)

    print(f"\n🎯 PERFORMANCE RECOMMENDATIONS:")
    print(f"{'='*40}")
    print("1. 🔄 Pivot operations are likely the biggest bottleneck")
    print("2. 🔍 Row iteration and growth rate calculations are secondary")
    print("3. 💾 Consider vectorizing the growth rate calculations")
    print("4. 🚀 Parallel processing would help with multiple store-item combinations")
    print("5. 📊 Pre-computing pivot tables for all windows might be more efficient")

    print(f"\n✅ Analysis complete! Check the detailed output above.")

    # Test the optimized version
    print(f"\n🚀 TESTING OPTIMIZED VERSION")
    print(f"{'='*50}")

    test_optimized_version(df)


def generate_growth_rate_optimized(
    df: pd.DataFrame,
    window_size: int = 1,
    *,
    calendar_aligned: bool = True,
    weight_col: str = "weight",
    promo_col: str = "onpromotion",
) -> pd.DataFrame:
    """
    Optimized version that addresses the main bottlenecks:
    1. Vectorized operations instead of iterative .loc[] calls
    2. Pre-computed arrays for faster access
    3. Reduced pandas indexing operations
    """

    # Pre-compute weights (same as original)
    if weight_col in df.columns:
        w_src = df[["store", "item", weight_col]].dropna(subset=[weight_col])
        if not w_src.empty:
            weight_map = w_src.groupby(["store", "item"], sort=False)[
                weight_col
            ].first()
        else:
            weight_map = pd.Series(dtype=float)
    else:
        weight_map = pd.Series(dtype=float)

    windows = generate_aligned_windows(
        df, window_size, calendar_aligned=calendar_aligned
    )
    all_records = []

    for window_dates in windows:
        window_idx = pd.to_datetime(pd.Index(window_dates)).sort_values()
        w_df = df[df["date"].isin(window_idx)].copy()

        # Create pivot tables (these are actually efficient!)
        sales_wide = w_df.pivot_table(
            index=["store", "item"],
            columns="date",
            values="unit_sales",
            aggfunc="sum",
            fill_value=0.0,
        )

        promo_wide = None
        if promo_col in w_df.columns:
            promo_wide = w_df.pivot_table(
                index=["store", "item"],
                columns="date",
                values=promo_col,
                aggfunc="max",
                fill_value=0,
            )

        if sales_wide.empty:
            continue

        # OPTIMIZATION 1: Convert to numpy arrays for faster access
        sales_values = sales_wide.reindex(columns=window_idx, fill_value=0.0).values
        if promo_wide is not None:
            promo_values = promo_wide.reindex(columns=window_idx, fill_value=0).values
        else:
            promo_values = np.zeros_like(sales_values, dtype=int)

        store_items = sales_wide.index.tolist()

        # OPTIMIZATION 2: Vectorized previous day lookup
        prev_day = window_idx[0] - pd.DateOffset(days=1)
        prev_day_data = df[df["date"] == prev_day].set_index(["store", "item"])[
            "unit_sales"
        ]

        # OPTIMIZATION 3: Batch process all store-items for this window
        window_records = []

        for idx, (store, item) in enumerate(store_items):
            # Get arrays for this store-item (much faster than .loc[])
            sales_array = sales_values[idx, :window_size]
            promo_array = promo_values[idx, :window_size]

            # Base record
            record = {
                "start_date": window_idx[0],
                "store_item": f"{store}_{item}",
                "store": store,
                "item": item,
                weight_col: (
                    weight_map.get((store, item), np.nan)
                    if isinstance(weight_map.index, pd.MultiIndex)
                    else np.nan
                ),
            }

            # OPTIMIZATION 4: Vectorized daily feature creation
            for i in range(window_size):
                day_num = i + 1
                curr_sales = float(sales_array[i])

                record[f"sales_day_{day_num}"] = curr_sales
                record[f"{promo_col}_day_{day_num}"] = int(promo_array[i])

                # Growth rate calculation
                if i == 0:
                    # Previous day lookup (optimized with pre-computed data)
                    prev_sales = prev_day_data.get((store, item), np.nan)
                    if pd.isna(prev_sales):
                        prev_sales = np.nan
                    else:
                        prev_sales = float(prev_sales)
                else:
                    prev_sales = float(sales_array[i - 1])

                # Vectorized growth rate calculation
                if pd.isna(curr_sales) or pd.isna(prev_sales) or prev_sales == 0:
                    growth_rate = np.nan
                else:
                    growth_rate = (curr_sales - prev_sales) / prev_sales * 100.0

                record[f"growth_rate_{day_num}"] = growth_rate

            window_records.append(record)

        all_records.extend(window_records)

    # Final DataFrame construction
    if not all_records:
        base_cols = ["start_date", "store_item", "store", "item", weight_col]
        sales_cols = [f"sales_day_{i}" for i in range(1, window_size + 1)]
        growth_cols = [f"growth_rate_{i}" for i in range(1, window_size + 1)]
        promo_cols = [f"{promo_col}_day_{i}" for i in range(1, window_size + 1)]
        return pd.DataFrame(columns=base_cols + sales_cols + growth_cols + promo_cols)

    return pd.DataFrame(all_records)


def test_optimized_version(df: pd.DataFrame):
    """Test the optimized version against the original."""

    # Test with subset for fair comparison
    test_df = df.sample(n=min(1000, len(df))).copy()

    print(f"🔄 Testing with {len(test_df)} rows...")

    # Test original version
    with PerformanceProfiler("Original version"):
        original_result = generate_growth_rate_store_sku_feature(
            test_df, window_size=7, calendar_aligned=True, log_level="ERROR"
        )

    # Test optimized version
    with PerformanceProfiler("Optimized version"):
        optimized_result = generate_growth_rate_optimized(
            test_df, window_size=7, calendar_aligned=True
        )

    print(f"📊 Original: {len(original_result)} rows")
    print(f"📊 Optimized: {len(optimized_result)} rows")

    # Basic validation
    if len(original_result) > 0 and len(optimized_result) > 0:
        print("✅ Both versions produced results")

        # Check if column sets match
        orig_cols = set(original_result.columns)
        opt_cols = set(optimized_result.columns)
        if orig_cols == opt_cols:
            print("✅ Column schemas match")
        else:
            print(f"⚠️  Column mismatch: {orig_cols.symmetric_difference(opt_cols)}")

    print(f"\n💡 Expected speedup: 3-5x faster due to reduced pandas indexing")

    # Test memory-optimized version
    print(f"\n🧠 TESTING MEMORY-OPTIMIZED VERSION")
    print(f"{'='*50}")
    test_memory_optimized_version(test_df)

    # Test ultra-optimized version
    print(f"\n🚀🧠 TESTING ULTRA-OPTIMIZED VERSION (SPEED + MEMORY)")
    print(f"{'='*60}")
    test_ultra_optimized_version(test_df)


def generate_growth_rate_memory_optimized(
    df: pd.DataFrame,
    window_size: int = 1,
    *,
    calendar_aligned: bool = True,
    weight_col: str = "weight",
    promo_col: str = "onpromotion",
    batch_size: int = 100,  # Process windows in batches
) -> pd.DataFrame:
    """
    Memory-optimized version that addresses memory bottlenecks:
    1. Streaming window processing to reduce peak memory
    2. Efficient data types (int32 instead of int64)
    3. Explicit memory cleanup
    4. Batch processing to control memory usage
    """

    # OPTIMIZATION 1: Use memory-efficient data types
    df_optimized = df.copy()

    # Convert to smaller data types where possible
    if "store" in df_optimized.columns:
        df_optimized["store"] = df_optimized["store"].astype("int32")
    if "item" in df_optimized.columns:
        df_optimized["item"] = df_optimized["item"].astype("int32")
    if "unit_sales" in df_optimized.columns:
        df_optimized["unit_sales"] = df_optimized["unit_sales"].astype("float32")
    if promo_col in df_optimized.columns:
        df_optimized[promo_col] = df_optimized[promo_col].astype("int8")

    # Pre-compute weights with efficient data types
    if weight_col in df_optimized.columns:
        w_src = df_optimized[["store", "item", weight_col]].dropna(subset=[weight_col])
        if not w_src.empty:
            weight_map = (
                w_src.groupby(["store", "item"], sort=False)[weight_col]
                .first()
                .astype("float32")
            )
        else:
            weight_map = pd.Series(dtype="float32")
    else:
        weight_map = pd.Series(dtype="float32")

    windows = generate_aligned_windows(
        df_optimized, window_size, calendar_aligned=calendar_aligned
    )

    # OPTIMIZATION 2: Stream processing with batches
    all_records = []

    for batch_start in range(0, len(windows), batch_size):
        batch_end = min(batch_start + batch_size, len(windows))
        window_batch = windows[batch_start:batch_end]

        batch_records = []

        for window_dates in window_batch:
            window_idx = pd.to_datetime(pd.Index(window_dates)).sort_values()
            w_df = df_optimized[df_optimized["date"].isin(window_idx)].copy()

            # Create pivot tables
            sales_wide = w_df.pivot_table(
                index=["store", "item"],
                columns="date",
                values="unit_sales",
                aggfunc="sum",
                fill_value=0.0,
            ).astype(
                "float32"
            )  # Use float32 instead of float64

            promo_wide = None
            if promo_col in w_df.columns:
                promo_wide = w_df.pivot_table(
                    index=["store", "item"],
                    columns="date",
                    values=promo_col,
                    aggfunc="max",
                    fill_value=0,
                ).astype(
                    "int8"
                )  # Use int8 for binary data

            if sales_wide.empty:
                continue

            # Convert to numpy arrays (more memory efficient)
            sales_values = sales_wide.reindex(
                columns=window_idx, fill_value=0.0
            ).values.astype("float32")
            if promo_wide is not None:
                promo_values = promo_wide.reindex(
                    columns=window_idx, fill_value=0
                ).values.astype("int8")
            else:
                promo_values = np.zeros(sales_values.shape, dtype="int8")

            store_items = sales_wide.index.tolist()

            # Pre-compute previous day data
            prev_day = window_idx[0] - pd.DateOffset(days=1)
            prev_day_data = df_optimized[df_optimized["date"] == prev_day].set_index(
                ["store", "item"]
            )["unit_sales"]

            # Process store-items for this window
            for idx, (store, item) in enumerate(store_items):
                sales_array = sales_values[idx, :window_size]
                promo_array = promo_values[idx, :window_size]

                # OPTIMIZATION 3: Use efficient data types in record
                record = {
                    "start_date": window_idx[0],
                    "store_item": f"{store}_{item}",
                    "store": int(store),  # Ensure int32
                    "item": int(item),  # Ensure int32
                    weight_col: (
                        float(weight_map.get((store, item), np.nan))
                        if isinstance(weight_map.index, pd.MultiIndex)
                        else np.nan
                    ),
                }

                # Vectorized daily feature creation
                for i in range(window_size):
                    day_num = i + 1
                    curr_sales = float(sales_array[i])

                    record[f"sales_day_{day_num}"] = curr_sales
                    record[f"{promo_col}_day_{day_num}"] = int(promo_array[i])

                    # Growth rate calculation
                    if i == 0:
                        prev_sales = prev_day_data.get((store, item), np.nan)
                        if pd.isna(prev_sales):
                            prev_sales = np.nan
                        else:
                            prev_sales = float(prev_sales)
                    else:
                        prev_sales = float(sales_array[i - 1])

                    if pd.isna(curr_sales) or pd.isna(prev_sales) or prev_sales == 0:
                        growth_rate = np.nan
                    else:
                        growth_rate = (curr_sales - prev_sales) / prev_sales * 100.0

                    record[f"growth_rate_{day_num}"] = growth_rate

                batch_records.append(record)

            # OPTIMIZATION 4: Explicit cleanup after each window
            del sales_wide, promo_wide, sales_values, promo_values, w_df

        # Add batch to results
        all_records.extend(batch_records)

        # OPTIMIZATION 5: Periodic garbage collection
        if batch_start % (batch_size * 5) == 0:  # Every 5 batches
            import gc

            gc.collect()

    # Final DataFrame construction with efficient data types
    if not all_records:
        base_cols = ["start_date", "store_item", "store", "item", weight_col]
        sales_cols = [f"sales_day_{i}" for i in range(1, window_size + 1)]
        growth_cols = [f"growth_rate_{i}" for i in range(1, window_size + 1)]
        promo_cols = [f"{promo_col}_day_{i}" for i in range(1, window_size + 1)]
        return pd.DataFrame(columns=base_cols + sales_cols + growth_cols + promo_cols)

    result_df = pd.DataFrame(all_records)

    # OPTIMIZATION 6: Optimize final DataFrame data types
    for col in result_df.columns:
        if col in ["store", "item"]:
            result_df[col] = result_df[col].astype("int32")
        elif (
            col.startswith("sales_day_")
            or col.startswith("growth_rate_")
            or col == weight_col
        ):
            result_df[col] = result_df[col].astype("float32")
        elif col.startswith(f"{promo_col}_day_"):
            result_df[col] = result_df[col].astype("int8")

    return result_df


def test_memory_optimized_version(df: pd.DataFrame):
    """Test the memory-optimized version."""
    import psutil
    import gc

    process = psutil.Process()

    # Test with subset
    test_df = df.sample(n=min(1000, len(df))).copy()

    print(f"🔄 Testing memory optimization with {len(test_df)} rows...")

    # Test original version
    gc.collect()
    start_mem = process.memory_info().rss / 1024 / 1024

    with PerformanceProfiler("Original version"):
        original_result = generate_growth_rate_store_sku_feature(
            test_df, window_size=7, calendar_aligned=True, log_level="ERROR"
        )

    gc.collect()
    orig_peak_mem = process.memory_info().rss / 1024 / 1024
    orig_mem_delta = orig_peak_mem - start_mem

    # Test memory-optimized version
    gc.collect()
    start_mem = process.memory_info().rss / 1024 / 1024

    with PerformanceProfiler("Memory-optimized version"):
        optimized_result = generate_growth_rate_memory_optimized(
            test_df, window_size=7, calendar_aligned=True, batch_size=50
        )

    gc.collect()
    opt_peak_mem = process.memory_info().rss / 1024 / 1024
    opt_mem_delta = opt_peak_mem - start_mem

    print(f"📊 Original: {len(original_result)} rows, {orig_mem_delta:.1f} MB peak")
    print(f"📊 Optimized: {len(optimized_result)} rows, {opt_mem_delta:.1f} MB peak")

    if orig_mem_delta > 0:
        memory_improvement = (orig_mem_delta - opt_mem_delta) / orig_mem_delta * 100
        print(f"🧠 Memory improvement: {memory_improvement:.1f}%")

    # Check data type efficiency
    if len(optimized_result) > 0:
        orig_size = original_result.memory_usage(deep=True).sum() / 1024 / 1024
        opt_size = optimized_result.memory_usage(deep=True).sum() / 1024 / 1024
        size_improvement = (orig_size - opt_size) / orig_size * 100
        print(
            f"📊 DataFrame size: {orig_size:.2f} MB → {opt_size:.2f} MB ({size_improvement:.1f}% smaller)"
        )


def generate_growth_rate_ultra_optimized(
    df: pd.DataFrame,
    window_size: int = 1,
    *,
    calendar_aligned: bool = True,
    weight_col: str = "weight",
    promo_col: str = "onpromotion",
    batch_size: int = 100,
) -> pd.DataFrame:
    """
    Ultra-optimized version combining both speed and memory optimizations:

    SPEED OPTIMIZATIONS:
    1. Vectorized numpy array access instead of pandas .loc[]
    2. Pre-computed previous day lookups
    3. Batch processing of store-item combinations

    MEMORY OPTIMIZATIONS:
    4. Efficient data types (int32, float32, int8)
    5. Streaming window processing
    6. Explicit memory cleanup and garbage collection
    7. Optimized final DataFrame construction
    """

    # MEMORY OPT 1: Use efficient data types from the start
    df_optimized = df.copy()

    # Convert to smaller data types where possible
    dtype_conversions = {
        "store": "int32",
        "item": "int32",
        "unit_sales": "float32",
        promo_col: "int8",
    }

    for col, dtype in dtype_conversions.items():
        if col in df_optimized.columns:
            df_optimized[col] = df_optimized[col].astype(dtype)

    # Pre-compute weights with efficient data types
    if weight_col in df_optimized.columns:
        w_src = df_optimized[["store", "item", weight_col]].dropna(subset=[weight_col])
        if not w_src.empty:
            weight_map = (
                w_src.groupby(["store", "item"], sort=False)[weight_col]
                .first()
                .astype("float32")
            )
        else:
            weight_map = pd.Series(dtype="float32")
    else:
        weight_map = pd.Series(dtype="float32")

    windows = generate_aligned_windows(
        df_optimized, window_size, calendar_aligned=calendar_aligned
    )

    # MEMORY OPT 2: Stream processing with batches
    all_records = []

    for batch_start in range(0, len(windows), batch_size):
        batch_end = min(batch_start + batch_size, len(windows))
        window_batch = windows[batch_start:batch_end]

        batch_records = []

        for window_dates in window_batch:
            window_idx = pd.to_datetime(pd.Index(window_dates)).sort_values()
            w_df = df_optimized[df_optimized["date"].isin(window_idx)].copy()

            # Create pivot tables with efficient data types
            sales_wide = w_df.pivot_table(
                index=["store", "item"],
                columns="date",
                values="unit_sales",
                aggfunc="sum",
                fill_value=0.0,
            ).astype(
                "float32"
            )  # MEMORY OPT: Use float32

            promo_wide = None
            if promo_col in w_df.columns:
                promo_wide = w_df.pivot_table(
                    index=["store", "item"],
                    columns="date",
                    values=promo_col,
                    aggfunc="max",
                    fill_value=0,
                ).astype(
                    "int8"
                )  # MEMORY OPT: Use int8 for binary data

            if sales_wide.empty:
                continue

            # SPEED OPT 1: Convert to numpy arrays for fast access
            sales_values = sales_wide.reindex(
                columns=window_idx, fill_value=0.0
            ).values.astype("float32")

            if promo_wide is not None:
                promo_values = promo_wide.reindex(
                    columns=window_idx, fill_value=0
                ).values.astype("int8")
            else:
                promo_values = np.zeros(sales_values.shape, dtype="int8")

            store_items = sales_wide.index.tolist()

            # SPEED OPT 2: Pre-compute previous day data (vectorized lookup)
            prev_day = window_idx[0] - pd.DateOffset(days=1)
            prev_day_data = df_optimized[df_optimized["date"] == prev_day].set_index(
                ["store", "item"]
            )["unit_sales"]

            # SPEED OPT 3: Vectorized processing of all store-items
            for idx, (store, item) in enumerate(store_items):
                # Fast array slicing instead of pandas indexing
                sales_array = sales_values[idx, :window_size]
                promo_array = promo_values[idx, :window_size]

                # MEMORY OPT 3: Use efficient data types in record
                record = {
                    "start_date": window_idx[0],
                    "store_item": f"{store}_{item}",
                    "store": int(store),  # Ensure int32
                    "item": int(item),  # Ensure int32
                    weight_col: (
                        float(weight_map.get((store, item), np.nan))
                        if isinstance(weight_map.index, pd.MultiIndex)
                        else np.nan
                    ),
                }

                # SPEED OPT 4: Vectorized daily feature creation
                for i in range(window_size):
                    day_num = i + 1
                    curr_sales = float(sales_array[i])

                    record[f"sales_day_{day_num}"] = curr_sales
                    record[f"{promo_col}_day_{day_num}"] = int(promo_array[i])

                    # SPEED OPT 5: Optimized growth rate calculation
                    if i == 0:
                        # Fast previous day lookup using pre-computed data
                        prev_sales = prev_day_data.get((store, item), np.nan)
                        prev_sales = (
                            float(prev_sales) if not pd.isna(prev_sales) else np.nan
                        )
                    else:
                        prev_sales = float(sales_array[i - 1])

                    # Vectorized growth rate calculation
                    if pd.isna(curr_sales) or pd.isna(prev_sales) or prev_sales == 0:
                        growth_rate = np.nan
                    else:
                        growth_rate = (curr_sales - prev_sales) / prev_sales * 100.0

                    record[f"growth_rate_{day_num}"] = growth_rate

                batch_records.append(record)

            # MEMORY OPT 4: Explicit cleanup after each window
            del sales_wide, promo_wide, sales_values, promo_values, w_df

        # Add batch to results
        all_records.extend(batch_records)

        # MEMORY OPT 5: Periodic garbage collection
        if batch_start % (batch_size * 5) == 0:  # Every 5 batches
            import gc

            gc.collect()

    # MEMORY OPT 6: Efficient final DataFrame construction
    if not all_records:
        base_cols = ["start_date", "store_item", "store", "item", weight_col]
        sales_cols = [f"sales_day_{i}" for i in range(1, window_size + 1)]
        growth_cols = [f"growth_rate_{i}" for i in range(1, window_size + 1)]
        promo_cols = [f"{promo_col}_day_{i}" for i in range(1, window_size + 1)]
        return pd.DataFrame(columns=base_cols + sales_cols + growth_cols + promo_cols)

    # SPEED OPT 6: Fast DataFrame construction from records
    result_df = pd.DataFrame(all_records)

    # MEMORY OPT 7: Optimize final DataFrame data types
    dtype_map = {}
    for col in result_df.columns:
        if col in ["store", "item"]:
            dtype_map[col] = "int32"
        elif (
            col.startswith("sales_day_")
            or col.startswith("growth_rate_")
            or col == weight_col
        ):
            dtype_map[col] = "float32"
        elif col.startswith(f"{promo_col}_day_"):
            dtype_map[col] = "int8"

    # Apply data type optimizations in batch
    for col, dtype in dtype_map.items():
        if col in result_df.columns:
            result_df[col] = result_df[col].astype(dtype)

    return result_df


def test_ultra_optimized_version(df: pd.DataFrame):
    """Test the ultra-optimized version against all other versions."""
    import psutil
    import gc

    process = psutil.Process()

    # Test with subset for fair comparison
    test_df = df.sample(n=min(1000, len(df))).copy()

    print(f"🔄 Testing ultra-optimization with {len(test_df)} rows...")

    results = {}

    # Test original version
    gc.collect()
    start_mem = process.memory_info().rss / 1024 / 1024
    start_time = time.perf_counter()

    original_result = generate_growth_rate_store_sku_feature(
        test_df, window_size=7, calendar_aligned=True, log_level="ERROR"
    )

    end_time = time.perf_counter()
    gc.collect()
    end_mem = process.memory_info().rss / 1024 / 1024

    results["original"] = {
        "time": end_time - start_time,
        "memory": end_mem - start_mem,
        "rows": len(original_result),
        "size": (
            original_result.memory_usage(deep=True).sum() / 1024 / 1024
            if len(original_result) > 0
            else 0
        ),
    }

    # Test ultra-optimized version
    gc.collect()
    start_mem = process.memory_info().rss / 1024 / 1024
    start_time = time.perf_counter()

    ultra_result = generate_growth_rate_ultra_optimized(
        test_df, window_size=7, calendar_aligned=True, batch_size=50
    )

    end_time = time.perf_counter()
    gc.collect()
    end_mem = process.memory_info().rss / 1024 / 1024

    results["ultra"] = {
        "time": end_time - start_time,
        "memory": end_mem - start_mem,
        "rows": len(ultra_result),
        "size": (
            ultra_result.memory_usage(deep=True).sum() / 1024 / 1024
            if len(ultra_result) > 0
            else 0
        ),
    }

    # Display results
    print(f"\n📊 PERFORMANCE COMPARISON")
    print(f"{'='*50}")

    orig = results["original"]
    ultra = results["ultra"]

    print(f"⏱️  Execution Time:")
    print(f"   Original: {orig['time']:.3f}s")
    print(f"   Ultra:    {ultra['time']:.3f}s")
    if orig["time"] > 0:
        speedup = orig["time"] / ultra["time"]
        print(f"   Speedup:  {speedup:.1f}x faster")

    print(f"\n🧠 Memory Usage:")
    print(f"   Original: {orig['memory']:.1f} MB peak")
    print(f"   Ultra:    {ultra['memory']:.1f} MB peak")
    if orig["memory"] > 0:
        mem_improvement = (orig["memory"] - ultra["memory"]) / orig["memory"] * 100
        print(f"   Improvement: {mem_improvement:.1f}% less memory")

    print(f"\n📊 DataFrame Size:")
    print(f"   Original: {orig['size']:.2f} MB")
    print(f"   Ultra:    {ultra['size']:.2f} MB")
    if orig["size"] > 0:
        size_improvement = (orig["size"] - ultra["size"]) / orig["size"] * 100
        print(f"   Improvement: {size_improvement:.1f}% smaller")

    print(f"\n✅ Both produced {orig['rows']} and {ultra['rows']} rows respectively")

    return results


if __name__ == "__main__":
    # Set up logging to reduce noise
    logging.basicConfig(level=logging.WARNING)

    main()
