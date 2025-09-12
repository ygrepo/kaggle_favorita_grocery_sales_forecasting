#!/usr/bin/env python3
"""
Line-by-line profiler for generate_growth_rate_store_sku_feature.

This script uses line_profiler to show exactly which lines are taking the most time.

Installation:
    pip install line_profiler

Usage:
    # Method 1: Direct execution
    python line_profiler_test.py

    # Method 2: Using kernprof (more detailed)
    kernprof -l -v line_profiler_test.py
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    from src.data_utils import generate_aligned_windows
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def create_test_data(n_stores=3, n_items=10, n_days=30):
    """Create small test dataset for line profiling."""
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")

    data = []
    for store in range(1, n_stores + 1):
        for item in range(1000, 1000 + n_items):
            for date in dates:
                data.append(
                    {
                        "store": store,
                        "item": item,
                        "date": date,
                        "unit_sales": np.random.poisson(5),
                        "onpromotion": np.random.choice([0, 1], p=[0.8, 0.2]),
                        "weight": 1.0,
                    }
                )

    return pd.DataFrame(data)


# Add @profile decorator for line_profiler
@profile
def generate_growth_rate_store_sku_feature_profiled(
    df: pd.DataFrame,
    window_size: int = 1,
    *,
    calendar_aligned: bool = True,
    weight_col: str = "weight",
    promo_col: str = "onpromotion",
) -> pd.DataFrame:
    """
    Profiled version of the function with @profile decorators on key sections.
    """

    # --- Precompute weights ---
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

    # --- Generate windows ---
    windows = generate_aligned_windows(
        df, window_size, calendar_aligned=calendar_aligned
    )
    records = []

    # --- Process each window ---
    for window_dates in windows:
        window_idx = pd.to_datetime(pd.Index(window_dates)).sort_values()
        w_df = df[df["date"].isin(window_idx)].copy()

        # --- BOTTLENECK 1: Sales pivot table ---
        sales_wide = w_df.pivot_table(
            index=["store", "item"],
            columns="date",
            values="unit_sales",
            aggfunc="sum",
            fill_value=0.0,
        )

        # --- BOTTLENECK 2: Promo pivot table ---
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

        # --- BOTTLENECK 3: Row iteration and processing ---
        for (store, item), s_sales in sales_wide.iterrows():
            s_sales = s_sales.reindex(window_idx, fill_value=0.0)

            # Align promo data
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

            # Weight lookup
            row[weight_col] = (
                weight_map.get((store, item))
                if isinstance(weight_map.index, pd.MultiIndex)
                else np.nan
            )

            # --- BOTTLENECK 4: Daily calculations ---
            for i, d in enumerate(window_idx[:window_size], start=1):
                curr = float(s_sales.loc[d]) if pd.notna(s_sales.loc[d]) else np.nan
                row[f"sales_day_{i}"] = curr
                row[f"{promo_col}_day_{i}"] = (
                    int(s_promo.loc[d]) if pd.notna(s_promo.loc[d]) else 0
                )

                # --- BOTTLENECK 5: Previous day lookup ---
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

        # Cleanup
        del sales_wide, promo_wide

    # --- Final DataFrame construction ---
    base_cols = ["start_date", "store_item", "store", "item", weight_col]
    sales_cols = [f"sales_day_{i}" for i in range(1, window_size + 1)]
    growth_cols = [f"growth_rate_{i}" for i in range(1, window_size + 1)]
    promo_cols = [f"{promo_col}_day_{i}" for i in range(1, window_size + 1)]
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


@profile
def pivot_table_benchmark(df: pd.DataFrame, window_dates):
    """Isolated benchmark of pivot table operations."""
    window_idx = pd.to_datetime(pd.Index(window_dates)).sort_values()
    w_df = df[df["date"].isin(window_idx)].copy()

    # Test different pivot approaches
    sales_wide = w_df.pivot_table(
        index=["store", "item"],
        columns="date",
        values="unit_sales",
        aggfunc="sum",
        fill_value=0.0,
    )

    return sales_wide


@profile
def growth_calculation_benchmark(sales_series, window_idx, df, store, item):
    """Isolated benchmark of growth rate calculations."""
    results = {}

    for i, d in enumerate(window_idx[:7], start=1):  # Test with window_size=7
        curr = float(sales_series.loc[d]) if pd.notna(sales_series.loc[d]) else np.nan

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
            prev = results.get(f"sales_day_{i-1}", np.nan)

        results[f"sales_day_{i}"] = curr
        results[f"growth_rate_{i}"] = (
            np.nan
            if (not np.isfinite(prev)) or prev == 0 or pd.isna(curr)
            else (curr - prev) / prev * 100.0
        )

    return results


def main():
    """Main function to run the profiled tests."""
    print("🔍 Running line-by-line profiler...")
    print(
        "Note: @profile decorators are active - use 'kernprof -l -v' for best results"
    )

    # Create test data
    df = create_test_data(n_stores=3, n_items=5, n_days=30)
    print(f"📊 Created test data: {len(df)} rows")

    # Run the profiled function
    result = generate_growth_rate_store_sku_feature_profiled(
        df, window_size=7, calendar_aligned=True
    )

    print(f"✅ Generated {len(result)} feature rows")

    # Run isolated benchmarks
    windows = generate_aligned_windows(df, 7, calendar_aligned=True)
    if windows:
        pivot_result = pivot_table_benchmark(df, windows[0])
        print(f"📊 Pivot benchmark: {pivot_result.shape}")

        if not pivot_result.empty:
            first_store_item = pivot_result.index[0]
            first_sales = pivot_result.iloc[0]
            window_idx = pd.to_datetime(pd.Index(windows[0])).sort_values()

            growth_result = growth_calculation_benchmark(
                first_sales, window_idx, df, first_store_item[0], first_store_item[1]
            )
            print(f"📈 Growth calculation benchmark: {len(growth_result)} calculations")


if __name__ == "__main__":
    main()
