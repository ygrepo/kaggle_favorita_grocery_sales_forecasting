import pandas as pd
from datetime import datetime
import numpy as np
from src.utils import (
    generate_nonoverlap_window_features,
    generate_cyclical_features,
    add_next_window_targets,
)


def test_sliding_windows_single_item_multiple_windows():
    """Sliding window test for one store/item with multiple overlapping windows."""
    df = pd.DataFrame(
        {
            "store": ["store1"] * 10,
            "item": ["item1"] * 10,
            "date": pd.date_range(start="2023-01-01", periods=10),
            "unit_sales": list(range(100, 110)),
        }
    )
    result = generate_nonoverlap_window_features(df, window_size=5)
    assert len(result) == 6  # 10 - 5 + 1 = 6 sliding windows
    assert result["store_item"].nunique() == 1
    assert result["sales_day_1"].iloc[0] == 100
    assert result["sales_day_5"].iloc[-1] == 109


def test_sliding_windows_multiple_items():
    """Sliding window test for multiple store/items."""
    df = pd.DataFrame(
        {
            "store": ["store1"] * 6 + ["store2"] * 6,
            "item": ["item1"] * 6 + ["item2"] * 6,
            "date": list(pd.date_range("2023-01-01", periods=6)) * 2,
            "unit_sales": list(range(100, 106)) + list(range(200, 206)),
        }
    )
    result = generate_nonoverlap_window_features(df, window_size=3)
    assert len(result) == 8  # 4 per item
    assert sorted(result["store_item"].unique()) == ["store1_item1", "store2_item2"]
    assert all(result.groupby("store_item")["start_date"].count() == 4)


def test_sliding_windows_insufficient_data():
    """Should return 0 rows if window is larger than the number of dates."""
    df = pd.DataFrame(
        {
            "store": ["store1"] * 3,
            "item": ["item1"] * 3,
            "date": pd.date_range("2023-01-01", periods=3),
            "unit_sales": [100, 200, 300],
        }
    )
    result = generate_nonoverlap_window_features(df, window_size=5)
    assert result.empty


def test_sliding_cyclical_single_item_multiple_windows():
    """Sliding cyclical window test for one store/item with overlapping windows."""
    df = pd.DataFrame(
        {
            "store": ["store1"] * 10,
            "item": ["item1"] * 10,
            "date": pd.date_range(start="2023-01-01", periods=10),
        }
    )
    result = generate_cyclical_features(df, window_size=5)
    assert len(result) == 6  # 10 - 5 + 1 = 6 windows
    assert result["store_item"].nunique() == 1
    assert result["start_date"].iloc[0] == datetime(2023, 1, 1)
    assert result["start_date"].iloc[-1] == datetime(2023, 1, 6)


def test_sliding_cyclical_multiple_items():
    """Sliding cyclical window test for multiple store/items."""
    df = pd.DataFrame(
        {
            "store": ["store1"] * 6 + ["store2"] * 6,
            "item": ["item1"] * 6 + ["item2"] * 6,
            "date": list(pd.date_range("2023-01-01", periods=6)) * 2,
        }
    )
    result = generate_cyclical_features(df, window_size=3)
    assert len(result) == 8  # 4 per item
    assert sorted(result["store_item"].unique()) == ["store1_item1", "store2_item2"]
    assert all(result.groupby("store_item")["start_date"].count() == 4)


def test_sliding_cyclical_insufficient_data():
    """Should return 0 rows if window is larger than available dates."""
    df = pd.DataFrame(
        {
            "store": ["store1"] * 2,
            "item": ["item1"] * 2,
            "date": pd.date_range("2023-01-01", periods=2),
        }
    )
    result = generate_cyclical_features(df, window_size=5)
    assert result.empty


def test_add_next_window_targets_basic_shift():
    """Test next-window target shifting for sales and cyclical features."""
    df = pd.DataFrame(
        {
            "store": ["store1"] * 6,
            "item": ["item1"] * 6,
            "date": pd.date_range("2023-01-01", periods=6),
            "unit_sales": [10, 20, 30, 40, 50, 60],
        }
    )

    # Generate merged feature set
    cyc_df = generate_cyclical_features(df, window_size=3)
    sales_df = generate_nonoverlap_window_features(df, window_size=3)
    merged_df = pd.merge(
        cyc_df, sales_df, on=["start_date", "store_item", "store", "item"]
    )

    result = add_next_window_targets(merged_df, window_size=3)

    # Should have 4 input rows, 3 target rows (last has NaNs)
    assert len(result) == 4

    # First row's target sales should match second row's input sales
    for i in range(1, 4):
        input_val = result[f"sales_day_{i}"].iloc[1]
        target_val = result[f"y_sales_day_{i}"].iloc[0]
        assert input_val == target_val

    # Last row should have NaNs in y_*
    assert (
        result.iloc[-1][[col for col in result.columns if col.startswith("y_")]]
        .isna()
        .all()
    )


def test_add_next_window_targets_column_integrity():
    """Ensure y_ prefixed columns are present and match expected count."""
    df = pd.DataFrame(
        {
            "store": ["store1"] * 5,
            "item": ["item1"] * 5,
            "date": pd.date_range("2023-01-01", periods=5),
            "unit_sales": [10, 20, 30, 40, 50],
        }
    )

    window_size = 2
    cyc_df = generate_cyclical_features(df, window_size=window_size)
    sales_df = generate_nonoverlap_window_features(df, window_size=window_size)
    merged_df = pd.merge(
        cyc_df, sales_df, on=["start_date", "store_item", "store", "item"]
    )
    result = add_next_window_targets(merged_df, window_size=window_size)

    y_cols = [col for col in result.columns if col.startswith("y_")]
    assert len(y_cols) == 2 + 3 * 2 * 2  # 2 sales + 3 cyc_feats × 2 trigs × 2 days
    assert all(col in result.columns for col in y_cols)


def test_add_next_window_targets_drop_nan_rows():
    """Ensure we can drop rows with incomplete target values after shifting."""
    df = pd.DataFrame(
        {
            "store": ["store1"] * 6,
            "item": ["item1"] * 6,
            "date": pd.date_range("2023-01-01", periods=6),
            "unit_sales": [10, 20, 30, 40, 50, 60],
        }
    )

    window_size = 3
    cyc_df = generate_cyclical_features(df, window_size=window_size)
    sales_df = generate_nonoverlap_window_features(df, window_size=window_size)
    merged_df = pd.merge(
        cyc_df, sales_df, on=["start_date", "store_item", "store", "item"]
    )

    result = add_next_window_targets(merged_df, window_size=window_size)

    # Drop rows where any y_ column has NaN
    y_cols = [col for col in result.columns if col.startswith("y_")]
    cleaned = result.dropna(subset=y_cols)

    # The last row should be dropped
    assert len(cleaned) == len(result) - 1

    # Check all y_ columns in cleaned data are fully non-null
    assert cleaned[y_cols].notna().all().all()
