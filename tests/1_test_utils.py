import pandas as pd
from datetime import datetime
import numpy as np
from src.utils import generate_sales_features, generate_cyclical_features

import pytest


def test_generate_sales_features_empty_df():
    """Test with empty DataFrame"""
    df = pd.DataFrame(columns=["store", "item", "date", "unit_sales"])
    result = generate_sales_features(df, window_size=1)
    assert result.empty
    assert "start_date" in result.columns
    assert "store_item" in result.columns
    assert "store" in result.columns
    assert "item" in result.columns
    assert "sales_day_1" in result.columns
    assert "store_med_day_1" in result.columns
    assert "item_med_day_1" in result.columns
    # Verify column order
    expected_columns = [
        "start_date",
        "store_item",
        "store",
        "item",
        "sales_day_1",
        "store_med_day_1",
        "item_med_day_1",
    ]
    assert list(result.columns) == expected_columns


def test_generate_nonoverlap_window_features_single_row():
    """Test with single row DataFrame"""
    df = pd.DataFrame(
        {
            "store": ["store1"],
            "item": ["item1"],
            "date": [datetime(2023, 1, 1)],
            "unit_sales": [100],
        }
    )
    result = generate_nonoverlap_window_features(df, window_size=1)
    assert len(result) == 1
    assert result["start_date"].iloc[0] == datetime(2023, 1, 1)
    assert result["store_item"].iloc[0] == "store1_item1"
    assert result["store"].iloc[0] == "store1"
    assert result["item"].iloc[0] == "item1"
    assert result["sales_day_1"].iloc[0] == 100
    assert result["store_med_day_1"].iloc[0] == 100
    assert result["item_med_day_1"].iloc[0] == 100
    # Verify column order
    expected_columns = [
        "start_date",
        "store_item",
        "store",
        "item",
        "sales_day_1",
        "store_med_day_1",
        "item_med_day_1",
    ]
    assert list(result.columns) == expected_columns
    # Verify column order
    expected_columns = [
        "start_date",
        "store_item",
        "store",
        "item",
        "sales_day_1",
        "store_med_day_1",
        "item_med_day_1",
    ]
    assert list(result.columns) == expected_columns


def test_generate_nonoverlap_window_features_single_window():
    """Test with exact window size"""
    df = pd.DataFrame(
        {
            "date": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            "store": ["store1", "store1"],
            "item": ["item1", "item1"],
            "unit_sales": [100, 200],
        }
    )
    result = generate_nonoverlap_window_features(df, window_size=2)
    assert len(result) == 1
    assert result["start_date"].iloc[0] == datetime(2023, 1, 1)
    assert result["store_item"].iloc[0] == "store1_item1"
    assert result["store"].iloc[0] == "store1"
    assert result["item"].iloc[0] == "item1"
    assert result["sales_day_1"].iloc[0] == 100
    assert result["sales_day_2"].iloc[0] == 200
    assert result["store_med_day_1"].iloc[0] == 100
    assert result["store_med_day_2"].iloc[0] == 200
    assert result["item_med_day_1"].iloc[0] == 100
    assert result["item_med_day_2"].iloc[0] == 200
    # Verify column order
    expected_columns = [
        "start_date",
        "store_item",
        "store",
        "item",
        "sales_day_1",
        "sales_day_2",
        "store_med_day_1",
        "store_med_day_2",
        "item_med_day_1",
        "item_med_day_2",
    ]
    assert list(result.columns) == expected_columns


def test_generate_nonoverlap_window_features_multiple_windows():
    """Test with multiple windows"""
    dates = pd.date_range(start="2023-01-01", periods=10)
    df = pd.DataFrame(
        {
            "store": ["store1"] * 10,
            "item": ["item1"] * 10,
            "date": dates,
            "unit_sales": range(100, 110),
        }
    )
    result = generate_nonoverlap_window_features(df, window_size=5)
    assert len(result) == 2
    assert result["start_date"].iloc[0] == datetime(2023, 1, 1)
    assert result["start_date"].iloc[1] == datetime(2023, 1, 6)
    assert result["store_item"].iloc[0] == "store1_item1"
    assert result["store_item"].iloc[1] == "store1_item1"
    assert result["sales_day_1"].iloc[0] == 100
    assert result["sales_day_5"].iloc[0] == 104
    assert result["sales_day_1"].iloc[1] == 105
    assert result["sales_day_5"].iloc[1] == 109
    # Verify column order
    expected_columns = [
        "start_date",
        "store_item",
        "store",
        "item",
        "sales_day_1",
        "sales_day_2",
        "sales_day_3",
        "sales_day_4",
        "sales_day_5",
        "store_med_day_1",
        "store_med_day_2",
        "store_med_day_3",
        "store_med_day_4",
        "store_med_day_5",
        "item_med_day_1",
        "item_med_day_2",
        "item_med_day_3",
        "item_med_day_4",
        "item_med_day_5",
    ]
    assert list(result.columns) == expected_columns


def test_generate_nonoverlap_window_features_different_stores_items():
    """Test with different stores and items"""
    dates = pd.date_range(start="2023-01-01", periods=5)
    df = pd.DataFrame(
        {
            "store": ["store1", "store1", "store2", "store2", "store2"],
            "item": ["item1", "item2", "item1", "item2", "item3"],
            "date": dates,
            "unit_sales": [100, 200, 300, 400, 500],
        }
    )
    result = generate_nonoverlap_window_features(df, window_size=5)
    assert len(result) == 5
    # Check store1_item1 window
    assert result.loc[0, "start_date"] == datetime(2023, 1, 1)
    assert result.loc[0, "store_item"] == "store1_item1"
    assert result.loc[0, "store"] == "store1"
    assert result.loc[0, "item"] == "item1"
    assert result.loc[0, "sales_day_1"] == 100
    assert result.loc[0, "store_med_day_1"] == 100
    assert result.loc[0, "item_med_day_1"] == 100
    # Check store2_item1 window
    assert result.loc[2, "start_date"] == datetime(2023, 1, 1)
    assert result.loc[2, "store_item"] == "store2_item1"
    assert result.loc[2, "store"] == "store2"
    assert result.loc[2, "item"] == "item1"
    assert result.loc[2, "sales_day_3"] == 300
    assert result.loc[2, "store_med_day_3"] == 300
    assert result.loc[2, "item_med_day_3"] == 300
    # Verify column order
    expected_columns = [
        "start_date",
        "store_item",
        "store",
        "item",
        "sales_day_1",
        "sales_day_2",
        "sales_day_3",
        "sales_day_4",
        "sales_day_5",
        "store_med_day_1",
        "store_med_day_2",
        "store_med_day_3",
        "store_med_day_4",
        "store_med_day_5",
        "item_med_day_1",
        "item_med_day_2",
        "item_med_day_3",
        "item_med_day_4",
        "item_med_day_5",
    ]
    assert list(result.columns) == expected_columns


def test_generate_nonoverlap_window_features_missing_dates():
    """Test with missing dates"""
    dates = pd.date_range(start="2023-01-01", periods=5)
    # Remove one date
    dates = dates.delete(2)
    df = pd.DataFrame(
        {
            "store": ["store1"] * 4,
            "item": ["item1"] * 4,
            "date": dates,
            "unit_sales": [100, 200, 300, 400],
        }
    )
    result = generate_nonoverlap_window_features(df, window_size=2)
    assert len(result) == 2
    assert result["start_date"].iloc[0] == dates[0]
    assert result["start_date"].iloc[1] == dates[2]
    assert result["sales_day_1"].iloc[0] == 100
    assert result["sales_day_2"].iloc[0] == 200
    assert result["sales_day_1"].iloc[1] == 300
    assert result["sales_day_2"].iloc[1] == 400
    # Verify column order
    expected_columns = [
        "start_date",
        "store_item",
        "store",
        "item",
        "sales_day_1",
        "sales_day_2",
        "store_med_day_1",
        "store_med_day_2",
        "item_med_day_1",
        "item_med_day_2",
    ]
    assert list(result.columns) == expected_columns


def test_generate_nonoverlap_window_features_large_window():
    """Test with window size larger than data"""
    df = pd.DataFrame(
        {
            "store": ["store1"] * 3,
            "item": ["item1"] * 3,
            "date": pd.date_range(start="2023-01-01", periods=3),
            "unit_sales": [100, 200, 300],
            "store_item": ["store1_item1"] * 3,
        }
    )
    result = generate_nonoverlap_window_features(df, window_size=5)
    assert len(result) == 1


def test_generate_nonoverlap_window_features_edge_dates():
    """Test with dates at year boundaries"""
    df = pd.DataFrame(
        {
            "store": ["store1"] * 3,
            "item": ["item1"] * 3,
            "date": [
                datetime(2023, 12, 31),
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
            ],
            "unit_sales": [100, 200, 300],
        }
    )
    result = generate_nonoverlap_window_features(df, window_size=3)
    assert len(result) == 1
    assert result["start_date"].iloc[0] == datetime(2023, 12, 31)
    assert result["store_item"].iloc[0] == "store1_item1"
    assert result["store"].iloc[0] == "store1"
    assert result["item"].iloc[0] == "item1"
    assert result["sales_day_1"].iloc[0] == 100
    assert result["sales_day_2"].iloc[0] == 200
    assert result["sales_day_3"].iloc[0] == 300
    # Verify column order
    expected_columns = [
        "start_date",
        "store_item",
        "store",
        "item",
        "sales_day_1",
        "sales_day_2",
        "sales_day_3",
        "store_med_day_1",
        "store_med_day_2",
        "store_med_day_3",
        "item_med_day_1",
        "item_med_day_2",
        "item_med_day_3",
    ]
    assert list(result.columns) == expected_columns


def test_generate_cyclical_features_empty_df():
    """Test with empty DataFrame"""
    df = pd.DataFrame(columns=["date", "store", "item"])
    result = generate_cyclical_features(df, window_size=2)
    assert len(result) == 0
    expected_columns = [
        "start_date",
        "store_item",
        "store",
        "item",
        "dayofweek_sin_1",
        "dayofweek_cos_1",
        "weekofmonth_sin_1",
        "weekofmonth_cos_1",
        "monthofyear_sin_1",
        "monthofyear_cos_1",
        "paycycle_sin_1",
        "paycycle_cos_1",
        "dayofweek_sin_2",
        "dayofweek_cos_2",
        "weekofmonth_sin_2",
        "weekofmonth_cos_2",
        "monthofyear_sin_2",
        "monthofyear_cos_2",
        "paycycle_sin_2",
        "paycycle_cos_2",
    ]
    assert list(result.columns) == expected_columns


def test_generate_cyclical_features_single_window():
    """Test with a single window of data using Monday=0, Sunday=6 convention"""
    df = pd.DataFrame(
        {
            "store": ["store1", "store1", "store1"],
            "item": ["item1", "item1", "item1"],
            "date": [
                datetime(2023, 1, 1),  # Sunday (6)
                datetime(2023, 1, 2),  # Monday (0)
                datetime(2023, 1, 3),  # Tuesday (1)
            ],
            "store_item": ["store1_item1"] * 3,
        }
    )
    result = generate_cyclical_features(df, window_size=3)

    assert len(result) == 1
    assert result["start_date"].iloc[0] == datetime(2023, 1, 1)
    assert result["store_item"].iloc[0] == "store1_item1"
    assert result["store"].iloc[0] == "store1"
    assert result["item"].iloc[0] == "item1"

    # Sunday (6): sin = sin(2π * 6 / 7), cos = cos(2π * 6 / 7)
    sin0 = np.sin(2 * np.pi * 6 / 7)
    cos0 = np.cos(2 * np.pi * 6 / 7)
    assert result["dayofweek_sin_1"].iloc[0] == pytest.approx(sin0, rel=1e-6)
    assert result["dayofweek_cos_1"].iloc[0] == pytest.approx(cos0, rel=1e-6)

    # Monday (0): sin = 0.0, cos = 1.0
    assert result["dayofweek_sin_2"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert result["dayofweek_cos_2"].iloc[0] == pytest.approx(1.0, abs=1e-6)

    # Tuesday (1)
    sin2 = np.sin(2 * np.pi * 1 / 7)
    cos2 = np.cos(2 * np.pi * 1 / 7)
    assert result["dayofweek_sin_3"].iloc[0] == pytest.approx(sin2, rel=1e-6)
    assert result["dayofweek_cos_3"].iloc[0] == pytest.approx(cos2, rel=1e-6)

    # Week of month = 1 for all (Jan 1–7)
    sin_wom = np.sin(2 * np.pi * 1 / 5)
    cos_wom = np.cos(2 * np.pi * 1 / 5)
    for i in range(1, 4):
        assert result[f"weekofmonth_sin_{i}"].iloc[0] == pytest.approx(
            sin_wom, rel=1e-6
        )
        assert result[f"weekofmonth_cos_{i}"].iloc[0] == pytest.approx(
            cos_wom, rel=1e-6
        )


def test_generate_cyclical_features_multiple_windows():
    """Test with multiple windows"""
    df = pd.DataFrame(
        {
            "store": ["store1"] * 5,
            "item": ["item1"] * 5,
            "date": [
                datetime(2023, 1, 1),  # Sunday (6)
                datetime(2023, 1, 2),  # Monday (0)
                datetime(2023, 1, 3),  # Tuesday (1)
                datetime(2023, 1, 4),  # Wednesday (2)
                datetime(2023, 1, 5),  # Thursday (3)
            ],
            "store_item": ["store1_item1"] * 5,
        }
    )
    result = generate_cyclical_features(df, window_size=3)
    assert len(result) == 2

    # First window: Jan 1-3
    assert result["start_date"].iloc[0] == datetime(2023, 1, 1)
    # Second window: Jan 4-5 (partial, padded)

    assert result["start_date"].iloc[1] == datetime(2023, 1, 4)

    # Expected dayofweek values
    days_first = [6, 0, 1]  # Sunday, Monday, Tuesday
    days_second = [2, 3, None]  # Wednesday, Thursday, padding

    for i, day in enumerate(days_first):
        sin_expected = np.sin(2 * np.pi * day / 7)
        assert result[f"dayofweek_sin_{i+1}"].iloc[0] == pytest.approx(
            sin_expected, rel=1e-6
        )

    for i, day in enumerate(days_second):
        if day is not None:
            sin_expected = np.sin(2 * np.pi * day / 7)
        else:
            sin_expected = 0.0
        assert result[f"dayofweek_sin_{i+1}"].iloc[1] == pytest.approx(
            sin_expected, rel=1e-6
        )


def test_generate_cyclical_features_different_stores():
    """Test with different stores"""
    df = pd.DataFrame(
        {
            "store": ["store1", "store1", "store2", "store2"],
            "item": ["item1", "item1", "item2", "item2"],
            "date": [
                datetime(2023, 1, 1),  # Sunday (6)
                datetime(2023, 1, 2),  # Monday (0)
                datetime(2023, 1, 1),  # Sunday (6)
                datetime(2023, 1, 2),  # Monday (0)
            ],
        }
    )

    result = generate_cyclical_features(df, window_size=2)
    assert len(result) == 2

    # Compute expected sine values
    sin_sunday = np.sin(2 * np.pi * 6 / 7)
    sin_monday = np.sin(2 * np.pi * 0 / 7)

    # First group: store1_item1
    assert result["store_item"].iloc[0] == "store1_item1"
    assert result["start_date"].iloc[0] == datetime(2023, 1, 1)
    assert result["dayofweek_sin_1"].iloc[0] == pytest.approx(sin_sunday, rel=1e-6)
    assert result["dayofweek_sin_2"].iloc[0] == pytest.approx(sin_monday, rel=1e-6)

    # Second group: store2_item2
    assert result["store_item"].iloc[1] == "store2_item2"
    assert result["start_date"].iloc[1] == datetime(2023, 1, 1)
    assert result["dayofweek_sin_1"].iloc[1] == pytest.approx(sin_sunday, rel=1e-6)
    assert result["dayofweek_sin_2"].iloc[1] == pytest.approx(sin_monday, rel=1e-6)
