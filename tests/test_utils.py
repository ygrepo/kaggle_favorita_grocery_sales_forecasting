import pandas as pd
from datetime import datetime
from src.utils import generate_nonoverlap_window_features


def test_generate_nonoverlap_window_features_empty_df():
    """Test with empty DataFrame"""
    df = pd.DataFrame(columns=["store", "item", "date", "unit_sales"])
    result = generate_nonoverlap_window_features(df, window_size=1)
    assert result.empty
    assert "store_item" in result.columns
    assert "store" in result.columns
    assert "item" in result.columns
    assert "sales_day_1" in result.columns
    assert "store_med_day_1" in result.columns
    assert "item_med_day_1" in result.columns
    # Verify column order
    expected_columns = [
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
    assert result["store_item"].iloc[0] == "store1_item1"
    assert result["store"].iloc[0] == "store1"
    assert result["item"].iloc[0] == "item1"
    assert result["sales_day_1"].iloc[0] == 100
    assert result["store_med_day_1"].iloc[0] == 100
    assert result["item_med_day_1"].iloc[0] == 100


def test_generate_nonoverlap_window_features_single_window():
    """Test with exact window size"""
    df = pd.DataFrame(
        {
            "store": ["store1", "store1"],
            "item": ["item1", "item1"],
            "date": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            "unit_sales": [100, 200],
        }
    )
    result = generate_nonoverlap_window_features(df, window_size=2)
    assert len(result) == 1
    assert result["store_item"].iloc[0] == "store1_item1"
    assert result["store"].iloc[0] == "store1"
    assert result["item"].iloc[0] == "item1"
    assert result["sales_day_1"].iloc[0] == 100
    assert result["sales_day_2"].iloc[0] == 200
    assert result["store_med_day_1"].iloc[0] == 100
    assert result["store_med_day_2"].iloc[0] == 200
    assert result["item_med_day_1"].iloc[0] == 100
    assert result["item_med_day_2"].iloc[0] == 200


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
    assert result["store_item"].iloc[0] == "store1_item1"
    assert result["store_item"].iloc[1] == "store1_item1"
    assert result["sales_day_1"].iloc[0] == 100
    assert result["sales_day_5"].iloc[0] == 104
    assert result["sales_day_1"].iloc[1] == 105
    assert result["sales_day_5"].iloc[1] == 109


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
    assert len(result) == 3
    # Check store1_item1 window
    assert result.loc[0, "store_item"] == "store1_item1"
    assert result.loc[0, "store"] == "store1"
    assert result.loc[0, "item"] == "item1"
    assert result.loc[0, "sales_day_1"] == 100
    assert result.loc[0, "store_med_day_1"] == 100
    assert result.loc[0, "item_med_day_1"] == 100
    # Check store2_item1 window
    assert result.loc[1, "store_item"] == "store2_item1"
    assert result.loc[1, "store"] == "store2"
    assert result.loc[1, "item"] == "item1"
    assert result.loc[1, "sales_day_1"] == 300
    assert result.loc[1, "store_med_day_1"] == 300
    assert result.loc[1, "item_med_day_1"] == 300


def test_generate_nonoverlap_window_features_missing_dates():
    """Test with missing dates"""
    dates = pd.date_range(start="2023-01-01", periods=5)
    # Remove one date
    dates = dates.drop(2)
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
    assert result["sales_day_1"].iloc[0] == 100
    assert result["sales_day_2"].iloc[0] == 200
    assert result["sales_day_1"].iloc[1] == 300
    assert result["sales_day_2"].iloc[1] == 400


def test_generate_nonoverlap_window_features_large_window():
    """Test with window size larger than data"""
    df = pd.DataFrame(
        {
            "store": ["store1"] * 3,
            "item": ["item1"] * 3,
            "date": pd.date_range(start="2023-01-01", periods=3),
            "unit_sales": [100, 200, 300],
        }
    )
    result = generate_nonoverlap_window_features(df, window_size=5)
    assert len(result) == 0


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
    assert result["store_item"].iloc[0] == "store1_item1"
    assert result["store"].iloc[0] == "store1"
    assert result["item"].iloc[0] == "item1"
    assert result["sales_day_1"].iloc[0] == 100
    assert result["sales_day_2"].iloc[0] == 200
    assert result["sales_day_3"].iloc[0] == 300


# if __name__ == '__main__':
#     pytest.main([__file__])
