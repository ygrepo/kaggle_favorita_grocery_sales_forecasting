import pandas as pd
import numpy as np
from datetime import datetime
from src.utils import (
    generate_sales_features,
    generate_cyclical_features,
    add_next_window_targets,
    build_feature_and_label_cols,
    generate_store_item_clusters,
)


def test_generate_sales_features_single_item():
    """Sales feature window test for one store/item using aligned windows."""
    df = pd.DataFrame(
        {
            "store": ["store1"] * 10,
            "item": ["item1"] * 10,
            "date": pd.date_range(start="2023-01-01", periods=10),
            "unit_sales": list(range(100, 110)),
        }
    )
    result = generate_sales_features(df, window_size=5)
    assert len(result) == 2  # 10 days -> two non-overlapping windows
    # assert "cluster_id" in result.columns
    # assert result["cluster_id"].isna().all()
    assert result["store_item"].nunique() == 1
    assert result["sales_day_1"].iloc[0] == 100
    assert result["sales_day_5"].iloc[-1] == 109


def test_generate_sales_features_multiple_items():
    """Sales features for multiple store/items with aligned windows."""
    df = pd.DataFrame(
        {
            "store": ["store1"] * 6 + ["store2"] * 6,
            "item": ["item1"] * 6 + ["item2"] * 6,
            "date": list(pd.date_range("2023-01-01", periods=6)) * 2,
            "unit_sales": list(range(100, 106)) + list(range(200, 206)),
        }
    )
    result = generate_sales_features(df, window_size=3)
    assert len(result) == 4  # two windows per item
    # assert "cluster_id" in result.columns
    # assert result["cluster_id"].isna().all()
    assert sorted(result["store_item"].unique()) == ["store1_item1", "store2_item2"]
    assert all(result.groupby("store_item")["start_date"].count() == 2)


def test_generate_sales_features_insufficient_data():
    """Should return 0 rows if window is larger than available dates."""
    df = pd.DataFrame(
        {
            "store": ["store1"] * 3,
            "item": ["item1"] * 3,
            "date": pd.date_range("2023-01-01", periods=3),
            "unit_sales": [100, 200, 300],
        }
    )
    result = generate_sales_features(df, window_size=5)
    assert result.empty
    # assert "cluster_id" in result.columns


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
    sales_df = generate_sales_features(df, window_size=3)
    merged_df = pd.merge(
        cyc_df, sales_df, on=["start_date", "store_item", "store", "item"]
    )

    result = add_next_window_targets(merged_df, window_size=3)

    # Should have 2 input rows, 1 target row (last has NaNs)
    assert len(result) == 2

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
    sales_df = generate_sales_features(df, window_size=window_size)
    merged_df = pd.merge(
        cyc_df, sales_df, on=["start_date", "store_item", "store", "item"]
    )
    result = add_next_window_targets(merged_df, window_size=window_size)

    y_cols = [col for col in result.columns if col.startswith("y_")]
    assert len(y_cols) == 6 + 20  # 3 sales + 5 cyc_feats × 2 trigs × 2 days
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
    sales_df = generate_sales_features(df, window_size=window_size)
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


def test_build_feature_and_label_cols():
    meta_cols, feature_cols, label_cols, y_sales_features, y_cyclical_features = (
        build_feature_and_label_cols(window_size=2)
    )
    assert meta_cols == ["start_date", "store_item", "store", "item"]
    assert y_sales_features == [
        "y_sales_day_1",
        "y_sales_day_2",
        "y_store_med_day_1",
        "y_store_med_day_2",
        "y_item_med_day_1",
        "y_item_med_day_2",
    ]
    assert y_cyclical_features == [
        "y_dayofweek_sin_1",
        "y_dayofweek_sin_2",
        "y_dayofweek_cos_1",
        "y_dayofweek_cos_2",
        "y_weekofmonth_sin_1",
        "y_weekofmonth_sin_2",
        "y_weekofmonth_cos_1",
        "y_weekofmonth_cos_2",
        "y_monthofyear_sin_1",
        "y_monthofyear_sin_2",
        "y_monthofyear_cos_1",
        "y_monthofyear_cos_2",
        "y_paycycle_sin_1",
        "y_paycycle_sin_2",
        "y_paycycle_cos_1",
        "y_paycycle_cos_2",
        "y_season_sin_1",
        "y_season_sin_2",
        "y_season_cos_1",
        "y_season_cos_2",
    ]
    assert feature_cols[0] == "sales_day_1"
    assert feature_cols[1] == "sales_day_2"
    assert feature_cols[2] == "store_med_day_1"
    assert feature_cols[3] == "store_med_day_2"
    assert feature_cols[4] == "item_med_day_1"
    assert feature_cols[5] == "item_med_day_2"
    assert feature_cols[6] == "dayofweek_sin_1"
    assert feature_cols[7] == "dayofweek_sin_2"
    assert feature_cols[8] == "dayofweek_cos_1"
    assert feature_cols[9] == "dayofweek_cos_2"
    assert feature_cols[10] == "weekofmonth_sin_1"
    assert feature_cols[11] == "weekofmonth_sin_2"
    assert feature_cols[12] == "weekofmonth_cos_1"
    assert feature_cols[13] == "weekofmonth_cos_2"
    assert feature_cols[14] == "monthofyear_sin_1"
    assert feature_cols[15] == "monthofyear_sin_2"
    assert feature_cols[16] == "monthofyear_cos_1"
    assert feature_cols[17] == "monthofyear_cos_2"
    assert feature_cols[18] == "paycycle_sin_1"
    assert feature_cols[19] == "paycycle_sin_2"
    assert feature_cols[20] == "paycycle_cos_1"
    assert feature_cols[21] == "paycycle_cos_2"
    assert feature_cols[22] == "season_sin_1"
    assert feature_cols[23] == "season_sin_2"
    assert feature_cols[24] == "season_cos_1"
    assert feature_cols[25] == "season_cos_2"
    assert len(feature_cols) == 6 + 20
    assert label_cols[0] == "y_sales_day_1"
    assert label_cols[1] == "y_sales_day_2"
    assert label_cols[2] == "y_store_med_day_1"
    assert label_cols[3] == "y_store_med_day_2"
    assert label_cols[4] == "y_item_med_day_1"
    assert label_cols[5] == "y_item_med_day_2"
    assert label_cols[6] == "y_dayofweek_sin_1"
    assert label_cols[7] == "y_dayofweek_sin_2"
    assert label_cols[8] == "y_dayofweek_cos_1"
    assert label_cols[9] == "y_dayofweek_cos_2"
    assert label_cols[10] == "y_weekofmonth_sin_1"
    assert label_cols[11] == "y_weekofmonth_sin_2"
    assert label_cols[12] == "y_weekofmonth_cos_1"
    assert label_cols[13] == "y_weekofmonth_cos_2"
    assert label_cols[14] == "y_monthofyear_sin_1"
    assert label_cols[15] == "y_monthofyear_sin_2"
    assert label_cols[16] == "y_monthofyear_cos_1"
    assert label_cols[17] == "y_monthofyear_cos_2"
    assert label_cols[18] == "y_paycycle_sin_1"
    assert label_cols[19] == "y_paycycle_sin_2"
    assert label_cols[20] == "y_paycycle_cos_1"
    assert label_cols[21] == "y_paycycle_cos_2"
    assert label_cols[22] == "y_season_sin_1"
    assert label_cols[23] == "y_season_sin_2"
    assert label_cols[24] == "y_season_cos_1"
    assert label_cols[25] == "y_season_cos_2"
    assert len(label_cols) == len(feature_cols)


def test_generate_store_item_clusters_basic():
    pivot = pd.DataFrame(
        [
            [1, 2],
            [2, 1],
            [10, 10],
            [11, 11],
        ],
        index=["s1_i1", "s1_i2", "s2_i1", "s2_i2"],
    )

    from sklearn.cluster import KMeans

    result = generate_store_item_clusters(
        pivot, n_clusters=2, cluster_algo=KMeans(random_state=42, n_init="auto")
    )

    assert list(result.columns) == ["store_item", "clusterId"]
    assert len(result) == 4
    assert set(result["clusterId"]).issubset({0, 1})


def test_generate_sales_features_with_clusters():
    """Ensure cluster medians are computed when cluster_map provided."""
    df = pd.DataFrame(
        {
            "store": ["s1"] * 4 + ["s2"] * 4,
            "item": ["i1"] * 4 + ["i2"] * 4,
            "date": list(pd.date_range("2023-01-01", periods=4)) * 2,
            "unit_sales": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    pivot = df.pivot_table(
        index=df["store"] + "_" + df["item"],
        columns="date",
        values="unit_sales",
    )
    from sklearn.cluster import SpectralClustering

    clusters = generate_store_item_clusters(
        pivot, n_clusters=1, model_class=SpectralClustering(random_state=0)
    )
    # Duplicate labels to mimic store/item clusters
    clusters["store_cluster_id"] = clusters["clusterId"]
    clusters["item_cluster_id"] = clusters["clusterId"]

    result = generate_sales_features(df, window_size=2, cluster_map=clusters)
    assert "cluster_id" in result.columns
    assert "cluster_med_day_1" in result.columns
    assert "store_cluster_med_day_1" in result.columns
    assert "item_cluster_med_day_1" in result.columns
    # All store_items should share the same cluster
    assert result["cluster_id"].nunique() == 1
    # Cluster median for first window day 1 should equal median of [1,5]
    first_med = np.median([1, 5])
    assert (
        result.loc[result["start_date"] == df["date"].min(), "cluster_med_day_1"].iloc[
            0
        ]
        == first_med
    )
