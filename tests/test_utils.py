import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
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
    assert "storeClusterId" in result.columns
    assert "itemClusterId" in result.columns
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
    assert len(result) == 2  
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
    assert len(result) == 4  # 2 windows per item
    assert sorted(result["store_item"].unique()) == ["store1_item1", "store2_item2"]
    assert all(result.groupby("store_item")["start_date"].count() == 2)


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
        cyc_df,
        sales_df,
        on=[
            "start_date",
            "store_item",
            "store",
            "item",
            "storeClusterId",
            "itemClusterId",
        ],
    )
    result = add_next_window_targets(merged_df, window_size=window_size)

    y_cols = [col for col in result.columns if col.startswith("y_")]
    assert len(y_cols) == 6 + 20  #  6 sales + 5 cyc_feats × 2 trigs × 2 days
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
        cyc_df,
        sales_df,
        on=[
            "start_date",
            "store_item",
            "store",
            "item",
            "storeClusterId",
            "itemClusterId",
        ],
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
    (
        meta_cols,
        x_sales_features,
        x_cyclical_features,
        x_feature_cols,
        label_cols,
        y_sales_features,
        y_cyclical_features,
    ) = build_feature_and_label_cols(window_size=2)
    assert meta_cols == [
        "start_date",
        "store_item",
        "store",
        "item",
        "storeClusterId",
        "itemClusterId",
    ]
    assert x_sales_features == [
        "sales_day_1",
        "sales_day_2",
        "store_med_day_1",
        "store_med_day_2",
        "item_med_day_1",
        "item_med_day_2",
    ]
    assert x_cyclical_features == [
        "dayofweek_sin_1",
        "dayofweek_sin_2",
        "dayofweek_cos_1",
        "dayofweek_cos_2",
        "weekofmonth_sin_1",
        "weekofmonth_sin_2",
        "weekofmonth_cos_1",
        "weekofmonth_cos_2",
        "monthofyear_sin_1",
        "monthofyear_sin_2",
        "monthofyear_cos_1",
        "monthofyear_cos_2",
        "paycycle_sin_1",
        "paycycle_sin_2",
        "paycycle_cos_1",
        "paycycle_cos_2",
        "season_sin_1",
        "season_sin_2",
        "season_cos_1",
        "season_cos_2",
    ]
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
    assert len(x_feature_cols) == 6 + 20
    assert x_feature_cols == x_sales_features + x_cyclical_features
    assert len(label_cols) == len(x_feature_cols)
    assert label_cols == [f"y_{c}" for c in x_feature_cols]


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

    pytest.importorskip("sklearn")
    from sklearn.cluster import SpectralClustering

    result = generate_store_item_clusters(
        pivot, n_clusters=2, model_class=SpectralClustering
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
    pytest.importorskip("sklearn")
    from sklearn.cluster import SpectralClustering

    clusters = generate_store_item_clusters(
        pivot, n_clusters=1, model_class=SpectralClustering
    )
    # Duplicate labels to mimic store/item clusters
    store_clusters = clusters.copy()
    store_clusters["store"] = store_clusters["store_item"].str.split("_").str[0]
    store_clusters["clusterId"] = store_clusters["clusterId"]
    item_clusters = clusters.copy()
    item_clusters["item"] = item_clusters["store_item"].str.split("_").str[1]
    item_clusters["clusterId"] = item_clusters["clusterId"]

    result = generate_sales_features(
        df,
        window_size=2,
        store_clusters=store_clusters,
        item_clusters=item_clusters,
    )
    assert "store_med_day_1" in result.columns
    assert "item_med_day_1" in result.columns
    assert "store_med_day_2" in result.columns
    assert "item_med_day_2" in result.columns
    assert "storeClusterId" in result.columns
    assert "itemClusterId" in result.columns

    # All store_items should share the same cluster label (0)
    assert set(result["storeClusterId"]) == {0}
    assert set(result["itemClusterId"]) == {0}

    # Cluster median for first window day 1 should equal median of [1, 5]
    first_med = np.median([1, 5])
    first_row = result[result["start_date"] == df["date"].min()].iloc[0]
    assert first_row["store_med_day_1"] == first_med
    assert first_row["item_med_day_1"] == first_med
