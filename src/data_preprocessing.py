import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def top_n_by_m(df, n_col="unit_sales", group_column="store_nbr", top_n=10):
    """
    Returns the top N stores by total unit sales.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        n_col (str): Column representing sales values.
        group_column (str): Column to group by (e.g., store number).
        top_n (int): Number of top results to return.

    Returns:
        pd.DataFrame: DataFrame of top N stores by total sales.
    """
    return (
        df.groupby(group_column)
        .agg({n_col: "sum"})
        .sort_values(n_col, ascending=False)
        .head(top_n)
    )


def top_values_with_percentage(df, group_column, value_column, n=5):
    """
    Returns the top N values with percentages for each group in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        group_column (str): The column to group by.
        value_column (str): The column to calculate percentages from.
        n (int): The number of top values to return.

    Returns:
        pd.DataFrame: A DataFrame containing the top N values and their percentages for each group.
    """
    grouped = df.groupby(group_column)[value_column].value_counts(normalize=True) * 100
    grouped = grouped.rename("percentage").reset_index()
    top_n = (
        grouped.groupby(group_column)
        .apply(lambda x: x.nlargest(n, "percentage"))
        .reset_index(drop=True)
    )
    return top_n


def value_counts_with_percentage(df, column_name, top_n=10):
    """
    Computes value counts and percentage distribution of a column.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of the column to analyze.

    Returns:
        pd.DataFrame: DataFrame with counts and percentages.
    """
    counts = df[column_name].value_counts()
    percentages = df[column_name].value_counts(normalize=True) * 100
    df = pd.DataFrame(
        {column_name + "_count": counts, column_name + "_percentage": percentages}
    )
    return df.sort_values(column_name, ascending=False).head(top_n)


def count_percent(series, n=3):
    counts = series.value_counts().head(n)
    percentages = counts / series.count() * 100
    result = pd.DataFrame({"Count": counts, "Percentage": percentages})
    return result


def select_extreme_and_median_neighbors(
    df,
    n_col="unit_sales",
    group_column="store_nbr",
    M=0,
    m=0,
    med=0,
    fn: str = None,
):
    """
    Returns M highest, m lowest, and 2*med around the median total sales groups.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        n_col (str): Column to sum (e.g., 'unit_sales').
        group_column (str): Column to group by (e.g., 'store_nbr').
        M (int): Number of top (max) groups to return.
        m (int): Number of bottom (min) groups to return.
        med (int): Number of groups to return on each side of the median.

    Returns:
        pd.DataFrame: Combined DataFrame of selected groups.
    """
    df = df.copy()
    logger.info(f"Selected {len(df)} groups")
    grouped = (
        df.groupby(group_column).agg({n_col: "sum"}).rename(columns={n_col: "total"})
    )
    sorted_grouped = grouped.sort_values("total")

    if M == 0 and m == 0 and med == 0:
        return sorted_grouped

    # Get extremes
    if m > 0:
        logger.info(f"Selecting {m} bottom groups")
        bottom_m = sorted_grouped.head(m)
        logger.info(f"Selected {len(bottom_m)} bottom groups")
    else:
        bottom_m = pd.DataFrame()
    if M > 0:
        logger.info(f"Selecting {M} top groups")
        top_M = sorted_grouped.tail(M)
        logger.info(f"Selected {len(top_M)} top groups")
    else:
        top_M = pd.DataFrame()

    # Find median-centered indices
    if med > 0:
        logger.info(f"Selecting {2*med} median groups")
        median_val = grouped["total"].median()

        # Compute absolute difference to median
        grouped["dist_to_median"] = np.abs(grouped["total"] - median_val)

        # Get 2*med rows closest to the median (excluding exact duplicates)
        median_neighbors = (
            grouped.sort_values("dist_to_median")
            .drop(index=bottom_m.index.union(top_M.index), errors="ignore")
            .head(2 * med)
            .drop(columns="dist_to_median")
        )
        logger.info(f"Selected {len(median_neighbors)} median groups")
    else:
        median_neighbors = pd.DataFrame()

    # Set indices for each group
    top_idx = set(top_M.index)
    bottom_idx = set(bottom_m.index)
    med_idx = set(median_neighbors.index)

    # Intersections
    top_and_bottom = top_idx & bottom_idx
    top_and_med = top_idx & med_idx
    bottom_and_med = bottom_idx & med_idx

    # Union of all indices
    all_indices = top_idx | bottom_idx | med_idx

    logger.info(f"Top ∩ Bottom: {len(top_and_bottom)}")
    logger.info(f"Top ∩ Median: {len(top_and_med)}")
    logger.info(f"Bottom ∩ Median: {len(bottom_and_med)}")
    logger.info(f"Total unique groups: {len(all_indices)}")

    # Combine and remove duplicates
    result = pd.concat([bottom_m, median_neighbors, top_M])
    # .drop_duplicates()
    if fn:
        logger.info(f"Saving selected groups to {fn}")
        result.to_csv(fn, index=True)
    return result


def prepare_data(
    df,
    group_store_column="store",
    group_item_column="item",
    value_column="unit_sales",
    store_top_n=0,
    store_med_n=0,
    store_bottom_n=0,
    item_top_n=0,
    item_med_n=0,
    item_bottom_n=0,
    item_fn: str = None,
    store_fn: str = None,
    fn: str = None,
):
    """
    Prepares a complete daily-level (store, item, date) grid for the top-N stores and globally top-M items.
    Fills missing (store, item, date) rows with unit_sales = -1, keeping all other columns (e.g., onpromotion, id) as NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sales data with at least: 'date', 'store', 'item', 'unit_sales'
    group_column : str
        Column used to select top-N groups (typically "store")
    value_column : str
        The sales value to aggregate (typically "unit_sales")
    store_top_n : int
        Number of top stores to retain
    store_med_n : int
        Number of stores to retain on each side of the median
    store_bottom_n : int
        Number of bottom stores to retain
    item_top_n : int
        Number of globally top items to retain
    item_med_n : int
        Number of items to retain on each side of the median
    item_bottom_n : int
        Number of bottom items to retain
    item_fn : str or None
        If given, saves the selected items to this path
    store_fn : str or None
        If given, saves the selected stores to this path
    fn : str or None
        If given, saves the resulting DataFrame to this path

    Returns
    -------
    pd.DataFrame
        Full (store, item, date) matrix with unit_sales filled as needed
    """
    df = df.copy()

    n_items = df[group_item_column].nunique()
    logger.info(f"# unique items: {n_items}")

    # Select top-M items globally
    df_top_items = select_extreme_and_median_neighbors(
        df,
        n_col=value_column,
        group_column=group_item_column,
        M=item_top_n,
        m=item_bottom_n,
        med=item_med_n,
        fn=item_fn,
    )
    logger.info(f"Selected top items: {df_top_items.head()}")
    valid_items = df_top_items.reset_index()[group_item_column].tolist()
    logger.info(f"# top items: {len(valid_items)}")

    n_stores = df[group_store_column].nunique()
    logger.info(f"# unique stores: {n_stores}")
    df_top_stores = select_extreme_and_median_neighbors(
        df,
        n_col=value_column,
        group_column=group_store_column,
        M=store_top_n,
        m=store_bottom_n,
        med=store_med_n,
        fn=store_fn,
    )
    logger.info(f"Selected top stores: {df_top_stores.head()}")
    valid_stores = df_top_stores.reset_index()[group_store_column].tolist()
    logger.info(f"# top stores: {len(valid_stores)}")
    unique_dates = df["date"].dropna().unique()
    grid = pd.MultiIndex.from_product(
        [valid_stores, valid_items, sorted(unique_dates)],
        names=["store", "item", "date"],
    ).to_frame(index=False)

    # Merge with filtered data
    df = pd.merge(grid, df, on=["store", "item", "date"], how="left")

    # Fill missing unit_sales with NaN
    missing_mask = df[value_column].isna()
    num_missing = missing_mask.sum()
    df.loc[missing_mask, value_column] = np.nan

    # Logging
    logger.info(
        f"Filled {num_missing} missing (store, item, date) rows with unit_sales = NaN"
    )
    logger.info(f"Final dataset shape: {df.shape}")
    logger.info(f"Unique stores: {df['store'].nunique()}")
    logger.info(f"Unique items: {df['item'].nunique()}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    if fn:
        logger.info(f"Saving final_df to {fn}")
        df.to_csv(fn, index=False)

    return df
