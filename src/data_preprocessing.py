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


def prepare_data(
    df,
    group_column="store",
    value_column="unit_sales",
    top_stores_n=10,
    top_items_n=500,
    fn: str = None,
):
    """
    Prepares a complete daily-level (store, item, date) grid for the top-N stores and top-M items.
    Fills missing (store, item, date) rows with unit_sales = -1, keeping other columns (e.g., onpromotion) as NaN.
    """
    df = df.copy()

    # Step 1: Select top-N stores
    df_top_stores = top_n_by_m(
        df, n_col=value_column, group_column=group_column, top_n=top_stores_n
    )
    valid_stores = df_top_stores.reset_index()[group_column].tolist()
    df = df[df["store"].isin(valid_stores)]

    # Step 2: Select top-M items among top stores only
    df_top_items = top_n_by_m(
        df, n_col=value_column, group_column="item", top_n=top_items_n
    )
    valid_items = df_top_items.reset_index()["item"].tolist()
    df = df[df["item"].isin(valid_items)]

    # Step 3: Create full (store, item, date) grid
    unique_dates = df["date"].dropna().unique()
    grid = pd.MultiIndex.from_product(
        [valid_stores, valid_items, sorted(unique_dates)],
        names=["store", "item", "date"],
    ).to_frame(index=False)

    # Step 4: Merge full grid with filtered daily data
    df = pd.merge(grid, df, on=["store", "item", "date"], how="left")

    # Step 5: Fill missing unit_sales with -1
    missing_mask = df[value_column].isna()
    num_missing = missing_mask.sum()
    df.loc[missing_mask, value_column] = -1

    # Optional: Add store_item composite key
    # df["store_item"] = df["store"].astype(str) + "_" + df["item"].astype(str)

    logger.info(
        f"Filled {num_missing} missing (store, item, date) rows with unit_sales = -1"
    )
    logger.info(f"Final dataset shape: {df.shape}")
    logger.info(f"Unique stores: {df['store'].nunique()}")
    logger.info(f"Unique items: {df['item'].nunique()}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    if fn:
        logger.info(f"Saving final_df to {fn}")
        df.to_csv(fn, index=False)

    return df


# def prepare_data(
#     df,
#     group_column="store",
#     value_column="unit_sales",
#     top_stores_n=10,
#     top_items_n=500,
#     fn: str = None,
# ):
#     """
#     Prepares the data by computing value counts and percentages for each group.

#     Parameters:
#         df (pd.DataFrame): Input DataFrame.
#         group_column (str): Column to group by.
#         value_column (str): Column to calculate percentages from.
#         top_stores_n (int): Number of top stores to return.
#         top_items_n (int): Number of top items to return.

#     Returns:
#         pd.DataFrame: DataFrame with counts and percentages for each group.
#     """
#     df = df.copy()
#     df_top_stores = top_n_by_m(
#         df, n_col=value_column, group_column=group_column, top_n=top_stores_n
#     )
#     valid_stores = df_top_stores.reset_index()[group_column].tolist()
#     df_top_stores = df[df[group_column].isin(valid_stores)]
#     df_top_stores = df_top_stores.reset_index()
#     df_top_stores.drop(["index"], axis=1, inplace=True)
#     logger.info(df_top_stores.head())
#     df_top_items = top_n_by_m(
#         df_top_stores, n_col=value_column, group_column="item", top_n=top_items_n
#     )
#     valid_items = df_top_items.reset_index()["item"].tolist()
#     df_top_items = df_top_stores[df_top_stores["item"].isin(valid_items)]
#     df_top_items = df_top_items.reset_index()
#     df_top_items.drop(["index"], axis=1, inplace=True)
#     logger.info(df_top_items.head())

#     logger.info(f"Number of rows: {len(df_top_items)}")
#     logger.info(f"Number of unique stores: {df_top_items['store'].nunique()}")
#     logger.info(f"Number of unique items: {df_top_items['item'].nunique()}")
#     logger.info(f"Shape of the dataset: {df.shape}")
#     if fn:
#         logger.info(f"Saving final_df to {fn}")
#         df.to_csv(fn, index=False)
#     return df
