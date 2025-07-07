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
    Prepares the data by computing value counts and percentages for each group.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        group_column (str): Column to group by.
        value_column (str): Column to calculate percentages from.
        top_stores_n (int): Number of top stores to return.
        top_items_n (int): Number of top items to return.

    Returns:
        pd.DataFrame: DataFrame with counts and percentages for each group.
    """
    df = df.copy()
    df = top_n_by_m(
        df, n_col=value_column, group_column=group_column, top_n=top_stores_n
    )
    valid_stores = df.reset_index()[group_column].tolist()
    df = df[df[group_column].isin(valid_stores)]
    df = df.reset_index()
    df.drop(["index"], axis=1, inplace=True)
    valid_item = count_percent(df["item"], n=top_items_n).reset_index()["item"].tolist()
    df = df[df["item"].isin(valid_item)]
    logger.info(f"Number of rows: {len(df)}")
    logger.info(f"Number of unique stores: {df['store'].nunique()}")
    logger.info(f"Number of unique items: {df['item'].nunique()}")
    logger.info(f"Shape of the dataset: {df.shape}")
    if fn:
        df.to_csv(fn, index=False)
    return df
