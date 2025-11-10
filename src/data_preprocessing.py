import pandas as pd
import numpy as np
from src.utils import get_logger

logger = get_logger(__name__)


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
    grouped = (
        df.groupby(group_column)[value_column].value_counts(normalize=True)
        * 100
    )
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
        {
            column_name + "_count": counts,
            column_name + "_percentage": percentages,
        }
    )
    return df.sort_values(column_name, ascending=False).head(top_n)


def count_percent(series, n=3):
    counts = series.value_counts().head(n)
    percentages = counts / series.count() * 100
    result = pd.DataFrame({"Count": counts, "Percentage": percentages})
    return result


def select_extreme_and_median_neighbors(
    df: pd.DataFrame,
    n_col: str = "unit_sales",
    group_column: str = "store",
    M: int = 0,
    m: int = 0,
    med: int = 0,
) -> np.ndarray:
    """
    Returns M highest, m lowest, and 2*med around the median total sales groups.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        n_col (str): Column to sum (e.g., 'unit_sales').
        group_column (str): Column to group by (e.g., 'store_nbr').
        M (int): Number of top (max) groups to return.
        m (int): Number of bottom (min) groups to return.
        med (int): Number of groups to return on each side of the median
                     (total of 2*med closest groups will be selected).

    Returns:
        np.ndarray: Array of selected group identifiers.
    """
    # logger.info(f"Processing {df[group_column].nunique()} unique groups")

    grouped = (
        df.groupby(group_column)
        .agg({n_col: "sum"})
        .rename(columns={n_col: "total"})
        .reset_index()
    )
    sorted_grouped = grouped.sort_values("total")
    # logger.info(f"Found {len(grouped)} unique groups")

    if M == 0 and m == 0 and med == 0:
        # logger.info("No selection criteria specified, returning all groups")
        idxs = sorted_grouped[group_column].values
        # np.unique is not needed here, 'values' is already unique
        # logger.info(f"Selected {len(idxs)} groups")
        return idxs

    # Get extremes
    if m > 0:
        # logger.info(f"Selecting {m} bottom groups")
        bottom_m = sorted_grouped.head(m)
        bottom_records = bottom_m[group_column].values
        # logger.info(f"Selected {len(bottom_records)} bottom groups")
    else:
        bottom_records = np.array([])

    if M > 0:
        # logger.info(f"Selecting {M} top groups")
        top_M = sorted_grouped.tail(M)
        top_records = top_M[group_column].values
        # logger.info(f"Selected {len(top_records)} top groups")
    else:
        top_records = np.array([])

    # Find median-centered groups
    if med > 0:
        # logger.info(f"Selecting {2*med} median groups")
        median_val = grouped["total"].median()

        grouped_with_dist = grouped.copy()
        grouped_with_dist["dist_to_median"] = np.abs(
            grouped_with_dist["total"] - median_val
        )

        # Get groups to exclude (already selected as extremes)
        exclude_groups = np.concatenate([bottom_records, top_records])

        # Filter out already selected groups
        available_for_median = grouped_with_dist[
            ~grouped_with_dist[group_column].isin(exclude_groups)
        ]

        # **REFINEMENT 1: Use nsmallest for better performance**
        median_neighbors = available_for_median.nsmallest(
            2 * med, "dist_to_median"
        )

        # **REFINEMENT 2: Remove redundant np.unique**
        med_records = median_neighbors[group_column].values
        # logger.info(f"Selected {len(med_records)} median groups")
    else:
        med_records = np.array([])

    all_records = np.concatenate(
        [top_records, med_records, bottom_records], axis=0
    )
    # Final unique call is still good defensive programming
    all_records = np.unique(all_records)
    # logger.info(f"Total unique groups: {len(all_records)}")
    return all_records
