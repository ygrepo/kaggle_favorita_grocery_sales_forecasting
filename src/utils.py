from __future__ import annotations

import heapq
import numpy as np
import pandas as pd
from scipy.linalg import dft
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import umap
from typing import List, Optional, Generator, Union
import logging
from tqdm import tqdm
from pathlib import Path
import gc


logging.basicConfig(
    level=logging.INFO,  # or DEBUG
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CYCLICAL_FEATURES = [
    "dayofweek",
    "weekofmonth",
    "monthofyear",
    "paycycle",
    "season",
]
TRIGS = ["sin", "cos"]


def build_feature_and_label_cols(window_size: int) -> tuple[list[str], list[str]]:
    """Return feature and label column names for a given window size."""
    meta_cols = [
        "start_date",
        "id",
        "store_item",
        "store_cluster",
        "item_cluster",
    ]
    x_cyclical_features = [
        f"{feat}_{trig}_{i}"
        for feat in CYCLICAL_FEATURES
        for trig in TRIGS
        for i in range(1, window_size + 1)
    ]

    x_sales_features = [
        f"{name}_{i}"
        for name in [
            "sales_day",
            "store_med_day",
            "item_med_day",
            "store_cluster_logpct_change",
            "item_cluster_logpct_change",
        ]
        for i in range(1, window_size + 1)
    ]

    x_feature_cols = x_sales_features + x_cyclical_features
    label_cols = [f"y_{c}" for c in x_feature_cols]
    y_sales_features = [f"y_{c}" for c in x_sales_features]
    y_cyclical_features = [f"y_{c}" for c in x_cyclical_features]
    return (
        meta_cols,
        x_sales_features,
        x_cyclical_features,
        x_feature_cols,
        label_cols,
        y_sales_features,
        y_cyclical_features,
    )


def generate_aligned_windows(
    df: pd.DataFrame,
    window_size: int,
    *,
    calendar_aligned: bool = False,
) -> list[list[pd.Timestamp]]:
    """
    Build a series of non‑overlapping, length‑`window_size` date windows.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``"date"`` column convertible to ``pd.Timestamp``.
    window_size : int
        Number of days in each window.
    calendar_aligned : bool, default False
        • **False**  →  *Data‑aligned* (original behaviour):
          Only the distinct dates that actually occur in *df* are used.
          If the number of dates is not a multiple of `window_size`,
          the **last window may be shorter** than `window_size`.
        • **True**   →  *Calendar‑aligned* (back‑to‑front behaviour):
          Anchors on the most recent calendar day in *df* and steps
          backward in *exact* `window_size`‑day blocks, discarding any
          leftover days that cannot form a full block.
          Returned windows are always exactly `window_size` long and
          include **all calendar days**, even ones missing from *df*.

    Returns
    -------
    List[List[pd.Timestamp]]
        A list of date‑lists (each list length == `window_size`
        unless `calendar_aligned=False` and the final window is partial).
    """
    if window_size <= 0:
        raise ValueError("`window_size` must be a positive integer.")

    df["date"] = pd.to_datetime(df["date"])

    if calendar_aligned:
        # --- back‑to‑front, gap‑free windows ---
        last_date = df["date"].max().normalize()  # ensure time == 00:00
        first_date = df["date"].min().normalize()

        starts = []
        current = last_date
        while current >= first_date + pd.Timedelta(days=window_size - 1):
            logger.debug(f"Current date: {current}")
            starts.append(current - pd.Timedelta(days=window_size - 1))
            current -= pd.Timedelta(days=window_size)

        # Reverse so windows are chronological (oldest→newest)
        return [
            list(pd.date_range(start, periods=window_size, freq="D"))
            for start in reversed(starts)
        ]

    else:
        # --- forward, data‑aligned windows (may be shorter at the tail) ---
        unique_dates = sorted(pd.to_datetime(df["date"].unique()))
        logger.debug(f"Unique dates: {unique_dates}")
        return [
            unique_dates[i : i + window_size]
            for i in range(0, len(unique_dates), window_size)
        ]


def generate_cyclical_features(
    df: pd.DataFrame,
    window_size: int = 16,
    *,
    calendar_aligned: bool = True,
    log_level: str = "INFO",
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate day‑/week‑/month/season/pay‑cycle sine & cosine features
    for non‑overlapping windows and *include* store/item cluster IDs.

    If `cluster_df` is None, every store (or item)
    is assigned to the single cluster label 'ALL_STORES' / 'ALL_ITEMS'.
    """
    # Drop duplicates to get one id per store_item
    id_mapping = df[["store_item", "id"]].drop_duplicates()

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    df["date"] = pd.to_datetime(df["date"])
    cols = [
        "start_date",
        "store_item",
        "store",
        "item",
    ]
    for i in range(1, window_size + 1):
        for feature in [
            "dayofweek",
            "weekofmonth",
            "monthofyear",
            "paycycle",
            "season",
        ]:
            for trig in ["sin", "cos"]:
                cols.append(f"{feature}_{trig}_{i}")

    if df.empty:
        return pd.DataFrame(columns=cols)

    # ───────────────── raw cyclical columns ─────────────────
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    df["weekofmonth"] = df["date"].apply(lambda d: (d.day - 1) // 7 + 1)
    df["weekofmonth_sin"] = np.sin(2 * np.pi * df["weekofmonth"] / 5)
    df["weekofmonth_cos"] = np.cos(2 * np.pi * df["weekofmonth"] / 5)

    df["monthofyear"] = df["date"].dt.month
    df["monthofyear_sin"] = np.sin(2 * np.pi * df["monthofyear"] / 12)
    df["monthofyear_cos"] = np.cos(2 * np.pi * df["monthofyear"] / 12)

    # Pay‑cycle helpers
    def _pay_cycle_ratio(d: pd.Timestamp) -> float:
        month_end_day = (d + pd.offsets.MonthEnd(0)).day
        if d.day >= 15:
            last_pay = d.replace(day=15)
            next_pay = d.replace(day=month_end_day)
        else:
            prev_month_end = d - pd.offsets.MonthEnd(1)
            last_pay = prev_month_end
            next_pay = d.replace(day=15)
        cycle_len = (next_pay - last_pay).days
        return ((d - last_pay).days / cycle_len) if cycle_len else 0.0

    df["paycycle_ratio"] = df["date"].apply(_pay_cycle_ratio)
    df["paycycle_sin"] = np.sin(2 * np.pi * df["paycycle_ratio"])
    df["paycycle_cos"] = np.cos(2 * np.pi * df["paycycle_ratio"])

    # Season helpers
    def _season_ratio(d: pd.Timestamp) -> float:
        base = pd.Timestamp(year=d.year, month=3, day=20)
        if d < base:
            base = pd.Timestamp(year=d.year - 1, month=3, day=20)
        return ((d - base).days % 365) / 365

    df["season_ratio"] = df["date"].apply(_season_ratio)
    df["season_sin"] = np.sin(2 * np.pi * df["season_ratio"])
    df["season_cos"] = np.cos(2 * np.pi * df["season_ratio"])

    # ───────────────── build window rows ─────────────────
    results: List[dict] = []
    iterator = df.groupby("store_item")
    if logger.level == logging.DEBUG:
        iterator = tqdm(iterator, desc="Generating cyclical features")

    for store_item, group in iterator:
        # logger.debug(f"Cyclical features: Processing {store_item}")
        group = group.sort_values("date").reset_index(drop=True)
        windows = generate_aligned_windows(
            group, window_size, calendar_aligned=calendar_aligned
        )

        for i, window_dates in enumerate(windows):
            window_df = group[group["date"].isin(window_dates)]
            row = {
                "start_date": window_dates[0],
                "store_item": store_item,
                "store": group["store"].iloc[0],
                "item": group["item"].iloc[0],
            }

            if window_df.empty:
                logger.warning(f"Empty window for {store_item} at {window_dates[0]}")
                continue

            else:
                for i in range(window_size):
                    if i < len(window_df):
                        r = window_df.iloc[i]
                        for f in CYCLICAL_FEATURES:
                            for t in TRIGS:
                                row[f"{f}_{t}_{i+1}"] = r[f"{f}_{t}"]
                    else:
                        for f in CYCLICAL_FEATURES:
                            for t in TRIGS:
                                row[f"{f}_{t}_{i+1}"] = 0.0
            results.append(row)

    df = pd.DataFrame(results, columns=cols)
    df = df.merge(id_mapping, on=["store_item"], how="left")
    cols.insert(cols.index("start_date") + 1, "id")
    df = df[cols]
    if output_path is not None:
        logger.info(f"Saving cyclical features to {output_path}")
        df.to_csv(output_path, index=False)
    return df


def generate_sales_features(
    df: pd.DataFrame,
    window_size: int = 5,
    *,
    calendar_aligned: bool = True,
    log_level: str = "INFO",
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate rolling window sales features for pre-aggregated store-item-date sales rows.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: ['date', 'store', 'item', 'unit_sales'].
        Each row corresponds to a unique (store, item, date).
    window_size : int
        Number of days in each rolling window.
    calendar_aligned : bool
        Use calendar-based windows if True.
    log_level : str
        Logging level.
    output_path : Optional[Path]
        Path to save output to.

    Returns
    -------
    pd.DataFrame
        Long-form table with features for each (store, item) across windows.
    """

    logger.info(f"Log level: {log_level}")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.info(f"Log level: {logger.level}")
    
    # Drop duplicates to get one id per store_item
    id_mapping = df[["store_item", "id"]].drop_duplicates()

    df["date"] = pd.to_datetime(df["date"])

    # Lookup dictionaries
    logger.debug("Generating lookup dictionaries")
    store_to_cluster = (
        df.drop_duplicates("store")[["store", "store_cluster"]]
        .set_index("store")["store_cluster"]
        .to_dict()
    )
    logger.debug(f"Store to cluster: {len(store_to_cluster)}")
    store_item_to_item_cluster = (
        df.drop_duplicates(["store", "item"])[["store", "item", "item_cluster"]]
        .set_index(["store", "item"])["item_cluster"]
        .to_dict()
    )
    logger.debug(f"Store item to item cluster: {len(store_item_to_item_cluster)}")

    logger.debug(f"Generating windows")
    windows = generate_aligned_windows(
        df, window_size, calendar_aligned=calendar_aligned
    )
    records: List[dict] = []

    prev_store_med = None
    prev_item_med = None
    for window_dates in windows:
        logger.debug(f"Window dates: {window_dates}")
        w_df = df[df["date"].isin(window_dates)]

        # cluster-level medians
        store_med = (
            w_df.groupby(["store_cluster", "date"])["unit_sales"]
            .median()
            .unstack(fill_value=0)
        )
        item_med = (
            w_df.groupby(["item_cluster", "date"])["unit_sales"]
            .median()
            .unstack(fill_value=0)
        )
        store_median_curr = store_med.median(axis=1)
        item_median_curr = item_med.median(axis=1)
        if prev_store_med is not None:
            store_median_change = (
                np.log1p(np.abs(store_median_curr - prev_store_med))
                / prev_store_med.replace(0, np.nan)
                * 100
            )
        else:
            store_median_change = pd.Series(np.nan, index=store_median_curr.index)

        if prev_item_med is not None:
            item_median_change = (
                np.log1p(np.abs(item_median_curr - prev_item_med))
                / prev_item_med.replace(0, np.nan)
                * 100
            )
        else:
            item_median_change = pd.Series(np.nan, index=item_median_curr.index)

        prev_store_med = store_median_curr.copy()
        prev_item_med = item_median_curr.copy()

        # no groupby needed for store-item, just pivot
        sales = w_df.pivot(
            index=["store", "item"], columns="date", values="unit_sales"
        ).fillna(0)

        iterator = sales.iterrows()
        if logger.level == logging.DEBUG:
            iterator = tqdm(
                iterator,
                total=sales.shape[0],
                desc=f"Window {window_dates[0].strftime('%Y-%m-%d')}",
            )

        for (store, item), sales_vals in iterator:
            # logger.debug(f"Sales features: Processing {store}_{item}")
            s_cl = store_to_cluster.get(store, "ALL_STORES")
            i_cl = store_item_to_item_cluster.get((store, item), "ALL_ITEMS")

            row = {
                "store_item": f"{store}_{item}",
                "store": store,
                "item": item,
                "store_cluster": s_cl,
                "item_cluster": i_cl,
                "start_date": window_dates[0],
            }

            for i in range(1, window_size + 1):
                d = window_dates[i - 1] if i - 1 < len(window_dates) else None

                if d is not None:
                    row[f"sales_day_{i}"] = sales_vals.get(d, 0)
                    row[f"store_med_day_{i}"] = (
                        store_med.loc[s_cl].get(d, 0)
                        if s_cl in store_med.index
                        else np.nan
                    )
                    row[f"store_cluster_logpct_change_{i}"] = (
                        store_median_change.get(s_cl, np.nan)
                        if s_cl in store_median_change.index
                        else np.nan
                    )
                    row[f"item_med_day_{i}"] = (
                        item_med.loc[i_cl].get(d, 0)
                        if i_cl in item_med.index
                        else np.nan
                    )
                    row[f"item_cluster_logpct_change_{i}"] = (
                        item_median_change.get(i_cl, np.nan)
                        if i_cl in item_median_change.index
                        else np.nan
                    )
                else:
                    continue

            records.append(row)

    # ------------------------------------------------------------------
    # Final column order
    # ------------------------------------------------------------------

    cols = [
        "start_date",
        "store_item",
        "store",
        "item",
        "store_cluster",
        "item_cluster",
    ]
    cols += [
        f"{prefix}{i}"
        for prefix in (
            "sales_day_",
            "store_med_day_",
            "item_med_day_",
            "store_cluster_logpct_change_",
            "item_cluster_logpct_change_",
        )
        for i in range(1, window_size + 1)
    ]
    df = pd.DataFrame(records, columns=cols) if records else pd.DataFrame(cols)
    df = df.merge(id_mapping, on=["store_item"], how="left")
    cols.insert(cols.index("start_date") + 1, "id")
    df = df[cols]
    if output_path is not None:
        df.to_csv(output_path, index=False)
    return df


def add_y_targets_from_shift(df, window_size=16):
    df = df.sort_values(["store_item", "start_date"]).reset_index(drop=True)
    result = []

    for _, group in df.groupby("store_item"):
        group = group.sort_values("start_date").reset_index(drop=True)

        for i in range(len(group) - 1):
            cur_row = group.iloc[i]
            next_row = group.iloc[i + 1]

            # Check if next start_date is exactly +16 days
            if (next_row["start_date"] - cur_row["start_date"]).days != window_size:
                continue

            row = cur_row.copy()
            for col in group.columns:
                if any(
                    col.startswith(prefix)
                    for prefix in [
                        "sales_day_",
                        "store_med_day_",
                        "item_med_day_",
                        "store_cluster_logpct_change",
                        "item_cluster_logpct_change",
                        "dayofweek_",
                        "weekofmonth_",
                        "monthofyear_",
                        "paycycle_",
                        "season_",
                    ]
                ):
                    row[f"y_{col}"] = next_row[col]

            result.append(row)

    return pd.DataFrame(result)


# def generate_sales_features(
#     df: pd.DataFrame,
#     window_size: int = 5,
#     *,
#     #cluster_df: Optional[pd.DataFrame] = None,
#     calendar_aligned: bool = True,
#     debug: bool = False,
#     debug_fn: Optional[Path] = None,
#     log_level: str = "INFO",
#     output_path: Optional[Path] = None,
# ) -> pd.DataFrame:
#     """
#     Generate rolling window sales features for pre-aggregated store-item-date sales rows.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Must have columns: ['date', 'store', 'item', 'unit_sales'].
#         Each row corresponds to a unique (store, item, date).
#     window_size : int
#         Number of days in each rolling window.
#     cluster_df : pd.DataFrame, optional
#         Must have columns: ['store', 'item', 'store_cluster', 'item_cluster'].
#     calendar_aligned : bool
#         Use calendar-based windows if True.
#     debug : bool
#         Save missing cluster mappings to 'missing_cluster_pairs.csv' if True.

#     Returns
#     -------
#     pd.DataFrame
#         Long-form table with features for each (store, item) across windows.
#     """

#     if debug:
#         logger.setLevel(logging.DEBUG)
#     else:
#         logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

#     # Drop duplicates to get one id per store_item
#     id_mapping = df[["store_item", "id"]].drop_duplicates()

#     df = df.copy()
#     df["date"] = pd.to_datetime(df["date"])

#     # ------------------------------------------------------------------
#     # Store + Item cluster attachment
#     # ------------------------------------------------------------------
#     if cluster_df is not None:
#         mapping = cluster_df[["store", "item", "store_cluster", "item_cluster"]].copy()
#         df = df.merge(mapping, on=["store", "item"], how="left")

#         # Check for missing mappings
#         missing = df[df["store_cluster"].isna() | df["item_cluster"].isna()][
#             ["store", "item"]
#         ]
#         if not missing.empty:
#             if debug:
#                 missing.drop_duplicates().to_csv(
#                     debug_fn or "missing_cluster_pairs.csv", index=False
#                 )
#                 logger.warning(
#                     f"[DEBUG] {len(missing)} (store, item) pairs missing cluster mapping. "
#                     f"Saved to 'missing_cluster_pairs.csv'."
#                 )
#             else:
#                 raise ValueError(
#                     f"{len(missing)} (store, item) pairs missing cluster assignments.\n"
#                     f"Use debug=True to write details to CSV."
#                 )

#         df["storeClusterId"] = df["store_cluster"]
#         df["itemClusterId"] = df["item_cluster"]
#         df.drop(columns=["store_cluster", "item_cluster"], inplace=True)
#     else:
#         df["storeClusterId"] = "ALL_STORES"
#         df["itemClusterId"] = "ALL_ITEMS"

#     df["storeClusterId"] = df["storeClusterId"].fillna("ALL_STORES")
#     df["itemClusterId"] = df["itemClusterId"].fillna("ALL_ITEMS")

#     # Lookup dictionaries
#     store_to_cluster = (
#         df.drop_duplicates("store")[["store", "storeClusterId"]]
#         .set_index("store")["storeClusterId"]
#         .to_dict()
#     )
#     store_item_to_item_cluster = (
#         df.drop_duplicates(["store", "item"])[["store", "item", "itemClusterId"]]
#         .set_index(["store", "item"])["itemClusterId"]
#         .to_dict()
#     )

#     windows = generate_aligned_windows(
#         df, window_size, calendar_aligned=calendar_aligned
#     )
#     records: List[dict] = []

#     for window_dates in windows:
#         w_df = df[df["date"].isin(window_dates)]

#         # cluster-level medians
#         store_med = (
#             w_df.groupby(["storeClusterId", "date"])["unit_sales"]
#             .median()
#             .unstack(fill_value=0)
#         )
#         item_med = (
#             w_df.groupby(["itemClusterId", "date"])["unit_sales"]
#             .median()
#             .unstack(fill_value=0)
#         )

#         # no groupby needed for store-item, just pivot
#         sales = w_df.pivot(
#             index=["store", "item"], columns="date", values="unit_sales"
#         ).fillna(0)

#         iterator = sales.iterrows()
#         if debug:
#             iterator = tqdm(
#                 iterator,
#                 total=sales.shape[0],
#                 desc=f"Window {window_dates[0].strftime('%Y-%m-%d')}",
#             )

#         for (store, item), sales_vals in iterator:
#             # logger.debug(f"Sales features: Processing {store}_{item}")
#             s_cl = store_to_cluster.get(store, "ALL_STORES")
#             i_cl = store_item_to_item_cluster.get((store, item), "ALL_ITEMS")

#             row = {
#                 "store_item": f"{store}_{item}",
#                 "store": store,
#                 "item": item,
#                 "storeClusterId": s_cl,
#                 "itemClusterId": i_cl,
#                 "start_date": window_dates[0],
#             }

#             for i in range(1, window_size + 1):
#                 d = window_dates[i - 1] if i - 1 < len(window_dates) else None

#                 if d is not None:
#                     row[f"sales_day_{i}"] = sales_vals.get(d, 0)
#                     row[f"store_med_day_{i}"] = (
#                         store_med.loc[s_cl].get(d, 0)
#                         if s_cl in store_med.index
#                         else np.nan
#                     )
#                     row[f"item_med_day_{i}"] = (
#                         item_med.loc[i_cl].get(d, 0)
#                         if i_cl in item_med.index
#                         else np.nan
#                     )
#                 else:
#                     continue

#             records.append(row)

#     # ------------------------------------------------------------------
#     # Final column order
#     # ------------------------------------------------------------------

#     cols = [
#         "start_date",
#         "store_item",
#         "store",
#         "item",
#         "storeClusterId",
#         "itemClusterId",
#     ]
#     cols += [
#         f"{prefix}{i}"
#         for prefix in ("sales_day_", "store_med_day_", "item_med_day_")
#         for i in range(1, window_size + 1)
#     ]
#     df = pd.DataFrame(records, columns=cols) if records else pd.DataFrame(cols)
#     df = df.merge(id_mapping, on=["store_item"], how="left")
#     cols.insert(cols.index("start_date") + 1, "id")
#     df = df[cols]
#     if output_path is not None:
#         df.to_csv(output_path, index=False)
#     return df


# def add_y_targets_from_shift(df, window_size=16):
#     df = df.sort_values(["store_item", "start_date"]).reset_index(drop=True)
#     result = []

#     for _, group in df.groupby("store_item"):
#         group = group.sort_values("start_date").reset_index(drop=True)

#         for i in range(len(group) - 1):
#             cur_row = group.iloc[i]
#             next_row = group.iloc[i + 1]

#             # Check if next start_date is exactly +16 days
#             if (next_row["start_date"] - cur_row["start_date"]).days != window_size:
#                 continue

#             row = cur_row.copy()
#             for col in group.columns:
#                 if any(
#                     col.startswith(prefix)
#                     for prefix in [
#                         "sales_day_",
#                         "store_med_day_",
#                         "item_med_day_",
#                         "dayofweek_",
#                         "weekofmonth_",
#                         "monthofyear_",
#                         "paycycle_",
#                         "season_",
#                     ]
#                 ):
#                     row[f"y_{col}"] = next_row[col]

#             result.append(row)

#     return pd.DataFrame(result)


def add_next_window_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Attach next window's features with a y_ prefix without dropping rows."""
    df = df.sort_values(["store_item", "start_date"]).reset_index(drop=True)
    prefixes = [
        "sales_day_",
        "store_med_day_",
        "item_med_day_",
        "dayofweek_",
        "weekofmonth_",
        "monthofyear_",
        "paycycle_",
        "season_",
    ]

    cols_to_shift = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
    result = df.copy()

    for _, group in df.groupby("store_item"):
        shifted = group[cols_to_shift].shift(-1)
        result.loc[group.index, [f"y_{c}" for c in cols_to_shift]] = shifted.values

    return result


def prepare_training_data_from_raw_df(
    df,
    *,
    window_size=16,
    calendar_aligned: bool = True,
    sales_fn: Optional[Path] = None,
    cyc_fn: Optional[Path] = None,
    log_level: str = "INFO",
) -> pd.DataFrame:
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if "store_item" not in df.columns:
        df["store_item"] = df["store"].astype(str) + "_" + df["item"].astype(str)

    if sales_fn is not None:
        if sales_fn.exists():
            logger.info(f"Loading sales features from {sales_fn}")
            sales_df = pd.read_csv(sales_fn)
        else:
            logger.info(f"Generating sales features to {sales_fn}")
            sales_df = generate_sales_features(
                df,
                window_size,
                calendar_aligned=calendar_aligned,
                log_level=log_level,
                output_path=sales_fn,
            )

    if cyc_fn is not None:
        if cyc_fn.exists():
            logger.info(f"Loading cyclical features from {cyc_fn}")
            cyc_df = pd.read_csv(cyc_fn)
        else:
            logger.info(f"Generating cyclical features to {cyc_fn}")
            cyc_df = generate_cyclical_features(
                df,
                window_size,
                calendar_aligned=calendar_aligned,
                log_level=log_level,
                output_path=cyc_fn,
            )

    sales_df["start_date"] = pd.to_datetime(sales_df["start_date"])
    cyc_df["start_date"] = pd.to_datetime(cyc_df["start_date"])

    logger.info(f"Merging sales and cyclical features")
    merged_df = pd.merge(
        sales_df,
        cyc_df,
        on=["start_date", "id", "store_item", "store", "item"],
    )

    logger.info(f"merged_df.shape: {merged_df.shape}")
    merged_df = add_y_targets_from_shift(merged_df, window_size)
    logger.info(f"merged_df.shape: {merged_df.shape}")

    return merged_df


# Redefine RunningMedian class
class RunningMedian:
    def __init__(self):
        self.low = []  # max-heap
        self.high = []  # min-heap

    def add(self, num):
        heapq.heappush(self.low, -heapq.heappushpop(self.high, num))
        if len(self.low) > len(self.high):
            heapq.heappush(self.high, -heapq.heappop(self.low))

    def median(self) -> float:
        if self.low or self.high:
            if len(self.high) > len(self.low):
                return float(self.high[0])
            return (self.high[0] - self.low[0]) / 2.0
        return 0.0


class SmoothOnlineMedian:
    def __init__(self, s: float, eta: float, debug: bool = False):
        self.estimate = s
        self.eta = eta
        self.debug = debug
        if self.debug:
            self.history = []

    def update(self, x: float) -> float:
        delta = self.eta * self._sgn(x - self.estimate)
        self.estimate += delta
        if self.debug:
            self.history.append(self.estimate)
        return self.estimate

    def _sgn(self, value: float) -> int:
        return 1 if value > 0 else -1 if value < 0 else 0

    def get_estimate(self) -> float:
        return self.estimate

    def get_history(self):
        if self.debug:
            return self.history
        else:
            raise AttributeError(
                "History tracking is disabled. Enable debug mode to track history."
            )


def preprocess_sales_matrix(
    df: pd.DataFrame, log_transform=True, smooth_window=7, zscore_rows=True
):
    """
    Preprocesses a pivoted sales matrix for GDKM:
    1. Optionally applies log1p transform
    2. Applies rolling mean smoothing
    3. Drops rows with zero variance
    4. Z-score normalization per row (optional)

    Parameters:
    - pivot_df: DataFrame of shape (store_item, date)
    - log_transform: whether to apply log1p to unit_sales
    - smooth_window: window size for rolling mean smoothing
    - zscore_rows: whether to z-score normalize each row

    Returns:
    - X: np.ndarray, processed matrix
    - pivot_df_filtered: filtered version of the input DataFrame
    """
    df = df.copy()

    # Apply rolling mean to smooth spikes
    df = df.rolling(window=smooth_window, axis=1, min_periods=1).mean()

    # Drop rows with no variance (flat after smoothing)
    df = df[df.var(axis=1) > 0]

    # Apply log(1 + x) to reduce spike influence
    if log_transform:
        df = np.log1p(df)

    # Normalize each row and columns (Z-score)
    if zscore_rows:
        X = MinMaxScaler().fit_transform(df)
        # X = MinMaxScaler().fit_transform(X.T).T
    else:
        X = df.values

    return X, df


def zscore_with_axis(
    df: pd.DataFrame, axis: int = 0, nan_policy: str = "omit"
) -> pd.DataFrame:
    """
    Compute z-score along rows (axis=1) or columns (axis=0), optionally ignoring NaNs.
    """
    values = df.values.astype(float)

    if nan_policy == "omit":
        mean = np.nanmean(values, axis=axis, keepdims=True)
        std = np.nanstd(values, axis=axis, ddof=0, keepdims=True)
    elif nan_policy == "propagate":
        mean = np.mean(values, axis=axis, keepdims=True)
        std = np.std(values, axis=axis, ddof=0, keepdims=True)
    else:
        raise ValueError("nan_policy must be 'omit' or 'propagate'")

    z = (values - mean) / std
    return pd.DataFrame(z, index=df.index, columns=df.columns)


def normalize_store_item_matrix(
    df: pd.DataFrame,
    freq="W",
    median_transform=True,
    mean_transform=False,
    log_transform=True,
    zscore_rows=True,
    zscore_cols=False,  # safer default to avoid double scaling
):
    """
    Builds a store × item matrix with weekly sales, optionally normalized.

    Parameters
    ----------
    df : pd.DataFrame
        Must include ['date', 'store', 'item', 'unit_sales']
    freq : str
        Resample frequency (e.g., 'W' for weekly)
    median_transform : bool
        Use median aggregation
    mean_transform : bool
        Use mean aggregation
    log_transform : bool
        Apply log1p to unit_sales
    zscore_rows : bool
        Z-score each row (store)
    zscore_cols : bool
        Z-score each column (item)

    Returns
    -------
    pd.DataFrame
        A normalized store × item matrix
    """
    if median_transform:
        df = (
            df.groupby([pd.Grouper(key="date", freq=freq), "store", "item"])[
                "unit_sales"
            ]
            .median()
            .reset_index()
        )
        df = df.groupby(["store", "item"])["unit_sales"].median().unstack(fill_value=0)
    elif mean_transform:
        df = (
            df.groupby([pd.Grouper(key="date", freq=freq), "store", "item"])[
                "unit_sales"
            ]
            .mean()
            .reset_index()
        )
        df = df.groupby(["store", "item"])["unit_sales"].mean().unstack(fill_value=0)
    else:
        raise ValueError("Set either median_transform or mean_transform to True.")

    if log_transform:
        df = np.log1p(df)

    if zscore_rows:
        df = zscore_with_axis(df, axis=1)

    if zscore_cols:
        df = zscore_with_axis(df, axis=0)

    return df


def generate_store_item_clusters(
    df: pd.DataFrame,
    n_clusters: int,
    model_class: type,
) -> pd.DataFrame:
    """Cluster store/item series and return labels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where each row corresponds to a ``store_item`` and
        columns are the time series values.
    n_clusters : int
        Desired number of clusters to fit.
    model_class : type
        Scikit‑learn clustering estimator class.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``store_item`` and ``clusterId`` columns.
    """

    model = model_class(n_clusters=n_clusters)
    labels = model.fit_predict(df.values)

    return pd.DataFrame(
        {
            "store_item": df.index,
            "clusterId": labels,
        }
    )


def reorder_data(data, row_labels, col_labels):
    row_order = np.argsort(row_labels)
    col_order = np.argsort(col_labels)
    return data[np.ix_(row_order, col_order)], row_order, col_order
