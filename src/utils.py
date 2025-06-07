import heapq
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import (
    SpectralBiclustering,
    SpectralClustering,
    SpectralCoclustering,
)


from typing import Union, Dict


def build_feature_and_label_cols(window_size: int) -> tuple[list[str], list[str]]:
    """Return feature and label column names for a given window size."""
    cyclical_features = [
        f"{feat}_{trig}_{i}"
        for feat in ["dayofweek", "weekofmonth", "monthofyear", "paycycle", "season"]
        for trig in ["sin", "cos"]
        for i in range(1, window_size + 1)
    ]

    sales_features = [
        f"{name}_{i}"
        for name in ["sales_day", "store_med_day", "item_med_day"]
        for i in range(1, window_size + 1)
    ]

    feature_cols = sales_features + cyclical_features
    label_cols = [f"y_{c}" for c in feature_cols]
    meta_cols = ["start_date", "store_item", "store", "item"]
    y_sales_features = [f"y_{c}" for c in sales_features]
    y_cyclical_features = [f"y_{c}" for c in cyclical_features]
    return meta_cols, feature_cols, label_cols, y_sales_features, y_cyclical_features


def generate_aligned_windows(df, window_size):
    """
    Returns a list of exact non-overlapping date-aligned windows ending at the last date in the data,
    stepping backwards from the latest date.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    last_date = df["date"].max()
    first_date = df["date"].min()

    start_dates = []
    current = last_date
    while current >= first_date + pd.Timedelta(days=window_size - 1):
        start_dates.append(current - pd.Timedelta(days=window_size - 1))
        current -= pd.Timedelta(days=window_size)

    return [
        pd.date_range(start, periods=window_size).to_pydatetime()
        for start in reversed(start_dates)
    ]


def _generate_sliding_windows(dates: pd.Series, window_size: int):
    """Return list of date ranges for standard sliding windows."""
    dates = pd.to_datetime(dates).sort_values().reset_index(drop=True)
    return [
        pd.date_range(dates[i], periods=window_size).to_pydatetime()
        for i in range(len(dates) - window_size + 1)
    ]


def generate_cyclical_features(df: pd.DataFrame, window_size: int = 7) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Ensure store_item column exists if needed
    if "store_item" not in df.columns:
        df["store_item"] = df["store"].astype(str) + "_" + df["item"].astype(str)

    # Prepare full column structure regardless of input size
    cols = ["start_date", "store_item", "store", "item"]
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

    # Add cyclical features
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["weekofmonth"] = df["date"].apply(lambda d: (d.day - 1) // 7 + 1)
    df["weekofmonth_sin"] = np.sin(2 * np.pi * df["weekofmonth"] / 5)
    df["weekofmonth_cos"] = np.cos(2 * np.pi * df["weekofmonth"] / 5)
    df["monthofyear"] = df["date"].dt.month
    df["monthofyear_sin"] = np.sin(2 * np.pi * df["monthofyear"] / 12)
    df["monthofyear_cos"] = np.cos(2 * np.pi * df["monthofyear"] / 12)

    # --- Pay cycle features ---
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

    def _season_ratio(d: pd.Timestamp) -> float:
        base = pd.Timestamp(year=d.year, month=3, day=20)
        if d < base:
            base = pd.Timestamp(year=d.year - 1, month=3, day=20)
        return ((d - base).days % 365) / 365

    df["season_ratio"] = df["date"].apply(_season_ratio)
    df["season_sin"] = np.sin(2 * np.pi * df["season_ratio"])
    df["season_cos"] = np.cos(2 * np.pi * df["season_ratio"])

    results = []

    for store_item, group in df.groupby("store_item"):
        group = group.sort_values("date").reset_index(drop=True)
        windows = _generate_sliding_windows(group["date"], window_size)

        for window_dates in windows:
            window_df = group[group["date"].isin(window_dates)]

            row = {
                "start_date": window_df["date"].min(),
                "store_item": store_item,
                "store": window_df["store"].iloc[0],
                "item": window_df["item"].iloc[0],
            }

            for i in range(window_size):
                if i < len(window_df):
                    r = window_df.iloc[i]
                    for f in [
                        "dayofweek",
                        "weekofmonth",
                        "monthofyear",
                        "paycycle",
                        "season",
                    ]:
                        for t in ["sin", "cos"]:
                            row[f"{f}_{t}_{i+1}"] = r[f"{f}_{t}"]
                else:
                    for f in [
                        "dayofweek",
                        "weekofmonth",
                        "monthofyear",
                        "paycycle",
                        "season",
                    ]:
                        for t in ["sin", "cos"]:
                            row[f"{f}_{t}_{i+1}"] = 0.0

            results.append(row)

    return pd.DataFrame(results, columns=cols)


# ------------------------------------------------------------------ #
# helper assumed to exist (unchanged)                                #
# ------------------------------------------------------------------ #
def generate_aligned_windows(
    df: pd.DataFrame, window_size: int
) -> list[list[pd.Timestamp]]:
    """
    Return a list of lists, each inner list holding exactly `window_size`
    consecutive dates covering the entire date span of `df`.
    """
    unique_dates = sorted(df["date"].unique())
    return [
        unique_dates[i : i + window_size]
        for i in range(0, len(unique_dates), window_size)
    ]


# ------------------------------------------------------------------ #
# main function                                                      #
# ------------------------------------------------------------------ #
def generate_sales_features(
    df: pd.DataFrame,
    window_size: int = 5,
    cluster_map: Union[pd.DataFrame, Dict[str, int], None] = None,
) -> pd.DataFrame:
    """
    Create rolling‑window sales features with optional cluster medians.
    """
    # --------------------------- housekeeping -------------------------- #
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")

    # ensure the composite key exists and is STRING everywhere
    if "store_item" not in df.columns:
        df["store_item"] = df["store"].astype(str) + "_" + df["item"].astype(str)
    df["store_item"] = df["store_item"].astype(str)

    # ------------- enrich df with cluster_id / store_cluster_id -------- #
    if cluster_map is not None:
        # normalise to DataFrame with a string store_item index
        if isinstance(cluster_map, dict):
            cluster_df = (
                pd.Series(cluster_map, name="cluster_id")
                .rename_axis("store_item")
                .reset_index()
            )
        else:  # DataFrame
            cluster_df = cluster_map.copy()
            # normalise common column names
            if "clusterId" in cluster_df.columns and "cluster_id" not in cluster_df.columns:
                cluster_df = cluster_df.rename(columns={"clusterId": "cluster_id"})

        cluster_df["store_item"] = cluster_df["store_item"].astype(str)

        # merge brings in any available columns; missing → NaN
        df = df.merge(cluster_df, on="store_item", how="left")

        # keep nullable integers (Int64) instead of float NaNs
        for col in ("cluster_id", "store_cluster_id", "item_cluster_id"):
            if col in df.columns:
                df[col] = df[col].astype("Int64")
            else:  # ensure column exists for downstream logic
                df[col] = pd.Series([pd.NA] * len(df), dtype="Int64")
    else:
        # create empty nullable‐int columns
        df["cluster_id"] = pd.Series([pd.NA] * len(df), dtype="Int64")
        df["store_cluster_id"] = pd.Series([pd.NA] * len(df), dtype="Int64")
        df["item_cluster_id"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    # ------------------------------------------------------------------ #
    # iterate over rolling windows                                       #
    # ------------------------------------------------------------------ #
    windows = generate_aligned_windows(df, window_size)
    records = []

    # pre‑declare all column names for final DataFrame
    base_cols = [
        "start_date",
        "store_item",
        "store",
        "item",
        "cluster_id",
        "store_cluster_id",
        "item_cluster_id",
    ]
    dyn_cols = sum(
        (
            [f"{prefix}_day_{i}" for i in range(1, window_size + 1)]
            for prefix in (
                "sales",
                "store_med",
                "item_med",
                "cluster_med",
                "store_cluster_med",
                "item_cluster_med",
            )
        ),
        [],
    )
    out_cols = base_cols + dyn_cols

    for window_dates in windows:
        w_df = df[df["date"].isin(window_dates)]

        # daily medians within this window
        store_med = (
            w_df.groupby(["store", "date"])["unit_sales"].median().unstack(fill_value=0)
        )
        item_med = (
            w_df.groupby(["item", "date"])["unit_sales"].median().unstack(fill_value=0)
        )
        cluster_med = (
            w_df.groupby(["cluster_id", "date"])["unit_sales"]
            .median()
            .unstack(fill_value=0)
        )
        sc_med = (
            w_df.groupby(["store_cluster_id", "date"])["unit_sales"]
            .median()
            .unstack(fill_value=0)
        )
        ic_med = (
            w_df.groupby(["item_cluster_id", "date"])["unit_sales"]
            .median()
            .unstack(fill_value=0)
        )

        # sales aggregated by (store,item,date)
        sales = (
            w_df.groupby(["store", "item", "date"])["unit_sales"]
            .sum()
            .unstack(fill_value=0)
        )

        for (store, item), sales_vals in sales.iterrows():
            sid = f"{store}_{item}"

            row = {
                "start_date": window_dates[0],
                "store_item": sid,
                "store": store,
                "item": item,
                "cluster_id": df.loc[df["store_item"] == sid, "cluster_id"].iloc[0],
                "store_cluster_id": df.loc[
                    df["store_item"] == sid, "store_cluster_id"
                ].iloc[0],
                "item_cluster_id": df.loc[
                    df["store_item"] == sid, "item_cluster_id"
                ].iloc[0],
            }

            for i, d in enumerate(window_dates, start=1):
                row[f"sales_day_{i}"] = sales_vals.get(d, 0)
                row[f"store_med_day_{i}"] = (
                    store_med.loc[store].get(d, 0) if store in store_med.index else 0
                )
                row[f"item_med_day_{i}"] = (
                    item_med.loc[item].get(d, 0) if item in item_med.index else 0
                )
                cid = row["cluster_id"]
                sc = row["store_cluster_id"]
                ic = row["item_cluster_id"]
                row[f"cluster_med_day_{i}"] = (
                    cluster_med.loc[cid].get(d, 0) if cid in cluster_med.index else 0
                )
                row[f"store_cluster_med_day_{i}"] = (
                    sc_med.loc[sc].get(d, 0) if sc in sc_med.index else 0
                )
                row[f"item_cluster_med_day_{i}"] = (
                    ic_med.loc[ic].get(d, 0) if ic in ic_med.index else 0
                )

            # pad missing days (if last window shorter than window_size)
            for i in range(len(window_dates) + 1, window_size + 1):
                for prefix in (
                    "sales",
                    "store_med",
                    "item_med",
                    "cluster_med",
                    "store_cluster_med",
                    "item_cluster_med",
                ):
                    row[f"{prefix}_day_{i}"] = 0

            records.append(row)

    return pd.DataFrame(records, columns=out_cols)


# def generate_sales_features(
#     df: pd.DataFrame,
#     window_size: int = 5,
#     cluster_map: pd.DataFrame | dict | None = None,
# ) -> pd.DataFrame:
#     """Generate sales based window features.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Input sales data with ``store``, ``item``, ``date`` and ``unit_sales``.
#     window_size : int, optional
#         Size of each rolling window, by default 5.
#     cluster_map : DataFrame or dict, optional
#         Mapping of ``store_item`` to cluster identifiers produced by
#         :func:`generate_store_item_clusters`.  If provided, a ``cluster_id``
#         column will be added as before.  Additionally, if ``cluster_map``
#         contains ``store_cluster_id`` or ``item_cluster_id`` columns,
#         corresponding median features will be generated.

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame containing sales features.
#     """

#     df = df.copy()
#     df["date"] = pd.to_datetime(df["date"])

#     if "store_item" not in df.columns:
#         df["store_item"] = df["store"].astype(str) + "_" + df["item"].astype(str)

#     cluster_lookup: dict[str, int] | None = None
#     store_cluster_lookup: dict[str, int] | None = None
#     item_cluster_lookup: dict[str, int] | None = None
#     if cluster_map is not None:
#         if isinstance(cluster_map, pd.DataFrame):
#             cols = cluster_map.columns
#             if "clusterId" in cols:
#                 cluster_lookup = cluster_map.set_index("store_item")[
#                     "clusterId"
#                 ].to_dict()
#             elif len(cols) > 1:
#                 cluster_lookup = cluster_map.set_index("store_item")[cols[-1]].to_dict()
#             if "store_cluster_id" in cols:
#                 store_cluster_lookup = cluster_map.set_index("store_item")[
#                     "store_cluster_id"
#                 ].to_dict()
#             if "item_cluster_id" in cols:
#                 item_cluster_lookup = cluster_map.set_index("store_item")[
#                     "item_cluster_id"
#                 ].to_dict()
#         else:
#             cluster_lookup = dict(cluster_map)
#         df["cluster_id"] = (
#             df["store_item"].map(cluster_lookup)
#             if cluster_lookup is not None
#             else np.nan
#         )
#         if store_cluster_lookup is not None:
#             df["store_cluster_id"] = df["store_item"].map(store_cluster_lookup)
#         else:
#             df["store_cluster_id"] = np.nan
#         if item_cluster_lookup is not None:
#             df["item_cluster_id"] = df["store_item"].map(item_cluster_lookup)
#         else:
#             df["item_cluster_id"] = np.nan
#     else:
#         df["cluster_id"] = np.nan
#         df["store_cluster_id"] = np.nan
#         df["item_cluster_id"] = np.nan

#     windows = generate_aligned_windows(df, window_size)
#     records = []

#     for window_dates in windows:
#         w_df = df[df["date"].isin(window_dates)]

#         store_med = (
#             w_df.groupby(["store", "date"])["unit_sales"].median().unstack(fill_value=0)
#         )
#         item_med = (
#             w_df.groupby(["item", "date"])["unit_sales"].median().unstack(fill_value=0)
#         )
#         cluster_med = None
#         store_cluster_med = None
#         item_cluster_med = None
#         if cluster_lookup is not None:
#             cluster_med = (
#                 w_df.groupby(["cluster_id", "date"])["unit_sales"]
#                 .median()
#                 .unstack(fill_value=0)
#             )
#         if store_cluster_lookup is not None:
#             store_cluster_med = (
#                 w_df.groupby(["store_cluster_id", "date"])["unit_sales"]
#                 .median()
#                 .unstack(fill_value=0)
#             )
#         if item_cluster_lookup is not None:
#             item_cluster_med = (
#                 w_df.groupby(["item_cluster_id", "date"])["unit_sales"]
#                 .median()
#                 .unstack(fill_value=0)
#             )
#         if store_cluster_lookup is not None:
#             store_cluster_med = (
#                 w_df.groupby(["store_cluster_id", "date"])["unit_sales"]
#                 .median()
#                 .unstack(fill_value=0)
#             )
#         if item_cluster_lookup is not None:
#             item_cluster_med = (
#                 w_df.groupby(["item_cluster_id", "date"])["unit_sales"]
#                 .median()
#                 .unstack(fill_value=0)
#             )
#         sales = (
#             w_df.groupby(["store", "item", "date"])["unit_sales"]
#             .sum()
#             .unstack(fill_value=0)
#         )

#         for (store, item), sales_vals in sales.iterrows():
#             row = {
#                 "store_item": f"{store}_{item}",
#                 "store": store,
#                 "item": item,
#                 "start_date": window_dates[0],
#             }
#             if cluster_lookup is not None:
#                 row["cluster_id"] = cluster_lookup.get(f"{store}_{item}")
#             else:
#                 row["cluster_id"] = np.nan
#             if store_cluster_lookup is not None:
#                 row["store_cluster_id"] = store_cluster_lookup.get(f"{store}_{item}")
#             else:
#                 row["store_cluster_id"] = np.nan
#             if item_cluster_lookup is not None:
#                 row["item_cluster_id"] = item_cluster_lookup.get(f"{store}_{item}")
#             else:
#                 row["item_cluster_id"] = np.nan

#             for i in range(1, window_size + 1):
#                 try:
#                     d = window_dates[i - 1]
#                     row[f"sales_day_{i}"] = sales_vals.get(d, 0)
#                     row[f"store_med_day_{i}"] = (
#                         store_med.loc[store].get(d, 0)
#                         if store in store_med.index
#                         else 0
#                     )
#                     row[f"item_med_day_{i}"] = (
#                         item_med.loc[item].get(d, 0) if item in item_med.index else 0
#                     )
#                     if cluster_lookup is not None:
#                         row[f"cluster_med_day_{i}"] = (
#                             cluster_med.loc[row["cluster_id"]].get(d, 0)
#                             if row["cluster_id"] in cluster_med.index
#                             else 0
#                         )
#                     if store_cluster_lookup is not None:
#                         row[f"store_cluster_med_day_{i}"] = (
#                             store_cluster_med.loc[row["store_cluster_id"]].get(d, 0)
#                             if row["store_cluster_id"] in store_cluster_med.index
#                             else 0
#                         )
#                     if item_cluster_lookup is not None:
#                         row[f"item_cluster_med_day_{i}"] = (
#                             item_cluster_med.loc[row["item_cluster_id"]].get(d, 0)
#                             if row["item_cluster_id"] in item_cluster_med.index
#                             else 0
#                         )
#                 except IndexError:
#                     row[f"sales_day_{i}"] = 0
#                     row[f"store_med_day_{i}"] = 0
#                     row[f"item_med_day_{i}"] = 0
#                     if cluster_lookup is not None:
#                         row[f"cluster_med_day_{i}"] = 0
#                     if store_cluster_lookup is not None:
#                         row[f"store_cluster_med_day_{i}"] = 0
#                     if item_cluster_lookup is not None:
#                         row[f"item_cluster_med_day_{i}"] = 0

#             records.append(row)

#     # Assemble DataFrame
#     cols = ["start_date", "store_item", "store", "item", "cluster_id"]
#     if store_cluster_lookup is not None:
#         cols.append("store_cluster_id")
#     if item_cluster_lookup is not None:
#         cols.append("item_cluster_id")
#     sales_cols = [f"sales_day_{i}" for i in range(1, window_size + 1)]
#     store_med_cols = [f"store_med_day_{i}" for i in range(1, window_size + 1)]
#     item_med_cols = [f"item_med_day_{i}" for i in range(1, window_size + 1)]
#     cluster_med_cols = (
#         [f"cluster_med_day_{i}" for i in range(1, window_size + 1)]
#         if cluster_lookup is not None
#         else []
#     )
#     store_cluster_med_cols = (
#         [f"store_cluster_med_day_{i}" for i in range(1, window_size + 1)]
#         if store_cluster_lookup is not None
#         else []
#     )
#     item_cluster_med_cols = (
#         [f"item_cluster_med_day_{i}" for i in range(1, window_size + 1)]
#         if item_cluster_lookup is not None
#         else []
#     )

#     cols.extend(
#         sales_cols
#         + store_med_cols
#         + item_med_cols
#         + cluster_med_cols
#         + store_cluster_med_cols
#         + item_cluster_med_cols
#     )

#     if not records:
#         return pd.DataFrame(columns=cols)

#     df = pd.DataFrame(records)
#     return df[cols]


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


def add_next_window_targets(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
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

    for store_item, group in df.groupby("store_item"):
        shifted = group[cols_to_shift].shift(-1)
        result.loc[group.index, [f"y_{c}" for c in cols_to_shift]] = shifted.values

    return result


def prepare_training_data_from_raw_df(df, window_size=16):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "store_item" not in df.columns:
        df["store_item"] = df["store"].astype(str) + "_" + df["item"].astype(str)

    # Removed unused variable
    cyc_df = generate_cyclical_features(df, window_size)
    print(f"cyc_df.shape: {cyc_df.shape}")
    sales_df = generate_sales_features(df, window_size)
    print(f"sales_df.shape: {sales_df.shape}")

    merged_df = pd.merge(
        cyc_df, sales_df, on=["start_date", "store_item", "store", "item"], how="inner"
    )
    print(f"merged_df.shape: {merged_df.shape}")
    merged_df = add_y_targets_from_shift(merged_df, window_size)
    print(f"merged_df.shape: {merged_df.shape}")

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
        X = StandardScaler().fit_transform(df)
        X = StandardScaler().fit_transform(X.T).T
    else:
        X = df.values

    return X, df


def compute_spectral_biclustering_row_cv_scores(
    data,
    n_clusters_row_range=range(2, 11),  # e.g. 2…10 clusters
    cv_folds=3,
    true_row_labels=None,
):
    """
    Cross‑validate SpectralBiclustering over a range of row‑cluster counts
    (one column cluster) and return a DataFrame with:
      * Explained Variance (%)
      * Mean Silhouette
      * Mean ARI (if `true_row_labels` supplied)
    """

    # ------------------------------------------------------------------
    # Helper to avoid warnings when all values are NaN
    # ------------------------------------------------------------------
    def _safe_mean(arr):
        return np.nan if np.all(np.isnan(arr)) else np.nanmean(arr)

    X = data.copy()
    n_rows, _ = X.shape
    results = []
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Restrict range so we never request more row‑clusters than rows‑1
    max_row = min(n_rows - 1, max(n_clusters_row_range))
    row_range = range(
        max(2, min(n_clusters_row_range)),  # start ≥2
        max_row + 1,  # inclusive end
    )

    for n_row in row_range:
        print(f"Evaluating n_row = {n_row}")
        pve_fold, sil_fold, ari_fold = [], [], []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]

            try:
                # --- Fit SpectralBiclustering with *row* clusters only ----
                model = SpectralBiclustering(
                    n_clusters=n_row,  # integer → clusters rows only
                    method="log",
                    random_state=42,
                )
                model.fit(X_train)

                row_labels = model.row_labels_

                # ----- % Variance Explained on test rows -----
                global_mean = X_train.mean(axis=0)
                total_ss = np.sum((X_test - global_mean) ** 2)

                recon_err = 0.0
                for xi in X_test:
                    # nearest row‑cluster centroid
                    best_err = np.inf
                    for cid in range(n_row):
                        mask = row_labels == cid
                        if np.any(mask):
                            centroid = X_train[mask].mean(axis=0)
                            best_err = min(best_err, np.linalg.norm(xi - centroid) ** 2)
                    recon_err += best_err

                pve_fold.append(100 * (1 - recon_err / total_ss))

                # ----- Assign test rows to nearest centroid -----
                test_labels = np.array(
                    [
                        np.argmin(
                            [
                                (
                                    np.linalg.norm(
                                        xi - X_train[row_labels == cid].mean(axis=0)
                                    )
                                    if np.any(row_labels == cid)
                                    else np.inf
                                )
                                for cid in range(n_row)
                            ]
                        )
                        for xi in X_test
                    ]
                )

                # ARI if ground‑truth supplied
                if true_row_labels is not None:
                    ari_fold.append(
                        adjusted_rand_score(true_row_labels[test_idx], test_labels)
                    )

                # Silhouette
                try:
                    sil_fold.append(silhouette_score(X_test, test_labels))
                except ValueError:  # only one label in this fold
                    sil_fold.append(np.nan)

            except Exception as e:
                print(f"[FAIL] n_row={n_row} → {e}")
                pve_fold.append(np.nan)
                sil_fold.append(np.nan)
                ari_fold.append(np.nan)

        # Aggregate across CV folds
        results.append(
            {
                "n_row": n_row,
                "Explained Variance (%)": _safe_mean(pve_fold),
                "Mean Silhouette": _safe_mean(sil_fold),
                "Mean ARI": _safe_mean(ari_fold),
            }
        )

    return pd.DataFrame(results)


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


def compute_spectral_clustering_cv_scores(
    data,
    *,
    model_class=SpectralBiclustering,
    n_clusters_row_range=range(2, 6),
    n_clusters_col_range=range(2, 6),
    cv_folds=3,
    true_row_labels=None,
    model_kwargs=None,
):
    """Cross‑validate Spectral[Bic|Co]clustering and SpectralClustering."""

    def _safe_mean(arr):
        return np.nan if np.all(np.isnan(arr)) else np.nanmean(arr)

    if model_kwargs is None:
        model_kwargs = {}
    model_kwargs = dict(model_kwargs)
    model_kwargs.setdefault("random_state", 42)

    X = np.asarray(data)
    n_rows, n_cols = X.shape
    results = []
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # ───────────────────────────────────────────────────────────────────
    # 0.  *NEW* — find the *smallest* training‑fold size
    # ───────────────────────────────────────────────────────────────────
    min_train_samples = int(np.floor(n_rows * (cv_folds - 1) / cv_folds))
    max_row_clusters = min(n_rows, min_train_samples)
    n_clusters_row_range = [k for k in n_clusters_row_range if k <= max_row_clusters]

    # Decide how many loop dimensions we actually have
    if model_class is SpectralClustering:
        col_range = [None]
        loop_over_col = False
    elif model_class is SpectralCoclustering:
        col_range = [None]
        loop_over_col = False
    else:  # SpectralBiclustering
        col_range = [None]
        #        col_range = [c for c in n_clusters_col_range if c <= n_cols]
        loop_over_col = False

    # ───────────────────────────────────────────────────────────────────
    # 1.  Grid search
    # ───────────────────────────────────────────────────────────────────
    for n_row in n_clusters_row_range:
        for n_col in col_range:
            msg = f"Evaluating n_row={n_row}"
            if loop_over_col:
                msg += f", n_col={n_col}"
            print(msg)

            pve_list, sil_list, ari_list = [], [], []

            for train_idx, test_idx in kf.split(X):

                # *NEW* — skip this fold if it has too few samples
                if n_row > len(train_idx):
                    pve_list.append(np.nan)
                    sil_list.append(np.nan)
                    ari_list.append(np.nan)
                    continue

                X_train, X_test = X[train_idx], X[test_idx]

                try:
                    # Instantiate model
                    if model_class is SpectralClustering:
                        model = model_class(n_clusters=n_row, **model_kwargs)
                    elif model_class is SpectralCoclustering:
                        model = model_class(n_clusters=n_row, **model_kwargs)
                    else:  # SpectralBiclustering
                        n_col_eff = n_row if n_col is None else n_col
                        model = model_class(n_clusters=(n_row, n_col_eff), **model_kwargs)

                    model.fit(X_train)

                    # Row labels
                    if hasattr(model, "row_labels_"):
                        row_labels = model.row_labels_
                    elif hasattr(model, "labels_"):
                        row_labels = model.labels_
                    else:  # SpectralCoclustering
                        row_labels = np.argmax(model.rows_, axis=0)

                    # % Variance explained
                    global_mean = X_train.mean(axis=0)
                    total_ss = np.sum((X_test - global_mean) ** 2)

                    recon_error = 0.0
                    for xi in X_test:
                        best_err = np.inf
                        for cid in range(n_row):
                            mask = row_labels == cid
                            if np.any(mask):
                                centroid = X_train[mask].mean(axis=0)
                                best_err = min(
                                    best_err,
                                    np.linalg.norm(xi - centroid) ** 2,
                                )
                        recon_error += best_err
                    pve_list.append(100 * (1 - recon_error / total_ss))

                    # Predicted labels for test rows
                    test_labels = np.array(
                        [
                            np.argmin(
                                [
                                    (
                                        np.linalg.norm(
                                            xi - X_train[row_labels == cid].mean(axis=0)
                                        )
                                        if np.any(row_labels == cid)
                                        else np.inf
                                    )
                                    for cid in range(n_row)
                                ]
                            )
                            for xi in X_test
                        ]
                    )

                    # ARI
                    if true_row_labels is not None:
                        ari_list.append(
                            adjusted_rand_score(true_row_labels[test_idx], test_labels)
                        )

                    # Silhouette
                    try:
                        sil_list.append(silhouette_score(X_test, test_labels))
                    except ValueError:
                        sil_list.append(np.nan)

                except Exception as e:
                    fail_msg = f"[FAIL] n_row={n_row}"
                    if loop_over_col:
                        fail_msg += f", n_col={n_col}"
                    print(f"{fail_msg} → {e}")
                    pve_list.append(np.nan)
                    sil_list.append(np.nan)
                    ari_list.append(np.nan)

            results.append(
                {
                    "n_row": n_row,
                    "n_col": n_col if loop_over_col else np.nan,
                    "Explained Variance (%)": _safe_mean(pve_list),
                    "Mean Silhouette": _safe_mean(sil_list),
                    "Mean ARI": _safe_mean(ari_list),
                }
            )

    return pd.DataFrame(results)
