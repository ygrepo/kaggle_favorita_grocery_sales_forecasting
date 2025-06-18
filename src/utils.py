from __future__ import annotations

import heapq
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import (
    SpectralBiclustering,
    SpectralClustering,
    SpectralCoclustering,
)

from typing import Optional, List


def build_feature_and_label_cols(window_size: int) -> tuple[list[str], list[str]]:
    """Return feature and label column names for a given window size."""
    meta_cols = [
        "start_date",
        "store_item",
        "store",
        "item",
        "storeClusterId",
        "itemClusterId",
    ]
    x_cyclical_features = [
        f"{feat}_{trig}_{i}"
        for feat in ["dayofweek", "weekofmonth", "monthofyear", "paycycle", "season"]
        for trig in ["sin", "cos"]
        for i in range(1, window_size + 1)
    ]

    x_sales_features = [
        f"{name}_{i}"
        for name in ["sales_day", "store_med_day", "item_med_day"]
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


# ------------------------------------------------------------------ #
# helper assumed to exist (unchanged)                                #
# ------------------------------------------------------------------ #
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

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if calendar_aligned:
        # --- back‑to‑front, gap‑free windows ---
        last_date = df["date"].max().normalize()  # ensure time == 00:00
        first_date = df["date"].min().normalize()

        starts = []
        current = last_date
        while current >= first_date + pd.Timedelta(days=window_size - 1):
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
        return [
            unique_dates[i : i + window_size]
            for i in range(0, len(unique_dates), window_size)
        ]


def generate_cyclical_features(
    df: pd.DataFrame,
    window_size: int = 7,
    *,
    store_clusters: Optional[pd.DataFrame] = None,  # ['store', 'clusterId']
    item_clusters: Optional[pd.DataFrame] = None,  # ['item',  'clusterId']
    calendar_aligned: bool = True,
) -> pd.DataFrame:
    """
    Generate day‑/week‑/month/season/pay‑cycle sine & cosine features
    for non‑overlapping windows and *include* store/item cluster IDs.

    If `store_clusters` or `item_clusters` is None, every store (or item)
    is assigned to the single cluster label 'ALL_STORES' / 'ALL_ITEMS'.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # ────────────────────────────────────────────────────────────────
    # Attach cluster IDs (mirrors generate_sales_features)
    # ────────────────────────────────────────────────────────────────
    if store_clusters is not None:
        mapping = store_clusters[["store", "clusterId"]].drop_duplicates()
        df = df.merge(mapping, on="store", how="left")
        df["storeClusterId"] = df["clusterId"]
        df.drop(columns=["clusterId"], inplace=True)
    else:
        df["storeClusterId"] = "ALL_STORES"
    df["storeClusterId"] = df["storeClusterId"].fillna("ALL_STORES")

    if item_clusters is not None:
        mapping = item_clusters[["item", "clusterId"]].drop_duplicates()
        df = df.merge(mapping, on="item", how="left")
        df["itemClusterId"] = df["clusterId"]
        df.drop(columns=["clusterId"], inplace=True)
    else:
        df["itemClusterId"] = "ALL_ITEMS"
    df["itemClusterId"] = df["itemClusterId"].fillna("ALL_ITEMS")

    # Handy lookup dicts (optional—used later for speed)
    store_to_cluster = (
        df.drop_duplicates("store")[["store", "storeClusterId"]]
        .set_index("store")["storeClusterId"]
        .to_dict()
    )
    item_to_cluster = (
        df.drop_duplicates("item")[["item", "itemClusterId"]]
        .set_index("item")["itemClusterId"]
        .to_dict()
    )

    # Ensure store_item column
    if "store_item" not in df.columns:
        df["store_item"] = df["store"].astype(str) + "_" + df["item"].astype(str)

    # ───────────────── column template ─────────────────
    cols = [
        "start_date",
        "store_item",
        "store",
        "item",
        "storeClusterId",
        "itemClusterId",
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

    for store_item, group in df.groupby("store_item"):
        group = group.sort_values("date").reset_index(drop=True)
        windows = generate_aligned_windows(
            group, window_size, calendar_aligned=calendar_aligned
        )

        # cluster ids for this store‑item
        s_cl = store_to_cluster.get(group["store"].iloc[0], "ALL_STORES")
        i_cl = item_to_cluster.get(group["item"].iloc[0], "ALL_ITEMS")

        for window_dates in windows:
            window_df = group[group["date"].isin(window_dates)]

            row = {
                "start_date": window_df["date"].min(),
                "store_item": store_item,
                "store": window_df["store"].iloc[0],
                "item": window_df["item"].iloc[0],
                "storeClusterId": s_cl,
                "itemClusterId": i_cl,
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


def generate_sales_features(
    df: pd.DataFrame,
    window_size: int = 5,
    *,
    store_clusters: Optional[pd.DataFrame] = None,  # ['store', 'clusterId']
    item_clusters: Optional[pd.DataFrame] = None,  # ['item',  'clusterId']
    calendar_aligned: bool = True,
) -> pd.DataFrame:
    """
    If `store_clusters` / `item_clusters` is None, all stores (or items)
    are assigned to a single shared cluster labelled 'ALL_STORES' / 'ALL_ITEMS'.
    """

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # ------------------------------------------------------------------
    # 1) 𝙎𝙩𝙤𝙧𝙚‑𝙘𝙡𝙪𝙨𝙩𝙚𝙧 attachment
    # ------------------------------------------------------------------
    if store_clusters is not None:
        mapping = store_clusters[["store", "clusterId"]].drop_duplicates()
        df = df.merge(mapping, on="store", how="left")
        df["storeClusterId"] = df["clusterId"]  # may still contain NaN
        df.drop(columns=["clusterId"], inplace=True)
    else:
        # every store goes into the same cluster
        df["storeClusterId"] = "ALL_STORES"

    # any store without a mapping (NaN) also joins the catch‑all cluster
    df["storeClusterId"] = df["storeClusterId"].fillna("ALL_STORES")

    # ------------------------------------------------------------------
    # 2) 𝙄𝙩𝙚𝙢‑𝙘𝙡𝙪𝙨𝙩𝙚𝙧 attachment
    # ------------------------------------------------------------------
    if item_clusters is not None:
        mapping = item_clusters[["item", "clusterId"]].drop_duplicates()
        df = df.merge(mapping, on="item", how="left")
        df["itemClusterId"] = df["clusterId"]
        df.drop(columns=["clusterId"], inplace=True)
    else:
        df["itemClusterId"] = "ALL_ITEMS"

    df["itemClusterId"] = df["itemClusterId"].fillna("ALL_ITEMS")

    # fast look‑ups: store → cluster, item → cluster
    store_to_cluster = (
        df.drop_duplicates("store")[["store", "storeClusterId"]]
        .set_index("store")["storeClusterId"]
        .to_dict()
    )
    item_to_cluster = (
        df.drop_duplicates("item")[["item", "itemClusterId"]]
        .set_index("item")["itemClusterId"]
        .to_dict()
    )

    windows = generate_aligned_windows(
        df, window_size, calendar_aligned=calendar_aligned
    )
    records: List[dict] = []

    for window_dates in windows:
        w_df = df[df["date"].isin(window_dates)]

        # cluster‑level medians
        store_med = (
            w_df.groupby(["storeClusterId", "date"])["unit_sales"]
            .median()
            .unstack(fill_value=0)
        )
        item_med = (
            w_df.groupby(["itemClusterId", "date"])["unit_sales"]
            .median()
            .unstack(fill_value=0)
        )

        # raw sales at store‑item level
        sales = (
            w_df.groupby(["store", "item", "date"])["unit_sales"]
            .sum()
            .unstack(fill_value=0)
        )

        for (store, item), sales_vals in sales.iterrows():
            s_cl = store_to_cluster.get(store, "ALL_STORES")
            i_cl = item_to_cluster.get(item, "ALL_ITEMS")

            row = {
                "store_item": f"{store}_{item}",
                "store": store,
                "item": item,
                "storeClusterId": s_cl,
                "itemClusterId": i_cl,
                "start_date": window_dates[0],
            }

            s_cl = store_to_cluster.get(store, "ALL_STORES")
            i_cl = item_to_cluster.get(item, "ALL_ITEMS")

            for i in range(1, window_size + 1):
                d = window_dates[i - 1] if i - 1 < len(window_dates) else None

                if d is not None:
                    row[f"sales_day_{i}"] = sales_vals.get(d, 0)
                    row[f"store_med_day_{i}"] = (
                        store_med.loc[s_cl].get(d, 0) if s_cl in store_med.index else 0
                    )
                    row[f"item_med_day_{i}"] = (
                        item_med.loc[i_cl].get(d, 0) if i_cl in item_med.index else 0
                    )
                else:
                    row[f"sales_day_{i}"] = 0
                    row[f"store_med_day_{i}"] = 0
                    row[f"item_med_day_{i}"] = 0

            records.append(row)

    # ------------------------------------------------------------------
    # final column order
    # ------------------------------------------------------------------
    cols = [
        "start_date",
        "store_item",
        "store",
        "item",
        "storeClusterId",
        "itemClusterId",
    ]
    cols += [
        f"{prefix}{i}"
        for prefix in ("sales_day_", "store_med_day_", "item_med_day_")
        for i in range(1, window_size + 1)
    ]
    return (
        pd.DataFrame(records, columns=cols) if records else pd.DataFrame(columns=cols)
    )


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


def prepare_training_data_from_raw_df(
    df, window_size=16, store_clusters=None, item_clusters=None, calendar_aligned=True
):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "store_item" not in df.columns:
        df["store_item"] = df["store"].astype(str) + "_" + df["item"].astype(str)

    # Removed unused variable
    cyc_df = generate_cyclical_features(
        df,
        window_size,
        store_clusters=store_clusters,
        item_clusters=item_clusters,
        calendar_aligned=calendar_aligned,
    )
    print(f"cyc_df.shape: {cyc_df.shape}")
    # print(cyc_df.columns)
    # print(cyc_df.head())
    sales_df = generate_sales_features(
        df,
        window_size,
        store_clusters=store_clusters,
        item_clusters=item_clusters,
        calendar_aligned=calendar_aligned,
    )
    print(f"sales_df.shape: {sales_df.shape}")
    # print(sales_df.columns)
    # print(sales_df.head())

    merged_df = pd.merge(
        sales_df,
        cyc_df,
        on=[
            "start_date",
            "store_item",
            "store",
            "item",
            "storeClusterId",
            "itemClusterId",
        ],
        how="inner",
    )
    print(f"merged_df.shape: {merged_df.shape}")
    # print(merged_df.columns)
    # print(merged_df.head())
    merged_df = add_y_targets_from_shift(merged_df, window_size)
    print(f"merged_df.shape: {merged_df.shape}")
    # print(merged_df.columns)
    # print(merged_df.head())

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
                        model = model_class(
                            n_clusters=(n_row, n_col_eff), **model_kwargs
                        )

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


def reorder_data(data, row_labels, col_labels):
    row_order = np.argsort(row_labels)
    col_order = np.argsort(col_labels)
    return data[np.ix_(row_order, col_order)], row_order, col_order
