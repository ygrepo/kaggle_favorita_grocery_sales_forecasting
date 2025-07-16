from __future__ import annotations

import heapq
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional, Iterator
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
        # "id",
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
            "store_med_change",
            "item_med_change",
            "store_cluster_logpct_change",
            "item_cluster_logpct_change",
        ]
        for i in range(1, window_size + 1)
    ]

    x_feature_cols = x_sales_features + x_cyclical_features
    label_cols = [f"y_sales_day_{i}" for i in range(1, window_size + 1)]
    # label_cols = [f"y_{c}" for c in x_feature_cols]
    # y_sales_features = [f"y_{c}" for c in x_sales_features]
    # y_cyclical_features = [f"y_{c}" for c in x_cyclical_features]
    return (
        meta_cols,
        x_sales_features,
        x_cyclical_features,
        x_feature_cols,
        label_cols,
        # y_sales_features,
        # y_cyclical_features,
    )


def load_data(
    data_fn: Path,
    window_size: int,
    output_fn: Path,
    log_level: str,
) -> pd.DataFrame:
    """Load and preprocess training data.

    Args:
        data_fn: Path to the training data file

    Returns:
        Preprocessed DataFrame
    """
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.info(f"Loading data from {data_fn}")
    try:
        dtype_dict = {
            "start_date": np.uint32,
            "store_item": str,
            "store_cluster": np.uint8,
            "item_cluster": np.uint8,
            "unit_sales": np.float32,
            **{f"store_med_day_{i}": np.float32 for i in range(1, window_size + 1)},
            **{f"item_med_day_{i}": np.float32 for i in range(1, window_size + 1)},
            **{f"store_med_change_{i}": np.float32 for i in range(1, window_size + 1)},
            **{f"item_med_change_{i}": np.float32 for i in range(1, window_size + 1)},
            **{
                f"store_cluster_logpct_change_{i}": np.float32
                for i in range(1, window_size + 1)
            },
            **{
                f"item_cluster_logpct_change_{i}": np.float32
                for i in range(1, window_size + 1)
            },
            **{
                f"dayofweek_{trig}_{i}": np.float32
                for i in range(1, window_size + 1)
                for trig in TRIGS
            },
            **{
                f"weekofmonth_{trig}_{i}": np.float32
                for i in range(1, window_size + 1)
                for trig in TRIGS
            },
            **{
                f"monthofyear_{trig}_{i}": np.float32
                for i in range(1, window_size + 1)
                for trig in TRIGS
            },
            **{
                f"paycycle_{trig}_{i}": np.float32
                for i in range(1, window_size + 1)
                for trig in TRIGS
            },
            **{
                f"season_{trig}_{i}": np.float32
                for i in range(1, window_size + 1)
                for trig in TRIGS
            },
        }
        df = pd.read_csv(
            data_fn, dtype=dtype_dict, parse_dates=["start_date"], low_memory=False
        )
        (meta_cols, _, _, x_feature_cols, label_cols) = build_feature_and_label_cols(
            window_size=window_size
        )
        df = df[meta_cols + x_feature_cols + label_cols]
        df["start_date"] = pd.to_datetime(df["start_date"])
        df.sort_values("start_date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        if output_fn:
            logger.info(f"Saving final_df to {output_fn}")
            df.to_csv(output_fn, index=False)

        logger.info(f"Loaded data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def load_X_y_data(
    data_fn: Path,
    window_size: int,
    log_level: str,
) -> pd.DataFrame:
    """Load and preprocess training data.

    Args:
        data_fn: Path to the training data file

    Returns:
        Preprocessed DataFrame
    """
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.info(f"Loading data from {data_fn}")
    try:
        dtype_dict = {
            "start_date": np.uint32,
            "store_item": str,
            "store_cluster": np.uint8,
            "item_cluster": np.uint8,
            **{f"store_med_day_{i}": np.float32 for i in range(1, window_size + 1)},
            **{f"item_med_day_{i}": np.float32 for i in range(1, window_size + 1)},
            **{f"store_med_change_{i}": np.float32 for i in range(1, window_size + 1)},
            **{f"item_med_change_{i}": np.float32 for i in range(1, window_size + 1)},
            **{
                f"store_cluster_logpct_change_{i}": np.float32
                for i in range(1, window_size + 1)
            },
            **{
                f"item_cluster_logpct_change_{i}": np.float32
                for i in range(1, window_size + 1)
            },
            **{
                f"dayofweek_{trig}_{i}": np.float32
                for i in range(1, window_size + 1)
                for trig in TRIGS
            },
            **{
                f"weekofmonth_{trig}_{i}": np.float32
                for i in range(1, window_size + 1)
                for trig in TRIGS
            },
            **{
                f"monthofyear_{trig}_{i}": np.float32
                for i in range(1, window_size + 1)
                for trig in TRIGS
            },
            **{
                f"paycycle_{trig}_{i}": np.float32
                for i in range(1, window_size + 1)
                for trig in TRIGS
            },
            **{
                f"season_{trig}_{i}": np.float32
                for i in range(1, window_size + 1)
                for trig in TRIGS
            },
            **{f"sales_day_{i}": np.float32 for i in range(1, window_size + 1)},
            **{f"y_sales_day_{i}": np.float32 for i in range(1, window_size + 1)},
        }
        df = pd.read_csv(
            data_fn, dtype=dtype_dict, parse_dates=["start_date"], low_memory=False
        )
        (meta_cols, _, _, x_feature_cols, label_cols) = build_feature_and_label_cols(
            window_size=window_size
        )
        df = df[meta_cols + x_feature_cols + label_cols]
        df["start_date"] = pd.to_datetime(df["start_date"])
        df = df.sort_values(["store_item", "start_date"]).reset_index(drop=True)

        logger.info(f"Loaded data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


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
            # logger.debug(f"Current date: {current}")
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
        # logger.debug(f"Unique dates: {unique_dates}")
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
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # df.dropna(subset=["date"], inplace=True)

    # Extract date components once
    dates = df["date"]
    days = dates.dt.day.astype("uint8")
    months = dates.dt.month.astype("uint8")
    years = dates.dt.year.astype("uint16")

    # Day/week/month
    df["dayofweek"] = dates.dt.dayofweek.astype("uint8")
    df["weekofmonth"] = ((days - 1) // 7 + 1).astype("uint8")
    df["monthofyear"] = months

    # Sin/Cos transforms
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7).astype("float32")
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7).astype("float32")
    df["weekofmonth_sin"] = np.sin(2 * np.pi * df["weekofmonth"] / 5).astype("float32")
    df["weekofmonth_cos"] = np.cos(2 * np.pi * df["weekofmonth"] / 5).astype("float32")
    df["monthofyear_sin"] = np.sin(2 * np.pi * months / 12).astype("float32")
    df["monthofyear_cos"] = np.cos(2 * np.pi * months / 12).astype("float32")

    # Paycycle
    month_end_day = (dates + pd.offsets.MonthEnd(0)).dt.day
    last_pay = dates.where(
        days >= 15,
        pd.to_datetime(
            {
                "year": years - (months == 1).astype("uint16"),
                "month": np.where(months > 1, months - 1, 12),
                "day": 1,
            }
        )
        + pd.offsets.MonthEnd(0),
    )
    next_pay = pd.to_datetime(
        {
            "year": years,
            "month": months,
            "day": np.where(days >= 15, month_end_day, 15),
        }
    )
    cycle_len = (next_pay - last_pay).dt.days.replace(0, np.nan).astype("float32")
    elapsed = (dates - last_pay).dt.days.astype("float32")
    paycycle_ratio = (elapsed / cycle_len).fillna(0).astype("float32")
    df["paycycle_sin"] = np.sin(2 * np.pi * paycycle_ratio).astype("float32")
    df["paycycle_cos"] = np.cos(2 * np.pi * paycycle_ratio).astype("float32")

    # Season
    spring_equinox = pd.to_datetime(dict(year=years, month=3, day=20))
    adjusted_years = np.where(dates < spring_equinox, years - 1, years)
    adjusted_equinox = pd.to_datetime(dict(year=adjusted_years, month=3, day=20))
    days_since = (dates - adjusted_equinox).dt.days.astype("float32")
    season_ratio = ((days_since % 365) / 365).astype("float32")
    df["season_sin"] = np.sin(2 * np.pi * season_ratio).astype("float32")
    df["season_cos"] = np.cos(2 * np.pi * season_ratio).astype("float32")

    # ───────────────── build window rows ─────────────────
    results: List[dict] = []
    iterator = df.groupby("store_item")
    if logger.level == logging.DEBUG:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="Generating cyclical features")

    for store_item, group in iterator:
        group = group.sort_values("date").reset_index(drop=True)
        windows = generate_aligned_windows(
            group, window_size, calendar_aligned=calendar_aligned
        )

        for i, window_dates in enumerate(windows):
            window_df = group[group["date"].isin(window_dates)]
            if window_df.empty:
                continue

            row = {
                "start_date": window_dates[0],
                "store_item": store_item,
                "store": group["store"].iloc[0],
                "item": group["item"].iloc[0],
            }

            for j in range(window_size):
                if j < len(window_df):
                    r = window_df.iloc[j]
                    for f in CYCLICAL_FEATURES:
                        for t in TRIGS:
                            row[f"{f}_{t}_{j+1}"] = r.get(f"{f}_{t}", 0.0)
                else:
                    for f in CYCLICAL_FEATURES:
                        for t in TRIGS:
                            row[f"{f}_{t}_{j+1}"] = 0.0
            results.append(row)

    final_cols = ["start_date", "store_item", "store", "item"] + [
        f"{f}_{t}_{i}"
        for i in range(1, window_size + 1)
        for f in CYCLICAL_FEATURES
        for t in TRIGS
    ]
    df = pd.DataFrame(results, columns=final_cols)
    del results
    gc.collect()

    if output_path is not None:
        if output_path.suffix == ".parquet":
            df.to_parquet(output_path)
        else:
            df.to_csv(output_path, index=False)
        logger.info(f"Saving cyclical features to {output_path}")

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

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Drop duplicates to get one id per store_item
    # id_mapping = df[["store_item", "id"]].drop_duplicates()

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
            store_median_change = store_median_curr.replace(
                0, np.nan
            ) / prev_store_med.replace(0, np.nan)
            store_median_log_pct_change = np.log(store_median_change)
        else:
            store_median_change = pd.Series(np.nan, index=store_median_curr.index)
            store_median_log_pct_change = pd.Series(
                np.nan, index=store_median_curr.index
            )

        if prev_item_med is not None:
            item_median_change = item_median_curr.replace(
                0, np.nan
            ) / prev_item_med.replace(0, np.nan)
            item_median_log_pct_change = np.log(item_median_change)
        else:
            item_median_change = pd.Series(np.nan, index=item_median_curr.index)
            item_median_log_pct_change = pd.Series(np.nan, index=item_median_curr.index)

        prev_store_med = store_median_curr.copy()
        prev_item_med = item_median_curr.copy()

        # no groupby needed for store-item, just pivot
        sales = w_df.pivot(
            index=["store", "item"], columns="date", values="unit_sales"
        ).fillna(0)
        del w_df
        gc.collect()

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
                    row[f"store_med_change_{i}"] = (
                        store_median_change.get(s_cl, np.nan)
                        if s_cl in store_median_change.index
                        else np.nan
                    )
                    row[f"store_cluster_logpct_change_{i}"] = (
                        store_median_log_pct_change.get(s_cl, np.nan)
                        if s_cl in store_median_log_pct_change.index
                        else np.nan
                    )
                    row[f"item_med_day_{i}"] = (
                        item_med.loc[i_cl].get(d, 0)
                        if i_cl in item_med.index
                        else np.nan
                    )
                    row[f"item_med_change_{i}"] = (
                        item_median_change.get(i_cl, np.nan)
                        if i_cl in item_median_change.index
                        else np.nan
                    )
                    row[f"item_cluster_logpct_change_{i}"] = (
                        item_median_log_pct_change.get(i_cl, np.nan)
                        if i_cl in item_median_log_pct_change.index
                        else np.nan
                    )
                else:
                    continue

            records.append(row)

    # Explicitly free up memory for each window
    del store_med, item_med
    gc.collect()
    del sales
    gc.collect()

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
            "store_med_change_",
            "item_med_change_",
            "store_cluster_logpct_change_",
            "item_cluster_logpct_change_",
        )
        for i in range(1, window_size + 1)
    ]
    df = pd.DataFrame(records, columns=cols) if records else pd.DataFrame(cols)
    del records
    gc.collect()
    # df = df.merge(id_mapping, on=["store_item"], how="left")
    # cols.insert(cols.index("start_date") + 1, "id")
    df = df[cols]
    # del id_mapping
    gc.collect()
    if output_path is not None:
        logger.info(f"Saving sales features to {output_path}")
        if output_path.suffix == ".parquet":
            df.to_parquet(output_path)
        else:
            df.to_csv(output_path, index=False)
    return df


def create_y_targets_from_shift(
    df: pd.DataFrame,
    window_size: int = 16,
    feature_prefixes: Optional[list[str]] = None,
    log_level: str = "INFO",
) -> pd.DataFrame:
    """
    Create a DataFrame with y-targets from shifted windows per store_item group.
    Only valid row pairs are kept (shifted by window_size days).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with 'store_item' and 'start_date'.
    window_size : int
        Expected gap in days between consecutive windows.
    feature_prefixes : list[str], optional
        List of feature prefixes to target for creating y_ features.
    log_level : str
        Logging level, e.g. "INFO", "DEBUG".

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with y_ columns added (only for valid rows).
    """

    def add_y_targets_from_shift(
        df: pd.DataFrame,
        window_size: int,
        feature_prefixes: list[str],
    ) -> Iterator[pd.DataFrame]:

        for store_item, group in tqdm(
            df.groupby("store_item", sort=False), desc="Processing store_items"
        ):
            group = group.sort_values("start_date").reset_index(drop=True)
            next_group = group.shift(-1)

            date_diff = (next_group["start_date"] - group["start_date"]).dt.days
            valid = date_diff == window_size

            if logger.isEnabledFor(logging.DEBUG):
                group_dates = group["start_date"].dt.strftime("%Y-%m-%d")
                next_dates = next_group["start_date"].dt.strftime("%Y-%m-%d")

                if len(group_dates) <= 10:
                    logger.debug(f"Group {store_item} dates: {group_dates.tolist()}")
                    logger.debug(f"Next dates: {next_dates.tolist()}")
                    logger.debug(f"Date diffs: {date_diff.tolist()}")
                else:
                    logger.debug(
                        f"Group {store_item}: {group_dates.iloc[0]} to {group_dates.iloc[-1]}"
                    )
                    logger.debug(f"Next: {next_dates.iloc[0]} to {next_dates.iloc[-1]}")
                    logger.debug(
                        f"Min/Max date diff: {date_diff.min()} / {date_diff.max()}"
                    )

            matched = group.loc[valid].copy()
            if matched.empty:
                logger.debug(f"No valid window for store_item {store_item}")
                continue

            shifted = next_group.loc[valid]

            for col in group.columns:
                if any(col.startswith(prefix) for prefix in feature_prefixes):
                    matched[f"y_{col}"] = shifted[col].values

            yield matched

            del group, next_group, matched, shifted, date_diff, valid
            gc.collect()

    if feature_prefixes is None:
        feature_prefixes = [
            "sales_day_",
            # "store_med_day_",
            # "item_med_day_",
            # "store_med_change_",
            # "item_med_change_",
            # "store_cluster_logpct_change_",
            # "item_cluster_logpct_change_",
            # "dayofweek_",
            # "weekofmonth_",
            # "monthofyear_",
            # "paycycle_",
            # "season_",
        ]

    # Set up logging
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.info(f"Window size: {window_size}")
    logger.info(f"Input shape: {df.shape}")

    df = df.sort_values(["store_item", "start_date"]).reset_index(drop=True)

    df = pd.concat(
        add_y_targets_from_shift(df, window_size, feature_prefixes),
        ignore_index=True,
    )

    df.sort_values(["store_item", "start_date"]).reset_index(drop=True)

    return df


def create_sale_features(
    df,
    *,
    window_size=16,
    calendar_aligned: bool = True,
    fn: Optional[Path] = None,
    log_level: str = "INFO",
) -> pd.DataFrame:
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if "store_item" not in df.columns:
        df["store_item"] = df["store"].astype(str) + "_" + df["item"].astype(str)

    if fn is not None:
        if fn.exists():
            logger.info(f"Loading sales features from {fn}")
            if fn.suffix == ".parquet":
                df = pd.read_parquet(fn)
            else:
                df = pd.read_csv(fn)
        else:
            logger.info(f"Generating sales features to {fn}")
            df = generate_sales_features(
                df,
                window_size,
                calendar_aligned=calendar_aligned,
                log_level=log_level,
                output_path=fn,
            )
    else:
        logger.info("Generating sales features")
        df = generate_sales_features(
            df,
            window_size,
            calendar_aligned=calendar_aligned,
            log_level=log_level,
        )
    df["start_date"] = pd.to_datetime(df["start_date"])
    return df


def create_cyc_features(
    df,
    *,
    window_size=16,
    calendar_aligned: bool = True,
    fn: Optional[Path] = None,
    log_level: str = "INFO",
) -> pd.DataFrame:
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if "store_item" not in df.columns:
        df["store_item"] = df["store"].astype(str) + "_" + df["item"].astype(str)

    if fn is not None:
        if fn.exists():
            if fn.suffix == ".parquet":
                df = pd.read_parquet(fn)
            else:
                df = pd.read_csv(fn)
            logger.info(f"Loading cyclical features from {fn}")
        else:
            logger.info(f"Generating cyclical features to {fn}")
            df = generate_cyclical_features(
                df,
                window_size,
                calendar_aligned=calendar_aligned,
                log_level=log_level,
                output_path=fn,
            )
    else:
        logger.info("Generating cyclical features")
        df = generate_cyclical_features(
            df,
            window_size,
            calendar_aligned=calendar_aligned,
            log_level=log_level,
        )
    df["start_date"] = pd.to_datetime(df["start_date"])
    return df


def create_features(
    add_y_targets: bool = False,
    sales_fn: Optional[Path] = None,
    cyc_fn: Optional[Path] = None,
    window_size: int = 16,
    log_level: str = "INFO",
    output_fn: Optional[Path] = None,
) -> pd.DataFrame:
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.info("Loading sales features")
    if sales_fn.exists():
        logger.info(f"Loading sales features from {sales_fn}")
        if sales_fn.suffix == ".parquet":
            sales_df = pd.read_parquet(sales_fn)
        else:
            sales_df = pd.read_csv(sales_fn)
    else:
        logger.warning(f"Sales features not found at {sales_fn}")
        sales_df = pd.DataFrame()
    logger.info(f"sales_df.shape: {sales_df.shape}")

    logger.info("Loading cyclical features")
    if cyc_fn.exists():
        logger.info(f"Loading cyclical features from {cyc_fn}")
        if cyc_fn.suffix == ".parquet":
            cyc_df = pd.read_parquet(cyc_fn)
        else:
            cyc_df = pd.read_csv(cyc_fn)
    else:
        logger.warning(f"Cyclical features not found at {cyc_fn}")
        cyc_df = pd.DataFrame()
    logger.info(f"cyc_df.shape: {cyc_df.shape}")

    logger.info("Merging sales and cyclical features")
    df = pd.merge(
        sales_df,
        cyc_df,
        on=["start_date", "store_item"],
    )
    logger.info(f"df.shape: {df.shape}")

    if add_y_targets:
        df = create_y_targets_from_shift(
            df, window_size, feature_prefixes=["sales_day_"]
        )
        logger.info(f"df.shape: {df.shape}")

    if output_fn is not None:
        logger.info(f"Saving features to {output_fn}")
        if output_fn.suffix == ".parquet":
            df.to_parquet(output_fn)
        else:
            df.to_csv(output_fn, index=False)

    return df


def prepare_training_data_from_raw_df(
    df,
    *,
    window_size=16,
    calendar_aligned: bool = True,
    add_y_targets: bool = True,
    sales_fn: Optional[Path] = None,
    cyc_fn: Optional[Path] = None,
    log_level: str = "INFO",
) -> pd.DataFrame:
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if "store_item" not in df.columns:
        df["store_item"] = df["store"].astype(str) + "_" + df["item"].astype(str)

    sales_df = create_sale_features(
        df,
        window_size=window_size,
        calendar_aligned=calendar_aligned,
        sales_fn=sales_fn,
        log_level=log_level,
    )
    cyc_df = create_cyc_features(
        df,
        window_size=window_size,
        calendar_aligned=calendar_aligned,
        cyc_fn=cyc_fn,
        log_level=log_level,
    )
    del df
    gc.collect()

    logger.info(f"Merging sales and cyclical features")
    merged_df = pd.merge(
        sales_df,
        cyc_df,
        on=["start_date", "store_item"],
    )

    logger.info(f"merged_df.shape: {merged_df.shape}")
    if add_y_targets:
        merged_df = create_y_targets_from_shift(
            merged_df,
            window_size=window_size,
            feature_prefixes=["sales_day_"],
            log_level=log_level,
        )
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
