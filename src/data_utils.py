from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional, Iterator
from tqdm import tqdm
from pathlib import Path
import gc
from statsmodels.tsa.arima.model import ARIMA
import logging
import multiprocessing
from src.utils import save_csv_or_parquet, get_logger
from concurrent.futures import ProcessPoolExecutor

import torch

logger = get_logger(__name__)

SALE_FEATURES = [
    "store_med_day",
    "item_med_day",
    "store_med_change",
    "item_med_change",
    "sales_day",
    "store_med_logpct_change",
    "item_med_logpct_change",
]

CYCLICAL_FEATURES = [
    "dayofweek",
    "weekofmonth",
    "monthofyear",
    "paycycle",
    "season",
]
TRIGS = ["sin", "cos"]

WEIGHT_COLUMN = "weight"
UNIT_SALES = "unit_sales"
META_FEATURES = "META_FEATURES"
X_SALE_FEATURES = "X_SALE_FEATURES"
X_CYCLICAL_FEATURES = "X_CYCLICAL_FEATURES"
X_FEATURES = "X_FEATURES"
X_TO_LOG_FEATURES = "X_TO_LOG_FEATURES"
X_LOG_FEATURES = "X_LOG_FEATURES"
LABELS = "LABELS"
Y_LOG_FEATURES = "Y_LOG_FEATURES"
Y_TO_LOG_FEATURES = "Y_TO_LOG_FEATURES"
ALL_FEATURES = "ALL_FEATURES"
UNIT_SALE_IDX = "UNIT_SALE_IDX"


def polar_engine():
    return "gpu" if torch.cuda.is_available() else "rust"


def build_feature_and_label_cols(
    window_size: int,
) -> dict[str, list[str]]:
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
        f"{name}_{i}" for name in SALE_FEATURES for i in range(1, window_size + 1)
    ]

    x_to_log_features = [
        f"{name}_{i}"
        for name in SALE_FEATURES
        for i in range(1, window_size + 1)
        if name not in ["store_med_logpct_change", "item_med_logpct_change"]
    ]
    x_log_features = [
        f"{name}_{i}"
        for name in SALE_FEATURES
        for i in range(1, window_size + 1)
        if name in ["store_med_logpct_change", "item_med_logpct_change"]
    ]

    x_feature_cols = x_to_log_features + x_log_features + x_cyclical_features
    assert x_feature_cols == x_sales_features + x_cyclical_features

    y_to_log_features = [f"y_sales_day_{i}" for i in range(1, window_size + 1)]
    y_log_features = [
        f"y_store_med_logpct_change_{i}" for i in range(1, window_size + 1)
    ] + [f"y_item_med_logpct_change_{i}" for i in range(1, window_size + 1)]
    label_cols = y_to_log_features + y_log_features
    assert label_cols == y_to_log_features + y_log_features

    unit_sales = get_X_unit_sales(window_size)

    all_features = meta_cols + x_feature_cols + label_cols
    features = dict(
        META_FEATURES=meta_cols,
        UNIT_SALES=unit_sales,
        X_SALE_FEATURES=x_sales_features,
        X_CYCLICAL_FEATURES=x_cyclical_features,
        X_FEATURES=x_feature_cols,
        X_TO_LOG_FEATURES=x_to_log_features,
        X_LOG_FEATURES=x_log_features,
        LABELS=label_cols,
        Y_LOG_FEATURES=y_log_features,
        Y_TO_LOG_FEATURES=y_to_log_features,
        ALL_FEATURES=all_features,
    )
    return features


def get_X_unit_sales(window_size: int = 16) -> list[str]:
    return [f"sales_day_{i}" for i in range(1, window_size + 1)]


def get_X_feature_idx(window_size: int = 1) -> dict[str, list[int]]:
    features = build_feature_and_label_cols(window_size)
    col_x_index_map = {col: idx for idx, col in enumerate(features[X_FEATURES])}
    x_to_log_idx = [col_x_index_map[c] for c in features[X_TO_LOG_FEATURES]]
    x_log_idx = [col_x_index_map[c] for c in features[X_LOG_FEATURES]]
    x_cyc_idx = [col_x_index_map[c] for c in features[X_CYCLICAL_FEATURES]]
    idx_features = dict(
        UNIT_SALE_IDX=[x_to_log_idx[0]],
        X_FEATURES=list(range(len(features[X_FEATURES]))),
        X_TO_LOG_FEATURES=x_to_log_idx,
        X_LOG_FEATURES=x_log_idx,
        X_CYCLICAL_FEATURES=x_cyc_idx,
    )
    return idx_features


def get_y_idx(window_size: int = 1) -> dict[str, list[int]]:
    features = build_feature_and_label_cols(window_size)
    col_y_index_map = {col: idx for idx, col in enumerate(features[LABELS])}
    y_to_log_idx = [col_y_index_map[c] for c in features[Y_TO_LOG_FEATURES]]
    y_log_idx = [col_y_index_map[c] for c in features[Y_LOG_FEATURES]]
    idx_features = dict(
        LABELS=list(range(len(features[LABELS]))),
        Y_TO_LOG_FEATURES=y_to_log_idx,
        Y_LOG_FEATURES=y_log_idx,
    )
    return idx_features


def sort_df(
    df: pd.DataFrame, window_size: int = 1, flag_duplicates: bool = True
) -> pd.DataFrame:
    # --- Assert uniqueness of rows ---
    if flag_duplicates:
        if df.duplicated(subset=["start_date", "store_item"]).any():
            dups = df[df.duplicated(subset=["start_date", "store_item"], keep=False)]
            raise ValueError(
                f"Duplicate rows detected for date/store_item:\n{dups[['start_date', 'store_item']]}"
            )
    df = df.sort_values(["store_item", "start_date"], inplace=False).reset_index(
        drop=True
    )
    features = build_feature_and_label_cols(window_size)
    df = df[features[ALL_FEATURES]]
    return df


def load_raw_data(data_fn: Path) -> pd.DataFrame:
    """Load and preprocess training data.

    Args:
        data_fn: Path to the training data file

    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"Loading data from {data_fn}")

    try:
        if data_fn.suffix == ".parquet":
            df = pd.read_parquet(data_fn)
        else:
            dtype_dict = {
                "store": "uint16",
                "item": "uint32",
                "store_item": "string",  # allow NaNs as <NA>
                "unit_sales": "float32",
                # "id": "Int64",  # nullable integer
                "onpromotion": "boolean",  # if you want True/False with nulls
            }
            df = pd.read_csv(
                data_fn,
                dtype=dtype_dict,
                parse_dates=["date"],
                keep_default_na=True,
                na_values=[""],
            )
        # Convert nullable Int64 or boolean to float64 with NaN
        cols = [
            "date",
            "store_item",
            "store",
            "item",
            "unit_sales",
            "onpromotion",
            "weight",
        ] + [
            c
            for c in df.columns
            if c
            not in (
                "date",
                "store_item",
                "store",
                "item",
                "unit_sales",
                "onpromotion",
                "weight",
            )
        ]
        df = df[cols]
        # df["id"] = df["id"].astype("float64")  # <NA> → np.nan
        # df["id"] = df["id"].astype(object).where(df["id"].notna(), np.nan)
        df["store_item"] = (
            df["store_item"].astype(object).where(df["store_item"].notna(), np.nan)
        )
        df["onpromotion"] = (
            df["onpromotion"].astype(object).where(df["onpromotion"].notna(), np.nan)
        )
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values(["store_item", "date"], inplace=True)
        logger.info(f"Loaded data with shape {df.shape}")
        df.fillna(0, inplace=True)
        logger.info(f"Filled NaN values with 0")
        # df = df[df["unit_sales"].notna()]
        # logger.info(f"Dropped rows with NaN unit_sales, new shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")


def load_full_data(
    data_fn: Path,
    window_size: int,
    *,
    output_fn: Path | None = None,
    log_level: str = "INFO",
) -> pd.DataFrame:
    """Load and preprocess training data.

    Args:
        data_fn: Path to the training data file

    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"Loading data from {data_fn}")
    try:
        if data_fn.suffix == ".parquet":
            df = pd.read_parquet(data_fn)
        else:
            dtype_dict = {
                "start_date": np.uint32,
                "store_item": str,
                "store_cluster": np.uint8,
                "item_cluster": np.uint8,
                "unit_sales": np.float32,
                "weight": np.float32,
                **{f"store_med_day_{i}": np.float32 for i in range(1, window_size + 1)},
                **{f"item_med_day_{i}": np.float32 for i in range(1, window_size + 1)},
                **{
                    f"store_med_change_{i}": np.float32
                    for i in range(1, window_size + 1)
                },
                **{
                    f"item_med_change_{i}": np.float32
                    for i in range(1, window_size + 1)
                },
                **{
                    f"store_med_logpct_change_{i}": np.float32
                    for i in range(1, window_size + 1)
                },
                **{
                    f"item_med_logpct_change_{i}": np.float32
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
        df = sort_df(df, window_size=window_size)
        if output_fn:
            logger.info(f"Saving final_df to {output_fn}")
            if output_fn.suffix == ".parquet":
                df.to_parquet(output_fn)
            else:
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
    logger.info(f"Loading data from {data_fn}")
    try:
        if data_fn.suffix == ".parquet":
            df = pd.read_parquet(data_fn)
        else:
            dtype_dict = {
                "start_date": np.uint32,
                "store_item": str,
                "store_cluster": np.uint8,
                "item_cluster": np.uint8,
                **{f"store_med_day_{i}": np.float32 for i in range(1, window_size + 1)},
                **{f"item_med_day_{i}": np.float32 for i in range(1, window_size + 1)},
                **{
                    f"store_med_change_{i}": np.float32
                    for i in range(1, window_size + 1)
                },
                **{
                    f"item_med_change_{i}": np.float32
                    for i in range(1, window_size + 1)
                },
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
        df = sort_df(df, window_size=window_size)
        logger.info(f"Loaded data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def compute_cluster_medians(
    df: pd.DataFrame,
    store_fn: Path | None = None,
    item_fn: Path | None = None,
    *,
    value_col: str = "unit_sales",
    log_level: str = "INFO",
):
    store_med = (
        df.groupby(["store_cluster", "date"], observed=True)[value_col]
        .median()
        .rename("store_cluster_median")
        .reset_index()
    )

    item_med = (
        df.groupby(["item_cluster", "date"], observed=True)[value_col]
        .median()
        .rename("item_cluster_median")
        .reset_index()
    )

    if store_fn is not None:
        logger.info(f"Saving store_med to {store_fn}")
        store_med.to_parquet(store_fn)
    if item_fn is not None:
        logger.info(f"Saving item_med to {item_fn}")
        item_med.to_parquet(item_fn)

    return store_med, item_med


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
                "day": np.ones_like(years, dtype="uint8"),
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
    spring_equinox = pd.to_datetime(
        {
            "year": years,
            "month": np.full_like(years, 3, dtype="uint8"),
            "day": np.full_like(years, 20, dtype="uint8"),
        }
    )
    adjusted_years = np.where(dates < spring_equinox, years - 1, years)
    adjusted_equinox = pd.to_datetime(
        {
            "year": adjusted_years,
            "month": np.full_like(adjusted_years, 3, dtype="uint8"),
            "day": np.full_like(adjusted_years, 20, dtype="uint8"),
        }
    )
    days_since = (dates - adjusted_equinox).dt.days.astype("float32")
    season_ratio = ((days_since % 365) / 365).astype("float32")
    df["season_sin"] = np.sin(2 * np.pi * season_ratio).astype("float32")
    df["season_cos"] = np.cos(2 * np.pi * season_ratio).astype("float32")

    # ───────────────── build window rows ─────────────────
    results: List[dict] = []
    iterator = df.groupby("store_item")
    if logger.level == logging.DEBUG:

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


def add_rolling_sales_summaries(
    df: pd.DataFrame,
    windows=(3, 7, 14, 30, 60, 140),
    decay: float = 0.9,
    log_level: str = "INFO",
) -> pd.DataFrame:
    """
    Add rolling stats and fast leakage-safe ARIMA(0,0,1) forecasts using recursive update.
    """

    need = {"store_item", "date", "unit_sales"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    out = df.sort_values(["store_item", "date"]).copy()
    g = out.groupby("store_item", group_keys=False)

    # --- Past-only series ---
    past = g["unit_sales"].shift(1, fill_value=0)

    # --- Rolling features ---
    for i in windows:
        logger.info(f"Adding unit_sales rolling features (window={i})")
        roll = past.rolling(i, min_periods=i)

        out[f"mean_{i}"] = roll.mean()
        out[f"median_{i}"] = roll.median()
        out[f"min_{i}"] = roll.min()
        out[f"max_{i}"] = roll.max()
        out[f"std_{i}"] = roll.std(ddof=1)

        w = (decay ** np.arange(i)[::-1]).astype(np.float64)
        out[f"mean_{i}_decay"] = past.shift(1, fill_value=0).transform(
            lambda s: s.rolling(i, min_periods=i).apply(
                lambda x: np.dot(x, w), raw=True
            )
        )

        out[f"diff_{i}_mean"] = (past - past.shift(i - 1)) / (i - 1) if i > 1 else 0.0

    # --- Fast ARIMA(0,0,1) forecasts ---
    def fast_arima001(series: pd.Series) -> pd.Series:
        """
        Fit ARIMA(0,0,1) once, then generate one-step-ahead forecasts
        recursively with only past residuals (leakage-safe).
        """
        series = series.astype(float)
        n = len(series)
        forecasts = [np.nan] * n

        # Need at least 3 points to fit MA(1)
        if n < 3:
            return pd.Series(forecasts, index=series.index)

        # Fit on the whole series once to get params
        try:
            model = ARIMA(series, order=(0, 0, 1))
            fitted = model.fit()
            mu = fitted.params.get("const", 0.0)
            theta = fitted.params["ma.L1"]
        except Exception as e:
            logger.debug(f"ARIMA fit failed: {e}")
            return pd.Series(forecasts, index=series.index)

        # Recursive one-step-ahead prediction
        eps_prev = 0.0
        for t in range(1, n):
            forecasts[t] = mu + theta * eps_prev
            eps_prev = series.iloc[t] - forecasts[t]  # update residual

        return pd.Series(forecasts, index=series.index)

    logger.info("Adding fast leakage-safe ARIMA(0,0,1) forecasts")
    out["arima001_forecast"] = g["unit_sales"].apply(fast_arima001)

    return out


def generate_growth_rate_features(
    df: pd.DataFrame,
    window_size: int = 1,
    *,
    calendar_aligned: bool = True,
    output_dir: Optional[Path] = None,
    output_fn: Optional[Path] = None,
    weight_col: str = "weight",
    promo_col: str = "onpromotion",
    n_jobs: int = -1,  # NEW: Number of parallel processes
    batch_size: int = 100,  # NEW: Batch size for multiprocessing
) -> pd.DataFrame:
    """
    Generate growth rate features for all store-item combinations.

    This function processes each store-item combination separately and either:
    1. Saves individual parquet files if output_dir is provided
    2. Returns a combined DataFrame with all growth rate features

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns: store, item, date, unit_sales
    window_size : int, default=1
        Size of rolling window for features
    calendar_aligned : bool, default=True
        Whether to align features to calendar dates
    output_dir : Optional[Path], default=None
        Directory to save individual parquet files. If None, returns combined DataFrame
    weight_col : str, default="weight"
        Name of weight column
    promo_col : str, default="onpromotion"
        Name of promotion column
    log_level : str, default="INFO"
        Logging level
    n_jobs : int, default=1
        Number of parallel processes. Use -1 for all CPU cores, 1 for single-threaded
    batch_size : int, default=100
        Number of store-item combinations per batch for multiprocessing

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with growth rate features for all store-item combinations
    """

    # Create output directory if specified
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get unique store-item combinations
    grouped = df[["store", "item"]].drop_duplicates()
    total_combinations = len(grouped)
    logger.info(f"Total store-item combinations: {total_combinations}")

    # Determine number of processes
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    elif n_jobs < 1:
        n_jobs = 1

    logger.info(f"Using {n_jobs} processes for parallel processing")

    # Single-threaded processing (original logic)
    if n_jobs == 1:
        return _generate_growth_rate_features_sequential(
            df,
            grouped,
            total_combinations,
            window_size,
            calendar_aligned,
            output_dir,
            weight_col,
            promo_col,
        )

    # Multi-threaded processing
    else:
        return _generate_growth_rate_features_parallel(
            df,
            grouped,
            total_combinations,
            window_size,
            calendar_aligned,
            output_dir,
            output_fn,
            weight_col,
            promo_col,
            n_jobs,
            batch_size,
        )


def _generate_growth_rate_features_sequential(
    df: pd.DataFrame,
    grouped: pd.DataFrame,
    total_combinations: int,
    window_size: int,
    calendar_aligned: bool,
    output_dir: Optional[Path],
    weight_col: str,
    promo_col: str,
) -> pd.DataFrame:
    """Sequential processing (original logic)."""

    # Prepare iterator with optional progress bar
    iterator = grouped.iterrows()
    if logger.level == logging.DEBUG:
        iterator = tqdm(
            iterator, desc="Generating growth rate features", total=total_combinations
        )

    # Collect results if not saving to files
    results = []

    # Work with a copy to avoid modifying the original
    # df_working = df.copy()
    df_working = df

    for idx, row in iterator:
        store, sku = row["store"], row["item"]
        logger.debug(f"Generating growth features for store: {store}, item: {sku}")

        # Extract data for this store-item combination
        mask = (df_working["store"] == store) & (df_working["item"] == sku)
        store_sku_df = df_working[mask].copy()

        if len(store_sku_df) == 0:
            logger.warning(f"No data found for store {store}, item {sku}")
            continue

        features_df = generate_growth_rate_store_sku_feature(
            store_sku_df,
            window_size=window_size,
            calendar_aligned=calendar_aligned,
            # output_path=None,  # Do not save to files
            output_path=(
                output_dir / f"growth_rate_{store}_{sku}.parquet"
                if output_dir
                else None
            ),
            weight_col=weight_col,
            promo_col=promo_col,
        )

        # Collect results for return (regardless of whether we're also saving to files)
        if features_df is not None and not features_df.empty:
            results.append(features_df)
        elif features_df is None:
            logger.warning(f"No features generated for store {store}, item {sku}")
        elif features_df.empty:
            logger.warning(f"Empty features DataFrame for store {store}, item {sku}")

        # Remove processed data to free memory
        df_working = df_working[~mask]

        # Optional: Force garbage collection periodically
        if (idx + 1) % 100 == 0:  # Every 100 store-item pairs
            gc.collect()
            logger.debug(
                f"Processed {idx + 1}/{total_combinations} combinations, freed memory"
            )

    logger.info(f"Processed {total_combinations} store-item combinations")

    # Return combined results or empty DataFrame
    if results:
        combined_df = pd.concat(results, ignore_index=True)
        logger.info(
            f"Generated growth rate features for {len(results)} store-item combinations"
        )
        logger.info(f"Combined DataFrame shape: {combined_df.shape}")
        return combined_df
    else:
        logger.warning("No growth rate features generated")
        return pd.DataFrame()


def _process_store_item_batch(args):
    """Process a batch of store-item combinations in parallel."""
    (
        df_subset,
        store_item_pairs,
        window_size,
        calendar_aligned,
        output_dir,
        weight_col,
        promo_col,
    ) = args

    results = []

    for store, item in store_item_pairs:
        # Extract data for this store-item combination
        mask = (df_subset["store"] == store) & (df_subset["item"] == item)
        store_sku_df = df_subset[mask].copy()

        if len(store_sku_df) == 0:
            continue

        features_df = generate_growth_rate_store_sku_feature(
            store_sku_df,
            window_size=window_size,
            calendar_aligned=calendar_aligned,
            output_path=None,  # Do not save to files
            # output_path=(
            #     output_dir / f"growth_rate_{store}_{item}.parquet"
            #     if output_dir
            #     else None
            # ),
            weight_col=weight_col,
            promo_col=promo_col,
        )

        # Collect results for return (regardless of whether we're also saving to files)
        if features_df is not None and not features_df.empty:
            results.append(features_df)

    return results


def _generate_growth_rate_features_parallel(
    df: pd.DataFrame,
    grouped: pd.DataFrame,
    total_combinations: int,
    window_size: int,
    calendar_aligned: bool,
    output_dir: Optional[Path],
    output_fn: Optional[Path],
    weight_col: str,
    promo_col: str,
    n_jobs: int,
    batch_size: int,
) -> pd.DataFrame:
    """Parallel processing using multiprocessing."""

    logger.info(f"Starting parallel processing with {n_jobs} processes")

    # Create batches of store-item combinations
    store_item_pairs = [(row["store"], row["item"]) for _, row in grouped.iterrows()]
    batches = [
        store_item_pairs[i : i + batch_size]
        for i in range(0, len(store_item_pairs), batch_size)
    ]

    logger.info(
        f"Created {len(batches)} batches of ~{batch_size} store-item pairs each"
    )

    # Prepare arguments for each batch
    batch_args = [
        (
            df,  # Each process gets the full dataset
            batch,
            window_size,
            calendar_aligned,
            output_dir,
            weight_col,
            promo_col,
        )
        for batch in batches
    ]

    # Process batches in parallel
    all_results = []

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Use tqdm for progress tracking
        if logger.level <= logging.INFO:
            from tqdm import tqdm

            batch_results = list(
                tqdm(
                    executor.map(_process_store_item_batch, batch_args),
                    total=len(batches),
                    desc="Processing batches",
                )
            )
        else:
            batch_results = list(executor.map(_process_store_item_batch, batch_args))

    # Flatten results from all batches
    for batch_result in batch_results:
        all_results.extend(batch_result)

    logger.info(
        f"Parallel processing completed. Processed {total_combinations} combinations"
    )

    # Return combined results or empty DataFrame
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        logger.info(
            f"Generated growth rate features for {len(all_results)} store-item combinations"
        )
        logger.info(f"Combined DataFrame shape: {combined_df.shape}")
        if output_fn is not None:
            logger.info(f"Saving combined growth rate features to {output_fn}")
            save_csv_or_parquet(combined_df, output_fn)

        return combined_df
    else:
        logger.warning("No growth rate features generated")
        return pd.DataFrame()


def generate_growth_rate_store_sku_feature(
    df: pd.DataFrame,
    window_size: int = 1,
    *,
    calendar_aligned: bool = True,
    log_level: str = "INFO",
    output_path: Optional[Path] = None,
    weight_col: str = "weight",  # <- keep this
    promo_col: str = "onpromotion",  # <- daily flags kept as *_day_i
    batch_size: int = 100,
) -> pd.DataFrame:
    """
    For each rolling window, build one row per (store,item) with:
      sales_day_1..window_size,
      growth_rate_1..window_size  where growth_rate_i = (sales_i - sales_{i-1})/sales_{i-1},
      weight (assumed constant per store-item),
      onpromotion_day_1..window_size (0/1, aligned to the window dates).
    """
    logger.debug(f"Total rows: {len(df)}")

    # MEMORY OPT 1: Use efficient data types from the start
    df_optimized = df

    # Convert to smaller data types where possible
    dtype_conversions = {
        "store": "int32",
        "item": "int32",
        "unit_sales": "float32",
        promo_col: "int8",
    }

    for col, dtype in dtype_conversions.items():
        if col in df_optimized.columns:
            df_optimized[col] = df_optimized[col].astype(dtype)

    # Pre-compute weights with efficient data types
    if weight_col in df_optimized.columns:
        w_src = df_optimized[["store", "item", weight_col]].dropna(subset=[weight_col])
        if not w_src.empty:
            weight_map = (
                w_src.groupby(["store", "item"], sort=False)[weight_col]
                .first()
                .astype("float32")
            )
        else:
            weight_map = pd.Series(dtype="float32")
    else:
        weight_map = pd.Series(dtype="float32")

    windows = generate_aligned_windows(
        df_optimized, window_size, calendar_aligned=calendar_aligned
    )

    # MEMORY OPT 2: Stream processing with batches
    all_records = []

    for batch_start in range(0, len(windows), batch_size):
        batch_end = min(batch_start + batch_size, len(windows))
        window_batch = windows[batch_start:batch_end]

        batch_records = []

        for window_dates in window_batch:
            window_idx = pd.to_datetime(pd.Index(window_dates)).sort_values()
            w_df = df_optimized[df_optimized["date"].isin(window_idx)].copy()

            # Create pivot tables with efficient data types
            sales_wide = w_df.pivot_table(
                index=["store", "item"],
                columns="date",
                values="unit_sales",
                aggfunc="sum",
                fill_value=0.0,
            ).astype(
                "float32"
            )  # MEMORY OPT: Use float32

            promo_wide = None
            if promo_col in w_df.columns:
                promo_wide = w_df.pivot_table(
                    index=["store", "item"],
                    columns="date",
                    values=promo_col,
                    aggfunc="max",
                    fill_value=0,
                ).astype(
                    "int8"
                )  # MEMORY OPT: Use int8 for binary data

            if sales_wide.empty:
                logger.warning(f"Empty sales data for window {window_idx}")
                continue

            # SPEED OPT 1: Convert to numpy arrays for fast access
            sales_values = sales_wide.reindex(
                columns=window_idx, fill_value=0.0
            ).values.astype("float32")

            if promo_wide is not None:
                promo_values = promo_wide.reindex(
                    columns=window_idx, fill_value=0
                ).values.astype("int8")
            else:
                promo_values = np.zeros(sales_values.shape, dtype="int8")

            store_items = sales_wide.index.tolist()

            # SPEED OPT 2: Pre-compute previous day data (vectorized lookup)
            prev_day = window_idx[0] - pd.DateOffset(days=1)
            prev_day_data = df_optimized[df_optimized["date"] == prev_day].set_index(
                ["store", "item"]
            )["unit_sales"]

            # SPEED OPT 3: Vectorized processing of all store-items
            for idx, (store, item) in enumerate(store_items):
                # Fast array slicing instead of pandas indexing
                sales_array = sales_values[idx, :window_size]
                promo_array = promo_values[idx, :window_size]

                # MEMORY OPT 3: Use efficient data types in record
                record = {
                    "start_date": window_idx[0],
                    "store_item": f"{store}_{item}",
                    "store": int(store),  # Ensure int32
                    "item": int(item),  # Ensure int32
                    weight_col: (
                        float(weight_map.get((store, item), np.nan))
                        if isinstance(weight_map.index, pd.MultiIndex)
                        else np.nan
                    ),
                }

                # SPEED OPT 4: Vectorized daily feature creation
                for i in range(window_size):
                    day_num = i + 1
                    curr_sales = float(sales_array[i])

                    record[f"sales_day_{day_num}"] = curr_sales
                    record[f"{promo_col}_day_{day_num}"] = int(promo_array[i])

                    # SPEED OPT 5: Optimized growth rate calculation
                    if i == 0:
                        # Fast previous day lookup using pre-computed data
                        prev_sales = prev_day_data.get((store, item), np.nan)
                        prev_sales = (
                            float(prev_sales) if not pd.isna(prev_sales) else np.nan
                        )
                    else:
                        prev_sales = float(sales_array[i - 1])

                    # Vectorized growth rate calculation
                    if pd.isna(curr_sales) or pd.isna(prev_sales) or prev_sales == 0:
                        growth_rate = np.nan
                    else:
                        growth_rate = (curr_sales - prev_sales) / prev_sales * 100.0

                    record[f"growth_rate_{day_num}"] = growth_rate

                batch_records.append(record)

            # MEMORY OPT 4: Explicit cleanup after each window
            del sales_wide, promo_wide, sales_values, promo_values, w_df

        # Add batch to results
        all_records.extend(batch_records)

        # MEMORY OPT 5: Periodic garbage collection
        if batch_start % (batch_size * 5) == 0:  # Every 5 batches
            import gc

            gc.collect()

    # MEMORY OPT 6: Efficient final DataFrame construction
    if not all_records:
        base_cols = ["start_date", "store_item", "store", "item", weight_col]
        sales_cols = [f"sales_day_{i}" for i in range(1, window_size + 1)]
        growth_cols = [f"growth_rate_{i}" for i in range(1, window_size + 1)]
        promo_cols = [f"{promo_col}_day_{i}" for i in range(1, window_size + 1)]
        return pd.DataFrame(columns=base_cols + sales_cols + growth_cols + promo_cols)

    # SPEED OPT 6: Fast DataFrame construction from records
    result_df = pd.DataFrame(all_records)
    logger.debug(f"Result DataFrame: {result_df.head()}")

    # MEMORY OPT 7: Optimize final DataFrame data types
    dtype_map = {}
    for col in result_df.columns:
        if col in ["store", "item"]:
            dtype_map[col] = "int32"
        elif (
            col.startswith("sales_day_")
            or col.startswith("growth_rate_")
            or col == weight_col
        ):
            dtype_map[col] = "float32"
        elif col.startswith(f"{promo_col}_day_"):
            dtype_map[col] = "int8"

    # Apply data type optimizations in batch
    for col, dtype in dtype_map.items():
        if col in result_df.columns:
            result_df[col] = result_df[col].astype(dtype)

    if output_path is not None:
        logger.info(f"Saving growth rate features to {output_path}")
        save_csv_or_parquet(result_df, output_path)

    return result_df


def generate_sales_features(
    df: pd.DataFrame,
    window_size: int = 1,
    *,
    calendar_aligned: bool = True,
    log_level: str = "INFO",
    output_path: Optional[Path] = None,
    epsilon: float = 1e-3,
) -> pd.DataFrame:

    df["date"] = pd.to_datetime(df["date"])
    assert "store_cluster" in df.columns, "Missing 'store_cluster' column"
    assert "item_cluster" in df.columns, "Missing 'item_cluster' column"
    assert "store_cluster_median" in df.columns, "Missing 'store_cluster_median' column"
    assert "item_cluster_median" in df.columns, "Missing 'item_cluster_median' column"
    assert (
        df[["store", "item", "date"]].duplicated().sum() == 0
    ), "Duplicate store-item-date rows"

    logger.info(f"Total rows: {len(df)}")

    store_item_to_clusters = (
        df.drop_duplicates(["store", "item"])[
            ["store", "item", "store_cluster", "item_cluster"]
        ]
        .set_index(["store", "item"])
        .to_dict("index")
    )

    logger.debug(f"Unique store_clusters: {df['store_cluster'].unique()}")
    logger.debug(f"Unique item_clusters: {df['item_cluster'].unique()}")
    logger.debug(
        f"Store vs. Item cluster crosstab:\n{pd.crosstab(df['store_cluster'], df['item_cluster'])}"
    )

    logger.debug("Generating rolling windows")
    windows = generate_aligned_windows(
        df, window_size, calendar_aligned=calendar_aligned
    )
    records: List[dict] = []

    for window_dates in windows:
        logger.info(f"Processing window: {window_dates[0]} to {window_dates[-1]}")
        w_df = df[df["date"].isin(window_dates)].copy()

        sales = w_df.pivot_table(
            index=["store", "item"],
            columns="date",
            values="unit_sales",
            aggfunc="sum",
            fill_value=0,
        )

        store_meds = w_df.pivot_table(
            index=["store", "item"],
            columns="date",
            values="store_cluster_median",
            aggfunc="first",
            fill_value=np.nan,
        )

        item_meds = w_df.pivot_table(
            index=["store", "item"],
            columns="date",
            values="item_cluster_median",
            aggfunc="first",
            fill_value=np.nan,
        )

        iterator = sales.iterrows()
        if logger.level == logging.DEBUG:
            iterator = tqdm(
                iterator,
                total=sales.shape[0],
                desc=f"Window {window_dates[0].strftime('%Y-%m-%d')}",
            )

        for (store, item), sales_vals in iterator:
            cluster_info = store_item_to_clusters.get((store, item), {})
            s_cl = cluster_info.get("store_cluster", "ALL_STORES")
            i_cl = cluster_info.get("item_cluster", "ALL_ITEMS")

            row = {
                "start_date": window_dates[0],
                "store_item": f"{store}_{item}",
                "store": store,
                "item": item,
                "store_cluster": s_cl,
                "item_cluster": i_cl,
            }

            if "weight" in df.columns:
                try:
                    weight_val = (
                        df.loc[(df["store"] == store) & (df["item"] == item), "weight"]
                        .dropna()
                        .iloc[0]
                    )
                except IndexError:
                    weight_val = np.nan
                row["weight"] = weight_val

            for i in range(1, window_size + 1):
                d = window_dates[i - 1] if i - 1 < len(window_dates) else None
                if d is None:
                    continue

                sales_val = sales_vals.get(d, 0)
                store_med_val = store_meds.loc[(store, item)].get(d, np.nan)
                # logger.debug(f"Store med: {store_med_val}")
                item_med_val = item_meds.loc[(store, item)].get(d, np.nan)
                # logger.debug(f"Item med: {item_med_val}")

                row[f"sales_day_{i}"] = sales_val
                row[f"store_med_day_{i}"] = store_med_val
                row[f"item_med_day_{i}"] = item_med_val

                smv = (
                    store_med_val
                    if pd.notna(store_med_val) and store_med_val > 0
                    else epsilon
                )
                imv = (
                    item_med_val
                    if pd.notna(item_med_val) and item_med_val > 0
                    else epsilon
                )

                row[f"store_med_change_{i}"] = sales_val / smv
                row[f"item_med_change_{i}"] = sales_val / imv
                row[f"store_med_logpct_change_{i}"] = np.log(
                    max(sales_val / smv, epsilon)
                )
                row[f"item_med_logpct_change_{i}"] = np.log(
                    max(sales_val / imv, epsilon)
                )

            records.append(row)

        del sales, store_meds, item_meds

    cols = [
        "start_date",
        "store_item",
        "store",
        "item",
        "store_cluster",
        "item_cluster",
        "weight",
    ]
    cols += [
        f"{prefix}{i}"
        for prefix in (
            "store_med_day_",
            "item_med_day_",
            "store_med_change_",
            "item_med_change_",
            "store_med_logpct_change_",
            "item_med_logpct_change_",
            "sales_day_",
        )
        for i in range(1, window_size + 1)
    ]

    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame(columns=cols)
    else:
        df = df[cols]

    if output_path is not None:
        logger.info(f"Saving sales features to {output_path}")
        if output_path.suffix == ".parquet":
            df.to_parquet(output_path)
        else:
            df.to_csv(output_path, index=False)

        debug_fn = output_path.with_name("debug_subset.csv")
        df.head(50).to_csv(debug_fn, index=False)
        logger.info(f"Saved debug sample to {debug_fn}")

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
    logger.info(f"Window size: {window_size}")
    logger.info(f"Input shape: {df.shape}")

    df = df.sort_values(["store_item", "start_date"]).reset_index(drop=True)

    logger.info(f"Feature prefixes: {feature_prefixes}")
    y_target_dfs = list(add_y_targets_from_shift(df, window_size, feature_prefixes))
    if not y_target_dfs:
        logger.warning(
            "No valid y targets were generated. Check your input data and window size."
        )
        return pd.DataFrame()
    df = pd.concat(y_target_dfs, ignore_index=True)
    df = df.sort_values(["store_item", "start_date"]).reset_index(drop=True)

    return df


def create_sale_features(
    df,
    *,
    window_size=16,
    calendar_aligned: bool = True,
    fn: Optional[Path] = None,
    log_level: str = "INFO",
) -> pd.DataFrame:

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
        (
            meta_cols,
            _,
            _,
            x_feature_cols,
            _,
            _,
            label_cols,
            _,
            _,
            _,
        ) = build_feature_and_label_cols(window_size=window_size)
        df = df[meta_cols + x_feature_cols + label_cols]
    else:
        logger.info("Not adding y targets")

        (
            meta_cols,
            _,
            _,
            x_feature_cols,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = build_feature_and_label_cols(window_size=window_size)
        df = df[meta_cols + x_feature_cols]
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
    df: pd.DataFrame, axis: int = 0, nan_policy: str = "omit", epsilon: float = 1e-8
) -> pd.DataFrame:
    """
    Compute z-score along rows (axis=1) or columns (axis=0), optionally ignoring NaNs.
    Avoids division by zero by adding epsilon to zero std values.
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

    std_safe = np.where(std == 0, epsilon, std)
    z = (values - mean) / std_safe
    return pd.DataFrame(z, index=df.index, columns=df.columns)


def median_mean_transform(
    df: pd.DataFrame,
    *,
    # freq="W",
    column_name="unit_sales",
    median_transform=True,
    mean_transform=False,
) -> pd.DataFrame:
    if median_transform:
        # df = (
        #     df.groupby([pd.Grouper(key="date", freq=freq), "store", "item"])[
        #         "unit_sales"
        #     ]
        #     .median()
        #     .reset_index()
        # )
        df = df.groupby(["store", "item"])[column_name].median().unstack(fill_value=0)
    elif mean_transform:
        # df = (
        #     df.groupby([pd.Grouper(key="date", freq=freq), "store", "item"])[
        #         "unit_sales"
        #     ]
        #     .mean()
        #     .reset_index()
        # )
        df = df.groupby(["store", "item"])[column_name].mean().unstack(fill_value=0)
    else:
        raise ValueError("Set either median_transform or mean_transform to True.")
    return df


def normalize_data(
    df: pd.DataFrame,
    # freq="W",
    *,
    column_name="unit_sales",
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
    # median_transform : bool
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
    df = median_mean_transform(
        df,
        column_name=column_name,
        median_transform=median_transform,
        mean_transform=mean_transform,
    )

    if log_transform:
        df = df.apply(np.log1p)

    if zscore_rows:
        df = zscore_with_axis(df, axis=1)

    if zscore_cols:
        df = zscore_with_axis(df, axis=0)

    return df


def normalize_store_item_data(
    df: pd.DataFrame,
    median_transform=True,
    mean_transform=False,
    log_transform=True,
    zscore_rows=True,
    zscore_cols=False,  # safer default to avoid double scaling
    normalized_column_name="normalized_unit_sales",
):
    """
    Returns the original dataframe with a new column containing normalized data.

    Parameters
    ----------
    df : pd.DataFrame
        Must include ['date', 'store', 'item', 'unit_sales']
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
    normalized_column_name : str
        Name for the new column containing normalized data

    Returns
    -------
    pd.DataFrame
        Original dataframe with a new column containing normalized data
    """
    # Keep a copy of the original dataframe
    original_df = df.copy()

    # Create the normalized matrix using the existing normalize_data function
    norm_matrix = normalize_data(
        df,
        median_transform=median_transform,
        mean_transform=mean_transform,
        log_transform=log_transform,
        zscore_rows=zscore_rows,
        zscore_cols=zscore_cols,
    )

    # Create store_item column if it doesn't exist
    if "store_item" not in original_df.columns:
        original_df["store_item"] = (
            original_df["store"].astype(str) + "_" + original_df["item"].astype(str)
        )

    # Convert the normalized matrix back to long format and merge with original data
    # The norm_matrix has stores as index and items as columns
    norm_long = norm_matrix.stack().reset_index()
    norm_long.columns = ["store", "item", normalized_column_name]
    norm_long["store_item"] = (
        norm_long["store"].astype(str) + "_" + norm_long["item"].astype(str)
    )

    # Merge the normalized data back to the original dataframe
    result_df = original_df.merge(
        norm_long[["store_item", normalized_column_name]], on="store_item", how="left"
    )

    return result_df


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


def save_parquets_by_cluster_pairs(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    to_parquet: bool = True,
    to_csv: bool = False,
    log_level: str = "INFO",
) -> None:
    """
    Splits the dataframe by (store_cluster, item_cluster) pairs and saves each to a compressed Parquet file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'store_cluster' and 'item_cluster' columns.
    output_dir : Path
        Directory where the Parquet files will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = df.groupby(["store_cluster", "item_cluster"])

    iterator = grouped
    if logger.level == logging.DEBUG:
        iterator = tqdm(iterator, desc="Generating cluster parquets")

    for (store_cluster, item_cluster), group in iterator:
        logger.info(f"Saving cluster {store_cluster}_{item_cluster}")
        if to_parquet:
            filename = f"cluster_{store_cluster}_{item_cluster}.parquet"
            group.to_parquet(
                output_dir / filename,
                index=False,
                compression="snappy",
                engine="pyarrow",
            )
        if to_csv:
            filename = f"cluster_{store_cluster}_{item_cluster}.csv"
            group.to_csv(output_dir / filename, index=False)
        del group


def mav(series: pd.Series, is_log1p: bool = True, include_zeros: bool = True):
    if is_log1p:
        vals = np.expm1(series)  # undo log1p
    else:
        vals = series
    mask = np.isfinite(vals) & (vals >= 0 if include_zeros else vals > 0)
    return np.abs(vals[mask]).mean() if mask.any() else 0.0


def add_mav_column(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    col: str,
    is_log1p: bool = True,
    include_zeros: bool = True,
) -> pd.DataFrame:

    new_col_name = f"{col1}_{col2}_mav"
    df[new_col_name] = df.groupby([col1, col2])[col].transform(
        lambda s: mav(s, is_log1p=is_log1p, include_zeros=include_zeros)
    )
    return df


def mav_by_cluster(
    df: pd.DataFrame,  # contains store, item, store_cluster, item_cluster
    matrix: pd.DataFrame,  # wide matrix: store × item
    *,
    col_mav_name: str = "store_item_mav",
    col_cluster_mav_name: str = "store_item_cluster_mav",
    is_log1p: bool = False,
    include_zeros: bool = True,
):
    """
    Compute MAVs at two levels:
      1. Per (store, item)
      2. Aggregated per (store_cluster, item_cluster)
    """

    # Convert matrix (store × item) into long form
    long_df = matrix.stack().rename("value").reset_index()
    long_df.columns = ["store", "item", "value"]

    # Attach cluster labels
    long_df = long_df.merge(
        df[["store", "item", "store_cluster", "item_cluster"]].drop_duplicates(),
        on=["store", "item"],
        how="left",
    )
    long_df = long_df.dropna(subset=["store_cluster", "item_cluster"])

    # --- 1) MAV per (store, item)
    per_store_item = (
        long_df.groupby(["store", "item"], group_keys=False)["value"]
        .apply(lambda g: mav(g, is_log1p=is_log1p, include_zeros=include_zeros))
        .reset_index(name=col_mav_name)
    )
    per_store_item = per_store_item.merge(
        long_df[["store", "item", "store_cluster", "item_cluster"]].drop_duplicates(),
        on=["store", "item"],
        how="left",
    )

    # --- 2) Aggregate per (store_cluster, item_cluster)
    per_cluster = (
        per_store_item.groupby(["store_cluster", "item_cluster"])[col_mav_name]
        .agg(["mean", "var", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": f"{col_cluster_mav_name}_mean",
                "var": f"{col_cluster_mav_name}_within_var",
                "count": "n_obs",
            }
        )
    )

    return per_store_item, per_cluster


def collapse_block_id_by_store_item(df, how="mean"):
    """
    Collapse to one row per (store,item) using mean/median for the value,
    keeping the (unique) block_id.
    """
    # sanity: block_id must be unique per pair
    bad = df.groupby(["store", "item"])["block_id"].nunique() > 1
    if bad.any():
        raise ValueError("Some (store,item) pairs have multiple block_id values.")

    agg_fn = {"mean": "mean", "median": "median"}[how]
    return df.groupby(["store", "item"], as_index=False).agg(
        growth_rate_1=("growth_rate_1", agg_fn), block_id=("block_id", "first")
    )
