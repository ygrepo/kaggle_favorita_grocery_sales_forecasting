from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional, Iterator, Literal, Union, Callable
from tqdm import tqdm
from pathlib import Path
import gc
from statsmodels.tsa.arima.model import ARIMA
import logging
import math
from src.utils import save_csv_or_parquet, polar_engine
import polars as pl
import torch

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,  # or DEBUG
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

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


def _ensure_polars(df) -> pl.DataFrame:
    """
    Normalize input to a clean Polars DataFrame:
      - pd.DataFrame  -> convert (date to pandas ts first)
      - pl.LazyFrame  -> collect
      - pl.DataFrame  -> clone
    """
    if isinstance(df, pd.DataFrame):
        pdf = df.copy()
        # keep any tz info but normalize later in Polars
        pdf["date"] = pd.to_datetime(pdf["date"])
        return pl.from_pandas(pdf)

    if isinstance(df, pl.LazyFrame):
        return df.collect()

    if isinstance(df, pl.DataFrame):
        return df.clone()

    raise TypeError(
        "df must be a pandas.DataFrame, polars.DataFrame, or polars.LazyFrame"
    )


def _ensure_lazy(df) -> pl.LazyFrame:
    """Normalize pandas / polars / lazyframe to a Polars LazyFrame."""
    if isinstance(df, pd.DataFrame):
        pdf = df.copy()
        pdf["date"] = pd.to_datetime(pdf["date"])
        return pl.from_pandas(pdf).lazy()
    if isinstance(df, pl.DataFrame):
        return df.lazy()
    if isinstance(df, pl.LazyFrame):
        return df
    raise TypeError(
        "df must be a pandas.DataFrame, polars.DataFrame, or polars.LazyFrame"
    )


def load_raw_data(data_fn: Path) -> pd.DataFrame:
    """Load and preprocess training data.

    Args:
        data_fn: Path to the training data file

    Returns:
        Preprocessed DataFrame
    """
    logger = logging.getLogger(__name__)
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


def load_raw_data_lazy(data_fn: Path) -> pl.LazyFrame:
    """
    Lazy, GPU-friendly loader for Favorita-like data.

    Produces columns:
      date: Date
      store: UInt16
      item: UInt32
      unit_sales: Float32
      onpromotion: Int8 (0/1)
      weight: Float32 (optional; filled with 0.0 if missing)
      store_item: Utf8  (e.g., "1_1047679")
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data (lazy) from {data_fn}")

    # 1) Scan lazily
    suffix = data_fn.suffix.lower()
    if suffix == ".parquet":
        lf = pl.scan_parquet(str(data_fn))
        # Parquet often already stores datetimes; cast to Date safely
        lf = lf.with_columns(pl.col("date").cast(pl.Date, strict=False))
    else:
        # CSV: let Polars try to parse datelike strings automatically
        lf = pl.scan_csv(
            str(data_fn),
            try_parse_dates=True,  # helps with common date formats
            null_values=[""],  # treat empty strings as nulls
        )
        # Ensure `date` is Date (if still Utf8, cast with permissive parsing)
        lf = lf.with_columns(pl.col("date").cast(pl.Date, strict=False))

    # # 2) Ensure required/optional columns exist
    # # (Create placeholders if absent to keep downstream stable.)
    # want_missing = []
    # for col in ("weight", "store_item", "onpromotion", "unit_sales", "store", "item"):
    #     if col not in lf.collect_schema().names():
    #         want_missing.append(col)
    # for col in want_missing:
    #     lf = lf.with_columns(pl.lit(None).alias(col))

    # # 3) Cast dtypes (permissive where appropriate)
    # lf = lf.with_columns(
    #     [
    #         pl.col("store").cast(pl.UInt16, strict=False),
    #         pl.col("item").cast(pl.UInt32, strict=False),
    #         pl.col("unit_sales").cast(pl.Float32, strict=False),
    #         pl.col("weight").cast(pl.Float32, strict=False),
    #     ]
    # )

    # # Normalize onpromotion to 0/1 Int8:
    # # - if boolean -> 1/0
    # # - if "True"/"False"/"1"/"0" strings -> cast(Boolean) then to Int8
    # # - null -> 0
    # lf = lf.with_columns(
    #     pl.when(pl.col("onpromotion").is_null())
    #     .then(0)
    #     .when(pl.col("onpromotion").cast(pl.Boolean, strict=False))
    #     .then(1)
    #     .otherwise(0)
    #     .cast(pl.Int8)
    #     .alias("onpromotion")
    # )

    # # 4) Build/clean store_item (Utf8). If missing/null/empty, compose from store/item.
    # lf = lf.with_columns(
    #     pl.when(pl.col("store_item").is_null() | (pl.col("store_item") == ""))
    #     .then(
    #         pl.col("store").cast(pl.Utf8) + pl.lit("_") + pl.col("item").cast(pl.Utf8)
    #     )
    #     .otherwise(pl.col("store_item").cast(pl.Utf8))
    #     .alias("store_item")
    # )

    # # 5) Fill numeric nulls only (avoid clobbering strings)
    # lf = lf.with_columns(
    #     [
    #         pl.col("unit_sales").fill_null(0.0),
    #         pl.col("weight").fill_null(0.0),
    #     ]
    # )

    # 6) Sort lazily (materializes on collect)
    lf = lf.sort(["store_item", "date"])

    # 7) Stable column order without triggering the .columns warning
    base_cols = [
        "date",
        "store_item",
        "store",
        "item",
        "unit_sales",
        "onpromotion",
        "weight",
    ]
    names = lf.collect_schema().names()  # cheap: schema only, no data
    other_cols = [c for c in names if c not in set(base_cols)]
    lf = lf.select(base_cols + other_cols)

    logger.info("Lazy loader prepared (no materialization yet).")
    return lf


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
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
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
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
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
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
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


def _dates_to_pandas(series_like: Iterable) -> pd.Series:
    """Robustly convert any iterable of dates/strings/py-dates to a pandas datetime Series (naive, normalized to date)."""
    s = pd.to_datetime(pd.Series(series_like), errors="coerce")
    s = s.dropna()
    # normalize to midnight to avoid time components messing with comparisons
    return s.dt.normalize()


def _extract_dates_as_pandas(df) -> pd.Series:
    """
    Return a pandas Series[datetime64[ns]] named 'date' from:
      - pd.DataFrame (expects 'date' col)
      - pl.DataFrame (expects 'date' col)
      - pl.LazyFrame (expects 'date' col; will collect only that column)
    """
    if isinstance(df, pd.DataFrame):
        if "date" not in df.columns:
            raise ValueError("Missing 'date' column.")
        return _dates_to_pandas(df["date"])

    if isinstance(df, pl.LazyFrame):
        df = df.select("date").collect(streaming=False)

    if isinstance(df, pl.DataFrame):
        if "date" not in df.columns:
            raise ValueError("Missing 'date' column.")
        s = df.get_column("date")
        # Convert polars column to python list then to pandas
        return _dates_to_pandas(s.to_list())

    raise TypeError(
        "df must be a pandas DataFrame, polars DataFrame, or polars LazyFrame."
    )


def generate_aligned_windows(
    df: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
    window_size: int,
    *,
    calendar_aligned: bool = False,
) -> List[List[pd.Timestamp]]:
    """
    Build non-overlapping date windows of length `window_size`.

    calendar_aligned=False  -> data-aligned (use only distinct dates present in df; last window may be shorter)
    calendar_aligned=True   -> calendar-aligned (exact `window_size` days; gaps filled implicitly; leftovers dropped)
    """
    if window_size <= 0:
        raise ValueError("`window_size` must be a positive integer.")

    dates = _extract_dates_as_pandas(df)
    if dates.empty:
        return []

    if calendar_aligned:
        # use full calendar days between min/max (inclusive), stepping from the back
        first_date = dates.min()
        last_date = dates.max()

        starts = []
        current_end = last_date
        full_span = pd.Timedelta(days=window_size - 1)

        while current_end - full_span >= first_date:
            start = current_end - full_span
            starts.append(start)
            # next block ends window_size days earlier
            current_end = current_end - pd.Timedelta(days=window_size)

        # oldest → newest
        starts.reverse()
        return [list(pd.date_range(s, periods=window_size, freq="D")) for s in starts]

    else:
        # use only distinct dates present in df; chunk forward
        unique_dates = sorted(pd.to_datetime(pd.unique(dates)))
        return [
            unique_dates[i : i + window_size]
            for i in range(0, len(unique_dates), window_size)
        ]


# def generate_aligned_windows(
#     df: pd.DataFrame,
#     window_size: int,
#     *,
#     calendar_aligned: bool = False,
# ) -> list[list[pd.Timestamp]]:
#     """
#     Build a series of non‑overlapping, length‑`window_size` date windows.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Must contain a ``"date"`` column convertible to ``pd.Timestamp``.
#     window_size : int
#         Number of days in each window.
#     calendar_aligned : bool, default False
#         • **False**  →  *Data‑aligned* (original behaviour):
#           Only the distinct dates that actually occur in *df* are used.
#           If the number of dates is not a multiple of `window_size`,
#           the **last window may be shorter** than `window_size`.
#         • **True**   →  *Calendar‑aligned* (back‑to‑front behaviour):
#           Anchors on the most recent calendar day in *df* and steps
#           backward in *exact* `window_size`‑day blocks, discarding any
#           leftover days that cannot form a full block.
#           Returned windows are always exactly `window_size` long and
#           include **all calendar days**, even ones missing from *df*.

#     Returns
#     -------
#     List[List[pd.Timestamp]]
#         A list of date‑lists (each list length == `window_size`
#         unless `calendar_aligned=False` and the final window is partial).
#     """
#     if window_size <= 0:
#         raise ValueError("`window_size` must be a positive integer.")

#     df["date"] = pd.to_datetime(df["date"])

#     if calendar_aligned:
#         # --- back‑to‑front, gap‑free windows ---
#         last_date = df["date"].max().normalize()  # ensure time == 00:00
#         first_date = df["date"].min().normalize()

#         starts = []
#         current = last_date
#         while current >= first_date + pd.Timedelta(days=window_size - 1):
#             # logger.debug(f"Current date: {current}")
#             starts.append(current - pd.Timedelta(days=window_size - 1))
#             current -= pd.Timedelta(days=window_size)

#         # Reverse so windows are chronological (oldest→newest)
#         return [
#             list(pd.date_range(start, periods=window_size, freq="D"))
#             for start in reversed(starts)
#         ]

#     else:
#         # --- forward, data‑aligned windows (may be shorter at the tail) ---
#         unique_dates = sorted(pd.to_datetime(df["date"].unique()))
#         # logger.debug(f"Unique dates: {unique_dates}")
#         return [
#             unique_dates[i : i + window_size]
#             for i in range(0, len(unique_dates), window_size)
#         ]


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
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

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
    log_level: str = "INFO",
    output_path: Optional[Path] = None,
    weight_col: str = "weight",  # <- keep this
    promo_col: str = "onpromotion",  # <- daily flags kept as *_day_i
) -> pd.DataFrame:
    """
    For each rolling window, build one row per (store,item) with:
      sales_day_1..window_size,
      growth_rate_1..window_size  where growth_rate_i = (sales_i - sales_{i-1})/sales_{i-1},
      weight (assumed constant per store-item),
      onpromotion_day_1..window_size (0/1, aligned to the window dates).
    """
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.info(f"Total rows: {len(df)}")

    # --- Precompute a single weight per (store,item); user says no variability
    if weight_col in df.columns:
        w_src = df[["store", "item", weight_col]].dropna(subset=[weight_col])
        if not w_src.empty:
            # pick the first value per pair (constant by assumption)
            weight_map = w_src.groupby(["store", "item"], sort=False)[
                weight_col
            ].first()
        else:
            weight_map = pd.Series(dtype=float)
    else:
        weight_map = pd.Series(dtype=float)

    windows = generate_aligned_windows(
        df, window_size, calendar_aligned=calendar_aligned
    )
    records: List[dict] = []

    for window_dates in windows:
        window_idx = pd.to_datetime(pd.Index(window_dates)).sort_values()
        logger.info(f"Processing window: {window_idx[0]} to {window_idx[-1]}")

        w_df = df[df["date"].isin(window_idx)].copy()

        # Sales (sum across potential duplicate rows)
        sales_wide = w_df.pivot_table(
            index=["store", "item"],
            columns="date",
            values="unit_sales",
            aggfunc="sum",
            fill_value=0.0,
        )

        # On-promo flags (max handles duplicates; treat missing as 0)
        promo_wide = None
        if promo_col in w_df.columns:
            promo_wide = w_df.pivot_table(
                index=["store", "item"],
                columns="date",
                values=promo_col,
                aggfunc="max",
                fill_value=0,
            )

        if sales_wide.empty:
            continue

        for (store, item), s_sales in sales_wide.iterrows():
            s_sales = s_sales.reindex(window_idx, fill_value=0.0)

            # align promo row if present
            if promo_wide is not None and (store, item) in promo_wide.index:
                s_promo = promo_wide.loc[(store, item)].reindex(
                    window_idx, fill_value=0
                )
            else:
                s_promo = pd.Series(0, index=window_idx)

            row = {
                "start_date": window_idx[0],
                "store_item": f"{store}_{item}",
                "store": store,
                "item": item,
            }

            # weight (constant per pair)
            row[weight_col] = (
                weight_map.get((store, item))
                if isinstance(weight_map.index, pd.MultiIndex)
                else np.nan
            )

            # per-day sales, promo flags, and growth
            for i, d in enumerate(window_idx[:window_size], start=1):
                curr = float(s_sales.loc[d]) if pd.notna(s_sales.loc[d]) else np.nan
                row[f"sales_day_{i}"] = curr
                row[f"{promo_col}_day_{i}"] = (
                    int(s_promo.loc[d]) if pd.notna(s_promo.loc[d]) else 0
                )

                # previous: calendar day for i==1; otherwise previous in-window
                if i == 1:
                    prev_day = pd.to_datetime(d) - pd.DateOffset(days=1)
                    prev_vals = df.loc[
                        (df["store"] == store)
                        & (df["item"] == item)
                        & (df["date"] == prev_day),
                        "unit_sales",
                    ]
                    prev = float(prev_vals.sum()) if not prev_vals.empty else np.nan
                else:
                    prev = row[f"sales_day_{i-1}"]

                # your rule: prev==0 or missing -> NaN
                row[f"growth_rate_{i}"] = (
                    np.nan
                    if (not np.isfinite(prev)) or prev == 0 or math.isnan(curr)
                    else (curr - prev) / prev * 100.0
                )

            records.append(row)

        del sales_wide, promo_wide

    # --- Column order
    base_cols = ["start_date", "store_item", "store", "item", weight_col]
    sales_cols = [f"sales_day_{i}" for i in range(1, window_size + 1)]
    growth_cols = [f"growth_rate_{i}" for i in range(1, window_size + 1)]
    promo_cols = [f"{promo_col}_day_{i}" for i in range(1, window_size + 1)]
    cols = base_cols + sales_cols + growth_cols + promo_cols

    out = pd.DataFrame.from_records(records)
    if out.empty:
        out = pd.DataFrame(columns=cols)
    else:
        # ensure all expected columns exist and are ordered
        for c in cols:
            if c not in out:
                out[c] = np.nan
        out = out[cols]

    if output_path is not None:
        logger.info(f"Saving growth rate features to {output_path}")
        save_csv_or_parquet(out, output_path)
        debug_fn = output_path.with_name("debug_subset.csv")
        out.head(50).to_csv(debug_fn, index=False)
        logger.info(f"Saved debug sample to {debug_fn}")

    return out


AggName = Literal["sum", "max", "min", "mean", "first", "last", "median", "count"]


def pivot_agg_lazy(
    in_lf: pl.LazyFrame,
    value: str,
    agg: Union[AggName, Callable[[pl.Expr], pl.Expr]] = "sum",
    *,
    engine: str = "rust",  # "gpu" or "rust"
) -> pl.LazyFrame:
    """
    Lazily aggregate (store,item,date_str) → value, then EAGER pivot, then return LazyFrame.
    """
    # 1) Build the aggregation expression (lazy-safe)
    if callable(agg):
        expr = agg(pl.col(value)).alias(value)
    else:
        match agg:
            case "sum":
                expr = pl.col(value).sum().alias(value)
            case "max":
                expr = pl.col(value).max().alias(value)
            case "min":
                expr = pl.col(value).min().alias(value)
            case "mean":
                expr = pl.col(value).mean().alias(value)
            case "first":
                expr = pl.col(value).first().alias(value)
            case "last":
                expr = pl.col(value).last().alias(value)
            case "median":
                expr = pl.col(value).median().alias(value)
            case "count":
                expr = pl.len().alias(value)
            case _:
                raise ValueError(f"Unsupported agg: {agg}")

    # 2) Lazy group-by to get a TALL table: (store, item, date_str, value_agg)
    tall_lf = in_lf.group_by(["store", "item", "date_str"]).agg(expr)

    # 3) Collect JUST this small table; pivot is eager-only
    tall_df = tall_lf.collect(engine=engine)

    # 4) Eager pivot to WIDE: dates become columns
    wide_df = tall_df.pivot(
        index=["store", "item"],
        columns="date_str",
        values=value,
        aggregate_function="first",  # safe: we've already aggregated
    )

    # 5) Return back as LazyFrame for the rest of your lazy pipeline
    return wide_df.lazy()


# Ensure all date columns exist & order them
def ensure_cols_lazy(xlf: pl.LazyFrame, zero: int | float, *cols_str):
    # Convert schema (no collect) to know what's present
    have_cols = set(xlf.collect_schema().names())
    exprs = [pl.col("store"), pl.col("item")]
    for c in cols_str:
        exprs.append((pl.lit(zero).alias(c)) if c not in have_cols else pl.col(c))
    return xlf.select(exprs)


def generate_growth_rate_features_polars(
    df: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
    window_size: int = 1,
    *,
    calendar_aligned: bool = True,
    log_level: str = "INFO",
    output_path: Optional[Path] = None,
    weight_col: str = "weight",
    promo_col: str = "onpromotion",
    return_format: Literal["pandas", "polars", "lazy"] = "pandas",
):
    """
    Build growth-rate features with Polars, keeping the plan LAZY until the end.

    return_format:
        - "lazy"   -> returns pl.LazyFrame
        - "polars" -> returns pl.DataFrame
        - "pandas" -> returns pd.DataFrame  (default)
    """
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Normalize to LazyFrame up-front (no materialization yet)
    lf = _ensure_lazy(df)

    # Schema-only ops are cheap on LazyFrame
    names = set(lf.collect_schema().names())
    required = {"store", "item", "date", "unit_sales"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Canonical dtypes (still lazy)
    lf = lf.with_columns(
        pl.col("store").cast(pl.Int64, strict=False),
        pl.col("item").cast(pl.Int64, strict=False),
        pl.col("date").cast(pl.Date, strict=False),
        pl.col("unit_sales").cast(pl.Float32, strict=False),
    )

    # Optional columns with defaults
    if weight_col not in names:
        lf = lf.with_columns(pl.lit(1.0).cast(pl.Float32).alias(weight_col))
    else:
        lf = lf.with_columns(pl.col(weight_col).cast(pl.Float32, strict=False))

    if promo_col not in names:
        lf = lf.with_columns(pl.lit(0).cast(pl.Int8).alias(promo_col))
    else:
        lf = lf.with_columns(pl.col(promo_col).cast(pl.Int8, strict=False))

    # Stable string key for pivot/filters
    lf = lf.with_columns(pl.col("date").cast(pl.Utf8).alias("date_str"))

    # Weight map (lazy)
    weight_map_lf = lf.group_by(["store", "item"]).agg(
        pl.col(weight_col).first().alias(weight_col)
    )

    # Windows: we need the unique dates; collect minimally just for planning
    if generate_aligned_windows is None:
        raise ValueError("You must pass `generate_aligned_windows`.")

    # Only tiny projection to pandas to build windows
    win_pdf = (
        lf.select("store", "item", "date", "unit_sales", weight_col, promo_col)
        .collect(streaming=False)  # small subset; safe to collect
        .to_pandas()
    )
    windows: List[List[pd.Timestamp]] = generate_aligned_windows(
        win_pdf, window_size, calendar_aligned=calendar_aligned
    )
    if not windows:
        # return appropriately empty according to return_format
        empty_cols = (
            ["start_date", "store_item", "store", "item", weight_col]
            + [f"sales_day_{i}" for i in range(1, window_size + 1)]
            + [f"growth_rate_{i}" for i in range(1, window_size + 1)]
            + [f"{promo_col}_day_{i}" for i in range(1, window_size + 1)]
        )
        if return_format == "lazy":
            return pl.DataFrame(schema=[(c, pl.Null) for c in empty_cols]).lazy()
        if return_format == "polars":
            return pl.DataFrame({c: [] for c in empty_cols})
        return pd.DataFrame(columns=empty_cols)
    # Build per-window lazy pieces, then concat (still lazy)
    parts: List[pl.LazyFrame] = []
    for window_dates in windows:
        idx = pd.to_datetime(pd.Index(window_dates)).sort_values()
        cols_str = [d.strftime("%Y-%m-%d") for d in idx]
        start_dt = idx[0]

        wlf = lf.filter(pl.col("date_str").is_in(cols_str))
        sales_wide_lf = pivot_agg_lazy(wlf, "unit_sales", "sum", engine=polar_engine())
        promo_wide_lf = pivot_agg_lazy(
            wlf.select(["store", "item", "date_str", promo_col]),
            promo_col,
            "max",
            engine=polar_engine(),
        )

        sales_wide_lf = ensure_cols_lazy(sales_wide_lf, 0.0, *cols_str)
        promo_wide_lf = ensure_cols_lazy(promo_wide_lf, 0, *cols_str)

        base_lf = sales_wide_lf.join(weight_map_lf, on=["store", "item"], how="left")

        # Rename date cols to sales_day_i and drop originals
        renames = []
        drops = []
        for i, c in enumerate(cols_str, start=1):
            renames.append(pl.col(c).cast(pl.Float32).alias(f"sales_day_{i}"))
            drops.append(c)
        base_lf = base_lf.with_columns(renames).drop(drops)

        # Promo flags
        p = promo_wide_lf
        prenames = []
        pdrops = []
        for i, c in enumerate(cols_str, start=1):
            prenames.append(pl.col(c).cast(pl.Int8).alias(f"{promo_col}_day_{i}"))
            pdrops.append(c)
        p = p.with_columns(prenames).drop(pdrops)
        p = p.select(
            ["store", "item"]
            + [f"{promo_col}_day_{i}" for i in range(1, window_size + 1)]
        )
        base_lf = base_lf.join(p, on=["store", "item"], how="left")

        # prev-day total (lazy)
        prev_str = (start_dt - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        prev_sum_lf = (
            lf.filter(pl.col("date_str") == prev_str)
            .group_by(["store", "item"])
            .agg(pl.col("unit_sales").sum().alias("prev"))
        )
        base_lf = base_lf.join(prev_sum_lf, on=["store", "item"], how="left")

        # growth rates
        base_lf = base_lf.with_columns(
            pl.when(
                pl.col("prev").is_null()
                | (pl.col("prev") == 0)
                | pl.col("sales_day_1").is_null()
            )
            .then(pl.lit(None, dtype=pl.Float32))
            .otherwise(
                ((pl.col("sales_day_1") - pl.col("prev")) / pl.col("prev")) * 100.0
            )
            .alias("growth_rate_1")
        ).drop("prev")

        for i in range(2, window_size + 1):
            base_lf = base_lf.with_columns(
                pl.when(
                    pl.col(f"sales_day_{i-1}").is_null()
                    | (pl.col(f"sales_day_{i-1}") == 0)
                    | pl.col(f"sales_day_{i}").is_null()
                )
                .then(pl.lit(None, dtype=pl.Float32))
                .otherwise(
                    (
                        (pl.col(f"sales_day_{i}") - pl.col(f"sales_day_{i-1}"))
                        / pl.col(f"sales_day_{i-1}")
                    )
                    * 100.0
                )
                .alias(f"growth_rate_{i}")
            )

        # start_date + store_item
        base_lf = base_lf.with_columns(
            pl.lit(start_dt.to_pydatetime()).cast(pl.Datetime).alias("start_date"),
            (
                pl.col("store").cast(pl.Utf8)
                + pl.lit("_")
                + pl.col("item").cast(pl.Utf8)
            ).alias("store_item"),
        )

        base_lf = base_lf.select(
            ["start_date", "store_item", "store", "item", weight_col]
            + [f"sales_day_{i}" for i in range(1, window_size + 1)]
            + [f"growth_rate_{i}" for i in range(1, window_size + 1)]
            + [f"{promo_col}_day_{i}" for i in range(1, window_size + 1)]
        )

        parts.append(base_lf)

    out_lf = pl.concat(parts, how="vertical_relaxed")  # still lazy

    # ----- Save & return according to return_format -----
    if output_path is not None:
        # If user provided a custom saver, collect to pandas and let them handle it
        pdf = out_lf.collect(engine=polar_engine()).to_pandas()
        save_csv_or_parquet(pdf, output_path)
        # Optional debug slice
        dbg = out_lf.limit(50).collect(engine=polar_engine()).to_pandas()
        dbg.to_csv(Path(output_path).with_name("debug_subset.csv"), index=False)

    if return_format == "lazy":
        return out_lf
    if return_format == "polars":
        return out_lf.collect(engine=polar_engine())
    # default: pandas
    return out_lf.collect(engine=polar_engine()).to_pandas()


def generate_sales_features(
    df: pd.DataFrame,
    window_size: int = 1,
    *,
    calendar_aligned: bool = True,
    log_level: str = "INFO",
    output_path: Optional[Path] = None,
    epsilon: float = 1e-3,
) -> pd.DataFrame:
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

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
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
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
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
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
