from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Iterator
from tqdm import tqdm
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
import logging
from src.utils import save_csv_or_parquet, get_logger

logger = get_logger(__name__)

META_COLS = [
    "date",
    "store_item",
    "store",
    "item",
    "block_id",
]

SALE_COLS = [
    "unit_sales",
    "onpromotion",
    "growth_rate",
    "cluster_growth_rate_median",
    "cluster_unit_sales_median",
    "unit_sales_rolling_median",
    "unit_sales_ewm_decay",
    "growth_rate_rolling_median",
    "growth_rate_ewm_decay",
    "unit_sales_arima",
    "growth_rate_arima",
    "unit_sales_block",
    "bid_unit_sales_arima",
    "growth_rate_block",
    "bid_growth_rate_arima",
]


CYCLICAL_PREFIX_COLS = [
    "dayofweek",
    "weekofmonth",
    "monthofyear",
    "paycycle",
    "season",
]
TRIGS = ["sin", "cos"]

WEIGHT_COLUMN = "weight"

META_FEATURES = "META_FEATURES"
X_SALE_FEATURES = "X_SALE_FEATURES"
X_CYCLICAL_FEATURES = "X_CYCLICAL_FEATURES"
X_FEATURES = "X_FEATURES"
Y_FEATURES = "Y_FEATURES"
WEIGHT_COLUMN = "weight"
ALL_FEATURES = "ALL_FEATURES"


def build_feature_and_label_cols() -> dict[str, list[str]]:
    """Return feature and label column names for a given window size."""

    x_cyclical_features = [
        f"{feat}_{trig}" for feat in CYCLICAL_PREFIX_COLS for trig in TRIGS
    ]

    x_feature_cols = SALE_COLS + x_cyclical_features
    y_feature_col = ["y"]

    all_features = META_COLS + x_feature_cols + [WEIGHT_COLUMN] + y_feature_col

    features = dict(
        META_FEATURES=META_COLS,
        X_SALE_FEATURES=SALE_COLS,
        X_CYCLICAL_FEATURES=x_cyclical_features,
        X_FEATURES=x_feature_cols,
        Y_FEATURES=y_feature_col,
        ALL_FEATURES=all_features,
    )
    return features


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


def sort_df(df: pd.DataFrame, *, flag_duplicates: bool = True) -> pd.DataFrame:
    # --- Assert uniqueness of rows ---
    if flag_duplicates:
        if df.duplicated(subset=["date", "store_item"]).any():
            dups = df[df.duplicated(subset=["date", "store_item"], keep=False)]
            raise ValueError(
                f"Duplicate rows detected for date/store_item:\n{dups[['date', 'store_item']]}"
            )
    df = df.sort_values(["store_item", "date"], inplace=False).reset_index(drop=True)
    features = build_feature_and_label_cols()
    df = df[features[ALL_FEATURES]]
    return df


def get_nan_stats(df: pd.DataFrame):
    arr = df.to_numpy()
    num_total = arr.size
    num_nans = np.isnan(arr).sum()
    num_finite = np.isfinite(arr).sum()
    return num_nans, num_total, num_finite


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
            df = pd.read_csv(
                data_fn,
                low_memory=False,
            )
        df["store_item"] = df["store"].astype(str) + "_" + df["item"].astype(str)
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values(["date", "store_item"], inplace=True, kind="mergesort")
        logger.info(f"Loaded data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")


def compute_cluster_medians(
    df: pd.DataFrame,
    date_col: str = "date",
    cluster_col: str = "block_id",
    *,
    value_col: str = "growth_rate",
) -> pd.DataFrame:
    df = (
        df.groupby([cluster_col, date_col], observed=True)[value_col]
        .median()
        .rename(f"cluster_{value_col}_median")
        .reset_index()
    )
    return df


def generate_aligned_windows(
    df: pd.DataFrame,
    window_size: int,
    *,
    date_col: str = "date",
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

    df[date_col] = pd.to_datetime(df[date_col])

    if calendar_aligned:
        # --- back‑to‑front, gap‑free windows ---
        last_date = df[date_col].max().normalize()  # ensure time == 00:00
        first_date = df[date_col].min().normalize()

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
        unique_dates = sorted(pd.to_datetime(df[date_col].unique()))
        # logger.debug(f"Unique dates: {unique_dates}")
        return [
            unique_dates[i : i + window_size]
            for i in range(0, len(unique_dates), window_size)
        ]


# --- Fast ARIMA(0,0,1) forecasts ---
def arima001_forecast(
    series: pd.Series,
    *,
    min_history: int = 7,
    enforce_stationarity: bool = False,
    enforce_invertibility: bool = False,
) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    n = len(s)
    fc = pd.Series(np.nan, index=s.index)
    if n < max(3, min_history):
        return fc

    for i in range(min_history, n):
        sub = s.iloc[:i]
        if sub.isna().any() or sub.nunique(dropna=True) < 2:
            continue
        try:
            # Use a simple RangeIndex to avoid unsupported-index warnings
            sub_ = sub.reset_index(drop=True)

            res = ARIMA(
                sub_,
                order=(0, 0, 1),
                trend="c",
                enforce_stationarity=enforce_stationarity,
                enforce_invertibility=enforce_invertibility,
            ).fit(method_kwargs={"warn_convergence": False})

            pred = res.get_forecast(steps=1)
            fc.iloc[i] = float(
                pred.predicted_mean.iloc[0]
            )  # .iloc[0] avoids FutureWarning
        except Exception as e:
            logger.warning(f"failed at i={i}: {e}")

    return fc


def create_cyclical_features(
    df: pd.DataFrame,
    window_size: int = 7,
    *,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

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

    # last pay: 15th if day>=15 else previous month-end
    last_pay = pd.to_datetime(
        {
            "year": np.where(days >= 15, years, np.where(months > 1, years, years - 1)),
            "month": np.where(days >= 15, months, np.where(months > 1, months - 1, 12)),
            "day": np.where(days >= 15, 15, 1),
        }
    ) + pd.to_timedelta(
        np.where(days >= 15, 0, 0), unit="D"
    )  # day already set
    last_pay = last_pay.where(
        days >= 15, last_pay + pd.offsets.MonthEnd(0)
    )  # turn day=1 into prev month-end

    # next pay: month-end if day>=15 else 15th
    month_end_day = (dates + pd.offsets.MonthEnd(0)).dt.day
    next_pay = pd.to_datetime(
        {
            "year": years,
            "month": months,
            "day": np.where(days >= 15, month_end_day, 15),
        }
    )

    cycle_len = (next_pay - last_pay).dt.days.replace(0, np.nan).astype("float32")
    elapsed = (dates - last_pay).dt.days.astype("float32")
    paycycle_ratio = (
        (elapsed / cycle_len).clip(lower=0, upper=1).fillna(0).astype("float32")
    )
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

    # Rolling, EWM and Arima
    #  Ensure per-series ordering once
    df = df.sort_values(["store_item", "date"], kind="mergesort")

    # ---- Rolling & EWM per store_item (no cross-series bleed) ----
    df["unit_sales_rolling_median"] = df.groupby("store_item")["unit_sales"].transform(
        lambda s: s.rolling(window_size, min_periods=1).median()
    )

    df["unit_sales_ewm_decay"] = df.groupby("store_item")["unit_sales"].transform(
        lambda s: s.ewm(span=window_size, adjust=False).mean()
    )

    df["growth_rate_rolling_median"] = df.groupby("store_item")[
        "growth_rate"
    ].transform(lambda s: s.rolling(window_size, min_periods=1).median())

    df["growth_rate_ewm_decay"] = df.groupby("store_item")["growth_rate"].transform(
        lambda s: s.ewm(span=window_size, adjust=False).mean()
    )

    # ---- ARIMA per store_item (walk-forward) ----
    logger.info("Adding ARIMA(0,0,1) per store_item")
    df["unit_sales_arima"] = df.groupby("store_item", group_keys=False)[
        "unit_sales"
    ].apply(lambda s: arima001_forecast(s, min_history=7, enforce_stationarity=True))

    df["growth_rate_arima"] = df.groupby("store_item", group_keys=False)[
        "growth_rate"
    ].apply(lambda s: arima001_forecast(s, min_history=7, enforce_stationarity=True))

    # ---- Block-level ARIMA (aggregate by date -> ARIMA -> merge back) ----
    logger.info("Adding ARIMA(0,0,1) per block_id")
    block_daily = (
        df.groupby(["block_id", "date"], as_index=False)
        .agg(unit_sales_block=("unit_sales", "sum"))
        .sort_values(["block_id", "date"], kind="mergesort")
    )

    block_daily["growth_rate_block"] = (
        block_daily.sort_values(["block_id", "date"])
        .groupby("block_id")["unit_sales_block"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)  # first day (and div-by-zero) -> 0
    )

    block_daily["bid_unit_sales_arima"] = block_daily.groupby(
        "block_id", group_keys=False
    )["unit_sales_block"].apply(
        lambda s: arima001_forecast(s, min_history=7, enforce_stationarity=True)
    )
    block_daily["bid_growth_rate_arima"] = block_daily.groupby(
        "block_id", group_keys=False
    )["growth_rate_block"].apply(
        lambda s: arima001_forecast(s, min_history=7, enforce_stationarity=True)
    )

    df = df.merge(
        block_daily[
            [
                "block_id",
                "date",
                "unit_sales_block",
                "bid_unit_sales_arima",
                "growth_rate_block",
                "bid_growth_rate_arima",
            ]
        ],
        on=["block_id", "date"],
        how="left",
    )

    save_csv_or_parquet(df, output_path)
    return df


def zscore_with_axis(
    df: pd.DataFrame,
    axis: int = 0,
    nan_policy: str = "omit",
    epsilon: float = 1e-8,
    empty: str = "nan",  # what to do for all-NaN slices: "nan" | "zero" | "skip"
) -> pd.DataFrame:
    """
    Z-score along columns (axis=0) or rows (axis=1).

    nan_policy:
      - "omit": ignore NaNs within each slice (per-row/col); if a slice has no finite values,
                 outputs NaNs for that slice (or zeros if empty="zero").
      - "propagate": any NaN in a slice -> mean/std is NaN and the whole slice becomes NaN.

    empty:
      - "nan": for all-NaN slices, return NaNs in that slice
      - "zero": for all-NaN slices, return zeros in that slice
      - "skip": for all-NaN slices, leave them as-is (no scaling)
    """

    if axis not in (0, 1):
        raise ValueError("axis must be 0 (columns) or 1 (rows)")

    x = df.to_numpy(dtype=float)

    if nan_policy == "propagate":
        # Use plain mean/std which will yield NaN if any NaN in the slice.
        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, ddof=0, keepdims=True)
        std_safe = np.where((std == 0) | ~np.isfinite(std), epsilon, std)
        z = (x - mean) / std_safe
        return pd.DataFrame(z, index=df.index, columns=df.columns)

    if nan_policy != "omit":
        raise ValueError("nan_policy must be 'omit' or 'propagate'")

    # --- Omit policy: do the math without ever calling nanmean/nanstd on empty slices ---
    mask = np.isfinite(x)  # True for non-NaN, finite values
    # Count of valid values per slice
    cnt = np.sum(mask, axis=axis, keepdims=True)

    # Sums for mean/var using only valid entries
    x_sum = np.where(mask, x, 0.0).sum(axis=axis, keepdims=True)
    x2_sum = np.where(mask, x * x, 0.0).sum(axis=axis, keepdims=True)

    # Mean and variance with guards
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = x_sum / cnt
        ex2 = x2_sum / cnt
        var = ex2 - mean * mean  # numerically stable enough for z-scoring
        var = np.where(var < 0, 0.0, var)  # clamp tiny negatives from roundoff
        std = np.sqrt(var)

    # Identify all-NaN slices (cnt == 0)
    empty_slice = cnt == 0

    # Decide behavior for empty slices
    if empty == "nan":
        mean = np.where(empty_slice, np.nan, mean)
        std = np.where(empty_slice, np.nan, std)
    elif empty == "zero":
        mean = np.where(empty_slice, 0.0, mean)
        std = np.where(
            empty_slice, 1.0, std
        )  # so (x - 0)/1 = x; then replaced by 0 below
    elif empty == "skip":
        # Leave slice unchanged: set std to 1 and mean to 0, then put back originals later
        mean = np.where(empty_slice, 0.0, mean)
        std = np.where(empty_slice, 1.0, std)
    else:
        raise ValueError("empty must be 'nan', 'zero', or 'skip'")

    # Avoid div-by-zero or NaN std
    std_safe = np.where((std == 0) | ~np.isfinite(std), epsilon, std)

    z = (x - mean) / std_safe

    # For "zero": an all-NaN slice should become zeros (not NaNs)
    if empty == "zero":
        # Put zeros where the slice was empty
        if axis == 0:  # per column
            z[:, empty_slice.ravel()] = 0.0
        else:  # per row
            z[empty_slice.ravel(), :] = 0.0

    # For "skip": restore original values for empty slices (i.e., don't scale them)
    if empty == "skip":
        if axis == 0:
            z[:, empty_slice.ravel()] = x[:, empty_slice.ravel()]
        else:
            z[empty_slice.ravel(), :] = x[empty_slice.ravel(), :]

    return pd.DataFrame(z, index=df.index, columns=df.columns)


def median_mean_transform(
    df: pd.DataFrame,
    *,
    column_name="growth_rate_1",
    median_transform=True,
    mean_transform=False,
) -> pd.DataFrame:
    if median_transform:
        df = df.pivot_table(
            values=column_name, index="store", columns="item", aggfunc="median"
        )
    elif mean_transform:
        df = df.pivot_table(
            values=column_name, index="store", columns="item", aggfunc="mean"
        )
    else:
        raise ValueError("Set either median_transform or mean_transform to True.")
    df = df.sort_index().sort_index(axis=1)
    return df


def impute_with_col_mean_median(
    df: pd.DataFrame,
    cnt: pd.DataFrame,
    median_transform=True,
    mean_transform=False,
) -> pd.DataFrame:
    """
    Replace entries in `df` where `cnt == 0` with column means or medians
    computed only from observed (cnt > 0) values.

    Parameters
    ----------
    df : pd.DataFrame
        Data matrix to impute.
    cnt : pd.DataFrame
        Same shape as df, counts of observations.

    Returns
    -------
    pd.DataFrame
        Copy of df with missing entries replaced by column means.
    """
    # Compute means or medians only on observed values (cnt > 0)
    if mean_transform:
        col_means_obs = df.where(cnt > 0).mean(axis=0)
    elif median_transform:
        col_means_obs = df.where(cnt > 0).median(axis=0)
    else:
        raise ValueError("method must be 'mean' or 'median'")

    # Broadcast means to full shape
    mean_matrix = pd.DataFrame(
        [col_means_obs.values] * len(df), columns=df.columns, index=df.index
    )

    # Mask where cnt == 0, fill from col_means_obs
    df = df.where(cnt > 0, mean_matrix)

    return df


def normalize_data(
    df: pd.DataFrame,
    # freq="W",
    *,
    column_name="growth_rate",
    median_transform=True,
    mean_transform=False,
    log_transform=False,
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
    # Create count matrix first (before transforming df)
    cnt = df.pivot_table(
        values=column_name, index="store", columns="item", aggfunc="count"
    )

    # Transform df to store × item matrix
    df = median_mean_transform(
        df,
        column_name=column_name,
        median_transform=median_transform,
        mean_transform=mean_transform,
    )

    # Align count matrix with transformed df
    cnt = cnt.reindex(index=df.index, columns=df.columns, fill_value=0)

    # Impute missing values with column means
    df = impute_with_col_mean_median(df, cnt, median_transform, mean_transform)

    num_nans, num_total, num_finite = get_nan_stats(df)
    logger.info(
        f"Finite: {num_finite}, NaNs: {num_nans}, {100 * num_nans / num_total:.1f}%)"
    )

    # Check if imputation worked - if still all NaNs, fill with zeros
    if df.isna().all().all():
        logger.warning("Imputation failed - all values are NaN. Filling with zeros.")
        df = df.fillna(0)
    elif df.isna().any().any():
        logger.warning(f"Some NaN values remain after imputation. Filling with zeros.")
        df = df.fillna(0)

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


def split_by_block_id(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Splits the dataframe by (store_cluster, item_cluster) pairs and saves each to a compressed Parquet file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'block_id' column.
    output_dir : Path
        Directory where the Parquet files will be saved.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = df.groupby("block_id", sort=True)

    iterator = grouped
    if logger.level == logging.DEBUG:
        iterator = tqdm(grouped, desc="Generating cluster parquets")

    for block_id, group in iterator:
        filename = output_dir / f"block_{block_id}.parquet"
        logger.info(f"Saving block {block_id} → {filename}")
        save_csv_or_parquet(group, filename)


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
