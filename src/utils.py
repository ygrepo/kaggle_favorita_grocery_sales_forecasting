import os
import sys
import argparse
from pathlib import Path
import torch
import random
import logging
import numpy as np
import pandas as pd
import multiprocessing


def polar_engine():
    """Return appropriate Polars engine based on GPU availability."""
    return "gpu" if torch.cuda.is_available() else "rust"


# ---- One base for everything ----
BASE_LOGGER = "base_logger"
_BASE = logging.getLogger(BASE_LOGGER)  # the only logger we configure here


def setup_logging(
    log_path: str | Path | None, level: str = "INFO"
) -> logging.Logger:
    """Configure the base logger once (file + console)."""
    if getattr(_BASE, "_configured", False):
        return _BASE

    _BASE.handlers.clear()
    _BASE.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - pid=%(process)d - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Optional file handler
    if log_path:
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setFormatter(fmt)
        _BASE.addHandler(fh)

    # Console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    _BASE.addHandler(sh)

    # Do not bubble to the *root* logger
    _BASE.propagate = False
    _BASE._configured = True
    return _BASE


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a child logger that inherits the base handlers."""
    return logging.getLogger(
        BASE_LOGGER if not name else f"{BASE_LOGGER}.{name}"
    )


logger = get_logger(__name__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(
            "Expected a boolean value (true/false)"
        )


def set_seed(seed: int = 42):
    """Set seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For full reproducibility in future versions
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_csv_or_parquet(df: pd.DataFrame, fn: Path | None) -> None:
    if fn is None:
        logger.warning("No output file specified. Not saving.")
        return
    logger.info(f"Saving df to {fn}")
    if fn.suffix == ".parquet":
        df.to_parquet(fn)
    else:
        df.to_csv(fn, index=False)


def read_csv_or_parquet(fn: Path) -> pd.DataFrame:
    logger.info(f"Loading df from {fn}")
    if fn.suffix == ".parquet":
        return pd.read_parquet(fn)
    else:
        return pd.read_csv(fn)


def get_n_jobs(n_jobs: int) -> int:
    # Determine number of processes
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    elif n_jobs <= 1:
        n_jobs = 1
    return n_jobs


def parse_range(arg: str):
    """
    Parse a CLI argument that can be either:
      - a colon-separated range 'START:END' (inclusive of START, exclusive of END),
      - or a comma-separated list 'a,b,c'.
    Returns a Python range or list of numbers (int or float).
    """
    # Handle empty string
    if not arg or arg.strip() == "":
        return None

    if ":" in arg:
        start, end = arg.split(":")
        # Try float first, fall back to int
        try:
            start_val = float(start)
            end_val = float(end)
            if start_val.is_integer() and end_val.is_integer():
                return range(int(start_val), int(end_val))
            else:
                # For float ranges, we can't use range(), return a list
                import numpy as np

                return np.arange(start_val, end_val, 0.001).tolist()
        except ValueError:
            return range(int(start), int(end))
    elif "," in arg:
        # Try to parse as floats first, fall back to ints
        try:
            return [float(x.strip()) for x in arg.split(",")]
        except ValueError:
            return [int(x.strip()) for x in arg.split(",")]
    else:
        # single value - try float first, fall back to int
        try:
            val = float(arg)
            return [val]
        except ValueError:
            return [int(arg)]


def safe_autocorr(x: pd.Series, lag: int) -> float:
    """
    compute the lag-lag autocorrelation (i.e.,
    Pearson correlation between x_t and x_{t−lag})
    while being robust to NaNs and tiny samples.
    """
    x = pd.to_numeric(x, errors="coerce").astype(float)
    if x.isna().sum() > len(x) - 3 or len(x) <= lag + 2:
        return np.nan

    a = x.values
    b = np.roll(a, lag)
    b[:lag] = np.nan
    ok = ~np.isnan(a) & ~np.isnan(b)

    if ok.sum() < 3:
        return np.nan

    a_valid, b_valid = a[ok], b[ok]

    # Check for variation
    if len(np.unique(a_valid)) == 1 or len(np.unique(b_valid)) == 1:
        return np.nan

    a0 = a_valid - a_valid.mean()
    b0 = b_valid - b_valid.mean()

    a_std = np.sqrt(np.mean(a0**2))
    b_std = np.sqrt(np.mean(b0**2))

    if a_std == 0 or b_std == 0:
        return np.nan

    return (a0 * b0).mean() / (a_std * b_std)


def trend_slope(y: pd.Series) -> float:
    """
    Returns the correlation between value and time
    """
    y = pd.to_numeric(y, errors="coerce").astype(float)
    n = len(y)
    if n < 5 or y.notna().sum() < 5:
        return np.nan

    x = np.arange(n, dtype=float)
    ok = ~y.isna()
    x, y = x[ok], y[ok]

    if len(y) < 5 or len(np.unique(y)) == 1:
        return np.nan

    # Calculate correlation manually to avoid ddof issues
    x_mean, y_mean = x.mean(), y.mean()
    x_centered = x - x_mean
    y_centered = y - y_mean

    x_std = np.sqrt(np.mean(x_centered**2))
    y_std = np.sqrt(np.mean(y_centered**2))

    if x_std == 0 or y_std == 0:
        return np.nan

    return float(np.mean(x_centered * y_centered) / (x_std * y_std))


def seasonal_corr(values: pd.Series, period: int = 52) -> float:
    """Correlate with fundamental seasonal sine/cosine"""
    k = np.arange(len(values), dtype=float)
    s = np.sin(2 * np.pi * k / period)
    c = np.cos(2 * np.pi * k / period)
    v = pd.to_numeric(values, errors="coerce").astype(float).values
    ok = ~np.isnan(v)

    if ok.sum() < 8:
        return np.nan

    v_valid = v[ok]
    s_valid = s[ok]
    c_valid = c[ok]

    # Check for variation
    if len(np.unique(v_valid)) == 1:
        return np.nan

    # Manual standardization to avoid ddof issues
    v_mean = np.mean(v_valid)
    s_mean = np.mean(s_valid)
    c_mean = np.mean(c_valid)

    v_centered = v_valid - v_mean
    s_centered = s_valid - s_mean
    c_centered = c_valid - c_mean

    v_std = np.sqrt(np.mean(v_centered**2))
    s_std = np.sqrt(np.mean(s_centered**2))
    c_std = np.sqrt(np.mean(c_centered**2))

    if v_std == 0 or s_std == 0 or c_std == 0:
        return np.nan

    v_norm = v_centered / v_std
    s_norm = s_centered / s_std
    c_norm = c_centered / c_std

    cs = np.mean(v_norm * s_norm)
    cc = np.mean(v_norm * c_norm)
    return float(np.sqrt(cs**2 + cc**2))


# --- base aggregates with safer std calculation ---
def safe_std(s: pd.Series) -> float:
    """Calculate std safely without ddof warnings"""
    s_numeric = pd.to_numeric(s, errors="coerce")
    s_clean = s_numeric.dropna()
    if len(s_clean) <= 1:
        return np.nan
    # Use population std to avoid ddof issues
    return float(np.std(s_clean))


def safe_iqr(s: pd.Series) -> float:
    """Calculate IQR safely"""
    s_numeric = pd.to_numeric(s, errors="coerce")
    s_clean = s_numeric.dropna()
    if len(s_clean) < 2:
        return np.nan
    return float(np.percentile(s_clean, 75) - np.percentile(s_clean, 25))


def safe_median(s: pd.Series) -> float:
    """Calculate median safely"""
    s_numeric = pd.to_numeric(s, errors="coerce")
    s_clean = s_numeric.dropna()
    if len(s_clean) == 0:
        return np.nan
    return float(np.median(s_clean))


def safe_nanmean(s: pd.Series) -> float:
    """Calculate mean safely"""
    s_numeric = pd.to_numeric(s, errors="coerce")
    if s_numeric.isna().all():
        return np.nan
    return float(np.nanmean(s_numeric))


def safe_rolling_mean(
    series: pd.Series, window: int, min_periods: int = 1
) -> pd.Series:
    """Safe rolling mean that handles edge cases"""
    if len(series) == 0 or series.isna().all():
        return pd.Series(index=series.index, dtype=float)
    return series.rolling(window, min_periods=min_periods).mean()


def build_multifeature_X_matrix(
    df: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build a 3D matrix of shape (I, J, D) from a DataFrame with D features"""
    df[["store", "item"]] = df["store_item"].str.split("_", n=1, expand=True)
    stores = np.sort(df["store"].unique())
    items = np.sort(df["item"].unique())
    I, J, D = len(stores), len(items), len(features)
    rpos = {s: i for i, s in enumerate(stores)}
    cpos = {t: j for j, t in enumerate(items)}

    X = np.full((I, J, D), np.nan, dtype=np.float64)
    M = np.zeros((I, J), dtype=bool)

    for _, row in df.iterrows():
        i = rpos[row["store"]]
        j = cpos[row["item"]]
        X[i, j, :] = row[features].to_numpy(dtype=np.float64)
        M[i, j] = True

    # z-score each channel using only observed cells; keep NaNs for now
    for d in range(D):
        vals = X[..., d][M]
        # Use mean for centering, matching the standard deviation for scaling
        mu = np.nanmean(vals)  
        s = np.nanstd(vals) if np.nanstd(vals) > 1e-12 else 1.0
        X[..., d] = (X[..., d] - mu) / s
    row_names = stores.tolist()
    col_names = items.tolist()
    return X, M, row_names, col_names
