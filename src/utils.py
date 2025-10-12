import os
import sys
import argparse
from datetime import datetime
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


def _safe_autocorr(x: pd.Series, lag: int) -> float:
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

    if ok.sum() < 3:  # Need at least 3 points for meaningful correlation
        return np.nan

    a_valid, b_valid = a[ok], b[ok]

    # Check if there's any variation
    if np.var(a_valid) == 0 or np.var(b_valid) == 0:
        return np.nan

    a0 = a_valid - a_valid.mean()
    b0 = b_valid - b_valid.mean()

    # Use ddof=0 for small samples, ddof=1 for larger samples
    ddof = 1 if len(a_valid) > 10 else 0

    try:
        a_std = np.std(a0, ddof=ddof)
        b_std = np.std(b0, ddof=ddof)

        if a_std == 0 or b_std == 0:
            return np.nan

        denom = a_std * b_std
        return (a0 * b0).mean() / denom if denom > 0 else np.nan

    except (RuntimeWarning, ValueError):
        logger.warning(f"Error computing autocorr for lag {lag}")
        return np.nan


def _trend_slope(y: pd.Series) -> float:
    """
    Returns the correlation between value and time (computed as the mean of
    the product of the z-scores).
    """
    y = pd.to_numeric(y, errors="coerce").astype(float)
    n = len(y)
    if n < 5 or y.notna().sum() < 5:
        return np.nan

    x = np.arange(n, dtype=float)
    ok = ~y.isna()
    x, y = x[ok], y[ok]

    if len(y) < 5:
        return np.nan

    # Check for variation
    if np.var(x) == 0 or np.var(y) == 0:
        return np.nan

    # Use ddof=0 for small samples to avoid the warning
    ddof = 1 if len(y) > 10 else 0

    try:
        x_std = np.std(x, ddof=ddof)
        y_std = np.std(y, ddof=ddof)

        if x_std == 0 or y_std == 0:
            return np.nan

        x = (x - x.mean()) / x_std
        y = (y - y.mean()) / y_std

        return float((x * y).mean())

    except (RuntimeWarning, ValueError):
        logger.warning("Error computing trend slope")
        return np.nan


def _seasonal_corr(weeks: pd.Series, values: pd.Series, period=52) -> float:
    """Correlate with fundamental seasonal sine/cosine over week index"""
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
    if np.var(v_valid) == 0:
        return np.nan

    # Use ddof=0 for small samples
    ddof = 1 if len(v_valid) > 15 else 0

    try:
        v_std = np.std(v_valid, ddof=ddof)
        s_std = np.std(s_valid, ddof=ddof)
        c_std = np.std(c_valid, ddof=ddof)

        if v_std == 0 or s_std == 0 or c_std == 0:
            return np.nan

        v_norm = (v_valid - np.mean(v_valid)) / v_std
        s_norm = (s_valid - np.mean(s_valid)) / s_std
        c_norm = (c_valid - np.mean(c_valid)) / c_std

        # magnitude of projection onto seasonal basis
        cs = np.mean(v_norm * s_norm)
        cc = np.mean(v_norm * c_norm)
        return float(np.sqrt(cs**2 + cc**2))

    except (RuntimeWarning, ValueError):
        logger.warning("Error computing seasonal correlation")
        return np.nan


# --- base aggregates with safer std calculation ---
def safe_std(s):
    """Calculate std with fallback for small samples"""
    s_numeric = pd.to_numeric(s, errors="coerce")
    s_clean = s_numeric.dropna()
    if len(s_clean) <= 1:
        logger.warning("Not enough data to compute std")
        return np.nan
    # Use ddof=0 for very small samples
    ddof = 1 if len(s_clean) > 5 else 0
    return s_clean.std(ddof=ddof)


def safe_iqr(s):
    """Calculate IQR safely"""
    s_numeric = pd.to_numeric(s, errors="coerce")
    s_clean = s_numeric.dropna()
    if len(s_clean) < 2:
        logger.warning("Not enough data to compute IQR")
        return np.nan
    return np.percentile(s_clean, 75) - np.percentile(s_clean, 25)
