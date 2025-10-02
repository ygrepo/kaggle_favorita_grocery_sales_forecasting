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


def parse_range(range_str):
    try:
        start, end = map(int, range_str.split(":"))
        return range(start, end)
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Invalid range format: {range_str}. Use START:END"
        ) from e
