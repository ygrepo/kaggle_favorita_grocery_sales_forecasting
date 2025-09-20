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


def polar_engine():
    """Return appropriate Polars engine based on GPU availability."""
    return "gpu" if torch.cuda.is_available() else "rust"


# def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
#     """Set up logging configuration.

#     Args:
#         log_dir: Directory to save log files
#         log_level: Logging level (e.g., 'INFO', 'DEBUG')

#     Returns:
#         Configured logger instance
#     """
#     # Create output directory if it doesn't exist
#     log_dir.mkdir(parents=True, exist_ok=True)

#     # Set up log file path with timestamp
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     log_file = log_dir / f"create_features_{timestamp}.log"

#     # Configure logging
#     logging.basicConfig(
#         level=getattr(logging, log_level),
#         format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#         handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
#     )

#     logger = logging.getLogger(__name__)
#     logger.info(f"Logging to {log_file}")
#     return logger


# ---- One base for everything ----
BASE_LOGGER = "retail"
_BASE = logging.getLogger(BASE_LOGGER)  # the only logger we configure here


def setup_logging(log_path: str | Path | None, level: str = "INFO") -> logging.Logger:
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
    return logging.getLogger(BASE_LOGGER if not name else f"{BASE_LOGGER}.{name}")


logger = get_logger(__name__)

# def get_logger(logger_name: str, log_level: str = "INFO") -> logging.Logger:
#     """Get a logger with the specified name and log level."""
#     logger = logging.getLogger(logger_name)
#     logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

#     # Ensure at least one handler exists
#     if not logger.handlers:
#         handler = logging.StreamHandler()
#         formatter = logging.Formatter(
#             fmt="%(asctime)s - %(levelname)s - %(message)s",
#             datefmt="%Y-%m-%d %H:%M:%S",
#         )
#         handler.setFormatter(formatter)
#         logger.addHandler(handler)

#     return logger


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Expected a boolean value (true/false)")


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


def save_csv_or_parquet(df: pd.DataFrame, fn: Path) -> None:
    if fn.suffix == ".parquet":
        df.to_parquet(fn)
    else:
        df.to_csv(fn, index=False)


def read_csv_or_parquet(fn: Path) -> pd.DataFrame:
    if fn.suffix == ".parquet":
        return pd.read_parquet(fn)
    else:
        return pd.read_csv(fn)
