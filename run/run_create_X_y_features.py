#!/usr/bin/env python3
"""
Training script for the Favorita Grocery Sales Forecasting model.

This script handles the complete training pipeline including:
- Data loading and preprocessing
- Logging
"""

import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_full_data
from src.utils import create_y_targets_from_shift


def add_y_targets(
    window_size: int,
    data_fn: Path,
    output_data_fn: Path,
    output_fn: Path,
    log_level: str,
):
    """Create features for training the model."""
    logger = logging.getLogger(__name__)
    logger.info("Starting adding data")
    df = load_full_data(
        data_fn=data_fn,
        window_size=window_size,
        output_fn=output_data_fn,
        log_level=log_level,
    )
    df = create_y_targets_from_shift(
        df,
        window_size=window_size,
        log_level=log_level,
        feature_prefixes=[
            "sales_day_",
            "store_med_logpct_change_",
            "item_med_logpct_change_",
        ],
    )
    if output_fn:
        logger.info(f"Saving final_df to {output_fn}")
        if output_fn.suffix == ".parquet":
            df.to_parquet(output_fn)
        else:
            df.to_csv(output_fn, index=False)
    return df


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration.

    Args:
        log_dir: Directory to save log files
        log_level: Logging level (e.g., 'INFO', 'DEBUG')

    Returns:
        Configured logger instance
    """
    # Create output directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set up log file path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"create_features_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create features for Favorita Grocery Sales Forecasting model"
    )
    parser.add_argument(
        "--data_fn",
        type=str,
        default="",
        help="Path to data file (relative to project root)",
    )
    parser.add_argument(
        "--output_data_fn",
        type=str,
        default="",
        help="Path to output data file (relative to project root)",
    )
    parser.add_argument(
        "--output_fn",
        type=str,
        default="",
        help="Path to output file (relative to project root)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=16,
        help="Size of the lookback window",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="../output/logs",
        help="Directory to save script outputs (relative to project root)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    data_fn = Path(args.data_fn).resolve()
    # output_data_fn = Path(args.output_data_fn).resolve()
    output_fn = Path(args.output_fn).resolve()
    log_dir = Path(args.log_dir).resolve()
    window_size = args.window_size

    # Set up logging
    print(f"Log dir: {log_dir}")
    logger = setup_logging(log_dir, args.log_level)

    try:
        # Log configuration
        logger.info("Starting adding y targets with configuration:")
        logger.info(f"  Data fn: {data_fn}")
        # logger.info(f"  Output data fn: {output_data_fn}")
        logger.info(f"  Output fn: {output_fn}")
        logger.info(f"  Log dir: {log_dir}")
        logger.info(f"  Window size: {window_size}")

        add_y_targets(
            window_size=window_size,
            data_fn=data_fn,
            output_data_fn=None,
            output_fn=output_fn,
            log_level=args.log_level,
        )

    except Exception as e:
        logger.error(f"Error adding y targets: {e}")
        raise


if __name__ == "__main__":
    main()
