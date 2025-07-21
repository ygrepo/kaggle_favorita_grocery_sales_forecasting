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

from src.data_utils import load_full_data, create_y_targets_from_shift
from src.utils import setup_logging


def add_y_targets(
    sale_cyc_dir: Path,
    window_size: int,
    *,
    output_dir: Path,
    prefix: str = "sale_cyc_features_X_1_day_y",
    log_level: str = "INFO",
):
    """
    Process each Parquet file in a directory, apply feature creation,
    and save the output with a prefix.

    Parameters
    ----------
    sale_cyc_dir : Path
        Path to a directory containing parquet files or a single parquet file.
    window_size : int
        Rolling window size for feature creation.
    log_level : str
        Logging level (e.g., "INFO", "DEBUG").
    prefix : str
        Prefix to use when saving processed files.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if sale_cyc_dir.is_file() and sale_cyc_dir.suffix == ".parquet":
        files = [sale_cyc_dir]
    else:
        files = list(sale_cyc_dir.glob("*.parquet"))

    logger.info(
        f"Processing sales (store cluster, SKU cluster) {len(files)} Parquet files..."
    )

    for file_path in files:
        logger.info(f"Processing {file_path.name}")
        parts = file_path.stem.split("_")
        store_cluster = int(parts[-2])
        item_cluster = int(parts[-1])
        logger.info(f"Store cluster: {store_cluster}")
        logger.info(f"Item cluster: {item_cluster}")
        output_path = output_dir / f"{prefix}_{store_cluster}_{item_cluster}.parquet"
        df = load_full_data(
            data_fn=file_path,
            window_size=window_size,
            log_level=log_level,
        )
        logger.info(f"df shape: {df.shape}")
        logger.info(f"Output path: {output_path}")
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
        logger.info(f"Saving df to {output_path}")
        if output_path.suffix == ".parquet":
            df.to_parquet(output_path)
        else:
            df.to_csv(output_path, index=False)

    logger.info("Features created successfully")
    return


# def add_y_targets(
#     window_size: int,
#     data_fn: Path,
#     output_data_fn: Path,
#     output_fn: Path,
#     log_level: str,
# ):
#     """Create features for training the model."""
#     logger = logging.getLogger(__name__)
#     logger.info("Starting adding data")
#     df = load_full_data(
#         data_fn=data_fn,
#         window_size=window_size,
#         output_fn=output_data_fn,
#         log_level=log_level,
#     )
#     df = create_y_targets_from_shift(
#         df,
#         window_size=window_size,
#         log_level=log_level,
#         feature_prefixes=[
#             "sales_day_",
#             "store_med_logpct_change_",
#             "item_med_logpct_change_",
#         ],
#     )
#     if output_fn:
#         logger.info(f"Saving final_df to {output_fn}")
#         if output_fn.suffix == ".parquet":
#             df.to_parquet(output_fn)
#         else:
#             df.to_csv(output_fn, index=False)
#     return df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create features for Favorita Grocery Sales Forecasting model"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Path to data directory (relative to project root)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Path to output directory (relative to project root)",
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
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    log_dir = Path(args.log_dir).resolve()
    window_size = args.window_size

    # Set up logging
    logger = setup_logging(log_dir, args.log_level)
    logger.info(f"Log dir: {log_dir}")

    try:
        # Log configuration
        logger.info("Starting adding y targets with configuration:")
        logger.info(f"  Data dir: {data_dir}")
        logger.info(f"  Output dir: {output_dir}")
        logger.info(f"  Log dir: {log_dir}")
        logger.info(f"  Window size: {window_size}")

        add_y_targets(
            sale_cyc_dir=data_dir,
            window_size=window_size,
            output_dir=output_dir,
            log_level=args.log_level,
        )

    except Exception as e:
        logger.error(f"Error adding y targets: {e}")
        raise


if __name__ == "__main__":
    main()
