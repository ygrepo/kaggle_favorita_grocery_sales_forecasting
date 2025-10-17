#!/usr/bin/env python3
"""
Clustering script for the Favorita Grocery Sales Forecasting model.

This script handles the complete clustering pipeline including:
- Data loading and preprocessing
- Logging
"""

import sys
import argparse
from pathlib import Path

# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_utils import (
    build_growth_features_for_clustering,
)
from src.utils import (
    setup_logging,
    get_logger,
    save_csv_or_parquet,
    read_csv_or_parquet,
)

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clustering for Favorita Grocery Sales Forecasting model"
    )

    parser.add_argument(
        "--data_fn",
        type=str,
        default="",
        help="Path to training data file (relative to project root)",
    )
    parser.add_argument(
        "--output_scaled_features_fn",
        type=str,
        default="",
        help="Path to output scaled features file (relative to project root)",
    )
    parser.add_argument(
        "--output_features_fn",
        type=str,
        default="",
        help="Path to output file (relative to project root)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.01,
        help="Tau parameter for robust summaries",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=4,
        help="Window size for smoothing trajectories",
    )
    parser.add_argument(
        "--log_fn",
        type=str,
        default="",
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
    setup_logging(Path(args.log_fn), args.log_level)

    try:
        # Log configuration
        logger.info("Starting with configuration:")
        logger.info(f"  Data fn: {args.data_fn}")
        logger.info(
            f"  Output scaled features fn: {args.output_scaled_features_fn}"
        )
        logger.info(f"  Output feature fn: {args.output_features_fn}")
        logger.info(f"  Tau: {args.tau}")
        logger.info(f"  Smooth window: {args.smooth_window}")
        logger.info(f"  Log fn: {args.log_fn}")
        logger.info(f"  Log level: {args.log_level}")

        data_fn = Path(args.data_fn).resolve()

        df = read_csv_or_parquet(data_fn)

        scaled_features_df, features_df, diag = (
            build_growth_features_for_clustering(
                df,
                tau=args.tau,
                smooth_window=args.smooth_window,
            )
        )
        # See which columns were pruned (near-empty) and their support
        logger.info(diag.sort_values("nan_frac", ascending=False).head(10))
        if any(
            (diag["feature"].isin(["ac_lag12", "seasonal_strength"]))
            & (diag["dropped"])
        ):
            logger.warning(
                "Long-horizon features were dropped due to low support (short history)."
            )

        scaled_features_fn = Path(args.output_scaled_features_fn).resolve()

        save_csv_or_parquet(scaled_features_df, scaled_features_fn)

        features_fn = Path(args.output_features_fn).resolve()
        save_csv_or_parquet(features_df, features_fn)

        logger.info("Completed successfully")
    except Exception as e:
        logger.error(f"Error Generating growth rate data: {e}")
        raise


if __name__ == "__main__":
    main()
