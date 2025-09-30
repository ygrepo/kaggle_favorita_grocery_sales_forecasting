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
import numpy as np

# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_utils import (
    load_raw_data,
    save_csv_or_parquet,
)
from src.utils import setup_logging, get_logger

logger = get_logger(__name__)


def parse_range(range_str):
    try:
        start, end = map(int, range_str.split(":"))
        return range(start, end)
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Invalid range format: {range_str}. Use START:END"
        ) from e


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
        "--output_fn",
        type=str,
        default="",
        help="Path to output file (relative to project root)",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel processes to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for multiprocessing",
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
        logger.info(f"  Output fn: {args.output_fn}")
        logger.info(f"  N jobs: {args.n_jobs}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Log fn: {args.log_fn}")
        logger.info(f"  Log level: {args.log_level}")

        data_fn = Path(args.data_fn).resolve()

        df = load_raw_data(data_fn)

        output_fn = Path(args.output_fn).resolve()
        df["unit_sales"] = df["unit_sales"].astype(float)
        # weekly aggregation
        wk = (
            df.set_index("date")
            .groupby("store_item")["unit_sales"]
            .resample("W-SUN")  # explicit: week ends Sunday
            .sum()
            .rename("sales_wk")
            .reset_index()
            .sort_values(["store_item", "date"])
        )

        # targets (same as you had) ...
        wk["growth_rate"] = (
            wk.groupby("store_item")["sales_wk"]
            .pct_change(fill_method=None)
            .replace([np.inf, -np.inf], np.nan)
        )
        lo, hi = wk["growth_rate"].quantile([0.01, 0.99])
        wk["growth_rate_clipped"] = wk["growth_rate"].clip(lo, hi)
        wk["growth_binary"] = (
            wk["sales_wk"] > wk.groupby("store_item")["sales_wk"].shift(1)
        ).astype("Int8")
        wk["growth_continuous"] = wk["growth_rate_clipped"].where(
            wk["growth_binary"] == 1
        )

        # --- make matching week keys ---
        wk = wk.rename(columns={"date": "week_end"})
        wk["week_end"] = wk["week_end"].dt.normalize()  # -> Sunday 00:00:00

        # This matches resample labels:
        df["week_end"] = (
            df["date"].dt.to_period("W-SUN").dt.end_time
        ).dt.normalize()
        # alternative (equivalent):
        # df["week_end"] = (df["date"] + pd.offsets.Week(weekday=6)).dt.normalize()

        # merge
        df = df.merge(
            wk[
                [
                    "store_item",
                    "week_end",
                    "sales_wk",
                    "growth_rate",
                    "growth_rate_clipped",
                    "growth_binary",
                    "growth_continuous",
                ]
            ],
            on=["store_item", "week_end"],
            how="left",
        )

        # df["growth_rate"] = (
        #     df["unit_sales"]
        #     .pct_change(fill_method=None)
        #     .fillna(df["unit_sales"].pct_change(fill_method=None).median())
        # ) * 100.0
        save_csv_or_parquet(df, output_fn)
        logger.info("Completed successfully")
    except Exception as e:
        logger.error(f"Error Generating growth rate data: {e}")
        raise


if __name__ == "__main__":
    main()
