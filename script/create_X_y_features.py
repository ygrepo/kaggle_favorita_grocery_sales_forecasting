#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path


# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import setup_logging, get_logger, save_csv_or_parquet
from src.data_utils import load_raw_data

logger = get_logger(__name__)


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
        "--output_fn",
        type=str,
        default="",
        help="Path to output file (relative to project root)",
    )
    parser.add_argument(
        "--log_fn",
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
    # Convert paths to absolute paths relative to project root
    data_fn = Path(args.data_fn).resolve()
    output_fn = Path(args.output_fn).resolve()
    log_fn = Path(args.log_fn).resolve()
    # Set up logging
    setup_logging(log_fn, args.log_level)
    try:
        # Log configuration
        logger.info("Starting:")
        logger.info(f"  Data fn: {data_fn}")
        logger.info(f"  Output fn: {output_fn}")
        logger.info(f"  Log fn: {log_fn}")

        df = load_raw_data(data_fn)

        # target: growth at t+1 (per series)
        df["y"] = df.groupby("store_item")["growth_rate"].shift(-1)

        # ARIMA forecasts for t+1, available at end of t
        df["unit_sales_arima_tplus1"] = df.groupby("store_item")[
            "unit_sales_arima"
        ].shift(-1)
        df["growth_rate_arima_tplus1"] = df.groupby("store_item")[
            "growth_rate_arima"
        ].shift(-1)
        df["bid_unit_sales_arima_tplus1"] = df.groupby("block_id")[
            "bid_unit_sales_arima"
        ].shift(-1)
        df["bid_growth_rate_arima_tplus1"] = df.groupby("block_id")[
            "bid_growth_rate_arima"
        ].shift(-1)

        # drop rows where next-step target doesn't exist
        df = df[df["y"].notna()].copy()

        logger.info(f"After valid observations shape: {df.shape}")
        save_csv_or_parquet(df, output_fn)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
