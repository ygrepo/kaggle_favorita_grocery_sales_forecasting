#!/usr/bin/env python3
"""
Grid search for tensor decomposition models.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tensor_models import tune_ranks
from src.utils import (
    setup_logging,
    read_csv_or_parquet,
    get_logger,
)

logger = get_logger(__name__)


def parse_ranks(ranks_str: str) -> tuple[int, int, int] | int | None:
    """
    Parse ranks from comma-separated string.
    Expected format: "rI,rJ,rK" e.g., "12,12,40" or "r" e.g., "10"
    Returns tuple of 3 integers or None if empty.
    """
    if not ranks_str or ranks_str.strip() == "":
        return None

    try:
        parts = [int(x.strip()) for x in ranks_str.split(",")]
        if len(parts) == 1:
            return parts[0]
        return tuple(parts)  # type: ignore
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid ranks format: {ranks_str}. "
            f"Expected 'rI,rJ,rK' with integers (e.g., '12,12,40')"
            f"or 'r' with an integer (e.g., '10'). "
            f"Error: {e}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid search for tensor decomposition models"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="tucker",
        choices=["tucker", "ntf", "parafac"],
        help="Decomposition method",
    )
    parser.add_argument(
        "--store_ranks",
        type=str,
        default="",
        help="Comma-separated list of store ranks to test (for Tucker only)",
    )
    parser.add_argument(
        "--sku_ranks",
        type=str,
        default="",
        help="Comma-separated list of item ranks to test (for Tucker only)",
    )
    parser.add_argument(
        "--feature_ranks",
        type=str,
        default="",
        help="Comma-separated list of feature ranks to test (for Tucker only)",
    )
    parser.add_argument(
        "--rank_list",
        type=str,
        default="",
        help="Comma-separated list of ranks to test (for PARAFAC/NTF only)",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Path to save results (relative to project root)",
    )
    parser.add_argument(
        "--data_fn",
        type=str,
        default="",
        help="Path to training data file (relative to project root)",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=500,
        help="Maximum number of iterations",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="Tolerance for convergence",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="",
        help="Comma-separated list of feature names",
    )
    parser.add_argument(
        "--log_fn",
        type=str,
        default="",
        help="Path to save script outputs (relative to project root)",
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
    log_fn = Path(args.log_fn).resolve()

    # Set up logging
    setup_logging(log_fn, args.log_level)
    try:
        # Log configuration
        logger.info("Starting grid search with configuration:")
        logger.info(f"  Method: {args.method}")
        logger.info(f"  Output path: {args.output_path}")
        logger.info(f"  Store ranks: {args.store_ranks}")
        logger.info(f"  SKU ranks: {args.sku_ranks}")
        logger.info(f"  Feature ranks: {args.feature_ranks}")
        logger.info(f"  Rank list: {args.rank_list}")
        logger.info(f"  Data fn: {args.data_fn}")
        logger.info(f"  Features: {args.features}")
        logger.info(f"  Max iter: {args.max_iter}")
        logger.info(f"  Tolerance: {args.tol}")
        logger.info(f"  Log level: {args.log_level}")

        data_fn = Path(args.data_fn).resolve()
        # Load and preprocess data
        df = read_csv_or_parquet(data_fn)
        logger.info(f"Data loaded: {df.head()}")
        output_path = args.output_path.resolve()
        try:
            # Parse the strings into lists of integers
            store_ranks_list = [int(r) for r in args.store_ranks.split(",")]
            sku_ranks_list = [
                int(r) for r in args.sku_ranks.split(",")
            ]  # Your log calls it 'SKU ranks'
            feature_ranks_list = [
                int(r) for r in args.feature_ranks.split(",")
            ]
            rank_list_list = [int(r) for r in args.rank_list.split(",")]

        except ValueError as e:
            logger.error(
                f"Invalid rank argument. Ranks must be comma-separated integers. Error: {e}"
            )
            # You should probably exit here
            return

        results_df = tune_ranks(
            args.method,
            df,
            args.features,
            output_path,
            rank_list=rank_list_list,
            store_ranks=store_ranks_list,
            sku_ranks=sku_ranks_list,
            feature_ranks=feature_ranks_list,
            n_iter=args.max_iter,
            tol=args.tol,
        )
        logger.info(f"Results:\n{results_df.head()}")
        logger.info("Grid search completed successfully")
    except Exception as e:
        logger.error(f"Error running grid search: {e}")
        raise


if __name__ == "__main__":
    main()
