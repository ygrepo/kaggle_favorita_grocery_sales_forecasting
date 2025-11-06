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
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tensor_models import (
    fit,
    get_threshold_k_assignments,
    create_assignments,
)
from src.utils import setup_logging, read_csv_or_parquet, get_logger

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
        description="Clustering for Retail Sales Forecasting model"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="tucker",
        choices=["tucker", "ntf", "parafac"],
        help="Decomposition method",
    )
    parser.add_argument(
        "--ranks",
        type=str,
        default="",
        help="Comma-separated list of ranks",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Threshold for top-k assignments",
    )
    parser.add_argument(
        "--factor_names",
        type=str,
        default="Store,SKU,Feature",
        help="Comma-separated list of factor names",
    )
    parser.add_argument(
        "--data_fn",
        type=str,
        default="",
        help="Path to training data file (relative to project root)",
    )
    parser.add_argument(
        "--model_fn",
        type=str,
        default="",
        help="Path to save model (relative to project root)",
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
    args = parse_args()
    log_fn = Path(args.log_fn).resolve()
    setup_logging(log_fn, args.log_level)

    try:
        # Log configuration
        logger.info("Starting data clustering with configuration:")
        logger.info(f"  Method: {args.method}")
        logger.info(f"  Method: {args.method}")
        logger.info(f"  Ranks: {args.ranks}")
        logger.info(f"  Threshold: {args.threshold}")
        logger.info(f"  Factor names: {args.factor_names}")
        logger.info(f"  Data fn: {args.data_fn}")
        logger.info(f"  Model fn: {args.model_fn}")
        logger.info(f"  Features: {args.features}")
        logger.info(f"  Max iter: {args.max_iter}")
        logger.info(f"  Tolerance: {args.tol}")
        logger.info(f"  Log level: {args.log_level}")

        data_fn = Path(args.data_fn).resolve()
        df = read_csv_or_parquet(data_fn)

        try:
            rank_ints = [int(r) for r in args.ranks.split(",")]

            if args.method == "tucker":
                if len(rank_ints) != 3:
                    logger.error(
                        f"For 'tucker', ranks must have 3 integers (e.g., '40,300,8'). Got: {args.ranks}"
                    )
                    return
                rank_ints = tuple(rank_ints)
            elif args.method in ["parafac", "ntf"]:
                if len(rank_ints) != 1:
                    logger.error(
                        f"For '{args.method}', ranks must have 1 integer. Got: {args.ranks}"
                    )
                    return
                rank_ints = rank_ints[0]  # PARAFAC/NTF take a single int
            else:
                logger.error(f"Unknown method for rank parsing: {args.method}")
                return

        except ValueError as e:
            logger.error(
                f"Invalid rank argument. Ranks must be comma-separated integers. Error: {e}"
            )
            return

        (
            weights,
            factors,
            row_names,
            col_names,
            feature_names,
            X_raw,
            M_raw,
            mus,
            sds,
        ) = fit(
            args.method,
            df,
            args.features,
            ranks=rank_ints,
            n_iter=args.max_iter,
            tol=args.tol,
        )
        logger.info("Data clustering completed successfully.")

        factor_names = [
            f.strip() for f in args.factor_names.split(",") if f.strip()
        ]
        logger.info(f"Factor names: {factor_names}")
        threshold = args.threshold
        logger.info(f"Threshold: {threshold}")
        threshold_k_assignments = get_threshold_k_assignments(
            factors, factor_names, threshold=threshold
        )

        output_assignments = {
            f"threshold_{threshold*100:.0f}pct_assignments": threshold_k_assignments,
        }
        name_map = {
            "Store": row_names,
            "SKU": col_names,
            "Feature": feature_names,
        }
        assignment_df = create_assignments(
            output_assignments,
            name_map,
        )
        logger.info("Building model dictionary...")
        model_dict = {
            # Move tensors to CPU for max portability
            "weights": weights.cpu(),
            "factors": [f.cpu() for f in factors],
            "row_names": row_names,
            "col_names": col_names,
            "feature_names": feature_names,
            "X_raw": X_raw,
            "M_raw": M_raw,
            "method": args.method,
            "ranks": rank_ints,
            "mus": mus,
            "sds": sds,
            "assignments": assignment_df,
        }
        model_fn = Path(args.model_fn).resolve()
        logger.info(f"Saving model to {model_fn}")
        torch.save(model_dict, model_fn)

    except Exception as e:
        logger.error(f"Error clustering data: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
