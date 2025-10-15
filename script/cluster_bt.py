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

from src.BTNMF_util import cluster_data_and_explain_blocks
from src.utils import (
    setup_logging,
    read_csv_or_parquet,
    get_logger,
    str2bool,
    parse_range,
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
        "--row_range",
        type=parse_range,
        default=range(2, 5),
        help="Range of rows: START:END or explicit list a,b,c",
    )
    parser.add_argument(
        "--col_range",
        type=parse_range,
        default=range(2, 5),
        help="Range of columns: START:END or explicit list a,b,c",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-2,
        help="Alpha parameter for BT-NMF",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.6,
        help="Beta parameter for BT-NMF",
    )
    parser.add_argument(
        "--block_l1",
        type=float,
        default=0.0,
        help="Block l1 parameter for BT-NMF",
    )
    parser.add_argument(
        "--b_inner",
        type=int,
        default=15,
        help="B inner parameter for BT-NMF",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=200,
        help="Maximum number of iterations",
    )
    parser.add_argument(
        "--k_row",
        type=int,
        default=1,
        help="K row for BT-NMF",
    )
    parser.add_argument(
        "--k_col",
        type=int,
        default=1,
        help="K col for BT-NMF",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience for BT-NMF",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="Tolerance for convergence",
    )
    parser.add_argument(
        "--max_pve_drop",
        type=float,
        default=0.01,
        help="Max PVE drop for BT-NMF",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top k for BT-NMF",
    )
    parser.add_argument(
        "--top_rank_fn",
        type=str,
        default="",
        help="Path to top rank file (relative to project root)",
    )
    parser.add_argument(
        "--summary_fn",
        type=str,
        default="",
        help="Path to summary file (relative to project root)",
    )
    parser.add_argument(
        "--output_fn",
        type=str,
        default="",
        help="Path to output file (relative to project root)",
    )
    parser.add_argument(
        "--model_fn",
        type=str,
        default="",
        help="Path to model file (relative to project root)",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel processes to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for multiprocessing",
    )
    parser.add_argument(
        "--log_fn",
        type=str,
        default="",
        help="Path to save script outputs (relative to project root)",
    )
    parser.add_argument(
        "--empty_cluster_penalty",
        type=float,
        default=0.0,
        help="Penalty for empty clusters",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=2,
        help="Minimum cluster size",
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
        logger.info("Starting data clustering with configuration:")
        logger.info(f"  Data fn: {args.data_fn}")
        logger.info(f"  Row range: {args.row_range}")
        logger.info(f"  Col range: {args.col_range}")
        logger.info(f"  Log level: {args.log_level}")
        logger.info(f"  Alpha: {args.alpha}")
        logger.info(f"  Beta: {args.beta}")
        logger.info(f"  Block l1: {args.block_l1}")
        logger.info(f"  B inner: {args.b_inner}")
        logger.info(f"  Max iter: {args.max_iter}")
        logger.info(f"  K row: {args.k_row}")
        logger.info(f"  K col: {args.k_col}")
        logger.info(f"  Tolerance: {args.tol}")
        logger.info(f"  Max PVE drop: {args.max_pve_drop}")
        logger.info(f"  Empty cluster penalty: {args.empty_cluster_penalty}")
        logger.info(f"  Min cluster size: {args.min_cluster_size}")
        logger.info(f"  Patience: {args.patience}")
        logger.info(f"  Top k: {args.top_k}")
        logger.info(f"  Top rank fn: {args.top_rank_fn}")
        logger.info(f"  N jobs: {args.n_jobs}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Summary fn: {args.summary_fn}")
        logger.info(f"  Model fn: {args.model_fn}")
        logger.info(f"  Log level: {args.log_level}")

        data_fn = Path(args.data_fn).resolve()

        # Load and preprocess data
        df = read_csv_or_parquet(data_fn)
        top_rank_fn = Path(args.top_rank_fn).resolve()
        summary_fn = Path(args.summary_fn).resolve()
        model_fn = Path(args.model_fn).resolve()

        cluster_data_and_explain_blocks(
            df,
            row_range=args.row_range,
            col_range=args.col_range,
            alpha=args.alpha,
            beta=args.beta,
            block_l1=args.block_l1,
            b_inner=args.b_inner,
            max_iter=args.max_iter,
            k_row=args.k_row,
            k_col=args.k_col,
            patience=args.patience,
            tol=args.tol,
            max_pve_drop=args.max_pve_drop,
            top_k=args.top_k,
            top_rank_fn=top_rank_fn,
            summary_fn=summary_fn,
            model_fn=model_fn,
            n_jobs=args.n_jobs,
            batch_size=args.batch_size,
            empty_cluster_penalty=args.empty_cluster_penalty,
            min_cluster_size=args.min_cluster_size,
        )
        logger.info("Data clustering completed successfully")
    except Exception as e:
        logger.error(f"Error clustering data: {e}")
        raise


if __name__ == "__main__":
    main()
