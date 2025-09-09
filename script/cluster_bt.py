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

from src.data_utils import load_raw_data
from src.BTNMF_util import cluster_data_and_explain_blocks
from src.utils import setup_logging, str2bool


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
        "--row_range",
        type=parse_range,
        default=range(2, 5),
        help="Range of number of rows to cluster (format: START:END)",
    )
    parser.add_argument(
        "--col_range",
        type=parse_range,
        default=range(2, 5),
        help="Range of number of columns to cluster (format: START:END)",
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
        "--min_sil",
        type=float,
        default=-0.05,
        help="Min Silhouette for BT-NMF",
    )
    parser.add_argument(
        "--min_keep",
        type=int,
        default=6,
        help="Min keep for BT-NMF",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top k for BT-NMF",
    )
    parser.add_argument(
        "--growth_rate_fn",
        type=str,
        default="",
        help="Path to growth rate file (relative to project root)",
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
        "--figure_fn",
        type=str,
        default="",
        help="Path to figure file (relative to project root)",
    )
    parser.add_argument(
        "--output_fn",
        type=str,
        default="",
        help="Path to output file (relative to project root)",
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
    log_dir = Path(args.log_dir).resolve()

    # Set up logging
    logger = setup_logging(log_dir, args.log_level)
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
        logger.info(f"  Tolerance: {args.tol}")
        logger.info(f"  Max PVE drop: {args.max_pve_drop}")
        logger.info(f"  Min Silhouette: {args.min_sil}")
        logger.info(f"  Min keep: {args.min_keep}")
        logger.info(f"  Top k: {args.top_k}")
        logger.info(f"  Growth rate fn: {args.growth_rate_fn}")
        logger.info(f"  Top rank fn: {args.top_rank_fn}")
        logger.info(f"  Summary fn: {args.summary_fn}")
        logger.info(f"  Figure fn: {args.figure_fn}")
        logger.info(f"  Output fn: {args.output_fn}")
        logger.info(f"  Log level: {args.log_level}")

        data_fn = Path(args.data_fn).resolve()

        # Load and preprocess data
        df = load_raw_data(data_fn)

        output_fn = Path(args.output_fn).resolve()
        figure_fn = Path(args.figure_fn).resolve()
        top_rank_fn = Path(args.top_rank_fn).resolve()
        summary_fn = Path(args.summary_fn).resolve()
        growth_rate_fn = Path(args.growth_rate_fn).resolve()
        cluster_data_and_explain_blocks(
            df,
            row_range=args.row_range,
            col_range=args.col_range,
            alpha=args.alpha,
            beta=args.beta,
            block_l1=args.block_l1,
            b_inner=args.b_inner,
            max_iter=args.max_iter,
            tol=args.tol,
            max_pve_drop=args.max_pve_drop,
            min_sil=args.min_sil,
            min_keep=args.min_keep,
            top_k=args.top_k,
            growth_rate_fn=growth_rate_fn,
            top_rank_fn=top_rank_fn,
            summary_fn=summary_fn,
            output_fn=output_fn,
            figure_fn=figure_fn,
            log_level=args.log_level,
        )
        logger.info("Data clustering completed successfully")
    except Exception as e:
        logger.error(f"Error clustering data: {e}")
        raise


if __name__ == "__main__":
    main()
