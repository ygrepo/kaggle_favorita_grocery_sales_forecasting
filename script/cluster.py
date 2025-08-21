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
from src.cluster_util import cluster_data
from src.utils import setup_logging, str2bool
from sklearn.cluster import SpectralCoclustering


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
        "--item_fn",
        type=str,
        default="",
        help="Path to item file (relative to project root)",
    )
    parser.add_argument(
        "--store_fn",
        type=str,
        default="",
        help="Path to store file (relative to project root)",
    )
    parser.add_argument(
        "--store_item_matrix_fn",
        type=str,
        default="",
        help="Path to store item matrix file (relative to project root)",
    )
    parser.add_argument(
        "--output_fn",
        type=str,
        default="",
        help="Path to output file (relative to project root)",
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
        "--mav_output_fn",
        type=str,
        default="",
        help="Path to MAV output file (relative to project root)",
    )
    parser.add_argument(
        "--only_best_model_path",
        type=str,
        default="",
        help="Path to only best model output file (relative to project root)",
    )
    parser.add_argument(
        "--only_top_n_clusters_path",
        type=str,
        default="",
        help="Path to only top n clusters output file (relative to project root)",
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
        "--norm",
        type=str,
        default="mav_ratio",
        help="Norm to use",
    )
    parser.add_argument(
        "--only_best_model",
        type=bool,
        default=True,
        help="Whether to only save the best model's MAV scores",
    )
    parser.add_argument(
        "--only_top_n_clusters",
        type=int,
        default=None,
        help="Whether to only save the top n clusters' MAV scores",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=2,
        help="Minimum cluster size",
    )
    parser.add_argument(
        "--skip_invalid",
        type=bool,
        default=True,
        help="Whether to skip invalid clusters",
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
    # Convert paths to absolute paths relative to project root
    data_fn = Path(args.data_fn).resolve()
    item_fn = Path(args.item_fn).resolve()
    store_fn = Path(args.store_fn).resolve()
    if args.store_item_matrix_fn:
        store_item_matrix_fn = Path(args.store_item_matrix_fn).resolve()
    else:
        store_item_matrix_fn = None
    mav_output_fn = Path(args.mav_output_fn).resolve()
    only_best_model_path = Path(args.only_best_model_path).resolve()
    only_top_n_clusters_path = Path(args.only_top_n_clusters_path).resolve()
    output_fn = Path(args.output_fn).resolve()
    row_range = args.row_range
    col_range = args.col_range

    log_dir = Path(args.log_dir).resolve()

    # Set up logging
    logger = setup_logging(log_dir, args.log_level)
    try:
        # Log configuration
        logger.info("Starting data clustering with configuration:")
        logger.info(f"  Data fn: {data_fn}")
        logger.info(f"  Item fn: {item_fn}")
        logger.info(f"  Store fn: {store_fn}")
        logger.info(f"  Store item matrix fn: {store_item_matrix_fn}")
        logger.info(f"  MAV output fn: {mav_output_fn}")
        logger.info(f"  Only best model path: {only_best_model_path}")
        logger.info(f"  Only top n clusters path: {only_top_n_clusters_path}")
        logger.info(f"  Max iter: {args.max_iter}")
        logger.info(f"  Tolerance: {args.tol}")
        logger.info(f"  Norm: {args.norm}")
        logger.info(f"  Row range: {row_range}")
        logger.info(f"  Col range: {col_range}")
        logger.info(f"  Only best model: {args.only_best_model}")
        logger.info(f"  Only top n clusters: {args.only_top_n_clusters}")
        logger.info(f"  Min cluster size: {args.min_cluster_size}")
        logger.info(f"  Skip invalid: {args.skip_invalid}")
        logger.info(f"  Output fn: {output_fn}")

        # Load and preprocess data
        df = load_raw_data(data_fn)
        model_kwargs = {
            "max_iter": args.max_iter,
            "tol": args.tol,
            "random_state": 123,
            "norm": args.norm,
        }
        cluster_data(
            df,
            store_item_matrix_fn=store_item_matrix_fn,
            mav_df_fn=mav_output_fn,
            only_best_model_path=only_best_model_path,
            only_top_n_clusters_path=only_top_n_clusters_path,
            item_fn=item_fn,
            store_fn=store_fn,
            output_fn=output_fn,
            row_range=row_range,
            col_range=col_range,
            only_best_model=str2bool(args.only_best_model),
            only_top_n_clusters=args.only_top_n_clusters,
            min_cluster_size=args.min_cluster_size,
            skip_invalid=str2bool(args.skip_invalid),
            model_kwargs=model_kwargs,
            log_level=args.log_level,
        )
        logger.info("Data clustering completed successfully")
    except Exception as e:
        logger.error(f"Error clustering data: {e}")
        raise


if __name__ == "__main__":
    main()
