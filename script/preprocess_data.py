#!/usr/bin/env python3
"""
Data preprocessing script for the Favorita Grocery Sales Forecasting model.

This script:
- Loads raw Favorita train data
- Optionally filters rows by date and by extreme/median stores & items
- Merges transactions, holidays, store metadata, and item metadata
- Computes per-(store,item,date) growth_rate
- Writes the resulting panel to CSV or Parquet
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_preprocessing import select_extreme_and_median_neighbors
from src.utils import setup_logging, get_logger, save_csv_or_parquet

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Data preprocessing for Favorita Grocery Sales Forecasting model"
    )
    parser.add_argument(
        "--data_fn",
        type=str,
        default="data/train.csv",
        help="Path to training data file (relative to project root)",
    )
    parser.add_argument(
        "--item_fn",
        type=str,
        default="data/items.csv",
        help="Path to item metadata file (relative to project root)",
    )
    parser.add_argument(
        "--store_fn",
        type=str,
        default="data/stores.csv",
        help="Path to store metadata file (relative to project root)",
    )
    parser.add_argument(
        "--output_fn",
        type=str,
        default="",
        help="Path to output file (relative to project root)",
    )
    parser.add_argument(
        "--store_top_n",
        type=int,
        default=0,
        help="Number of top stores to retain (by unit_sales)",
    )
    parser.add_argument(
        "--store_med_n",
        type=int,
        default=0,
        help="Number of median stores to retain (neighbors around median)",
    )
    parser.add_argument(
        "--store_bottom_n",
        type=int,
        default=0,
        help="Number of bottom stores to retain (by unit_sales)",
    )
    parser.add_argument(
        "--item_top_n",
        type=int,
        default=0,
        help="Number of top items to retain (by unit_sales)",
    )
    parser.add_argument(
        "--item_med_n",
        type=int,
        default=0,
        help="Number of median items to retain (neighbors around median)",
    )
    parser.add_argument(
        "--item_bottom_n",
        type=int,
        default=0,
        help="Number of bottom items to retain (by unit_sales)",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=0,
        help="Number of rows to load from train.csv (0 = all)",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="",
        help="Lower bound date (inclusive), e.g. '2014-01-01'. Empty = no lower bound.",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="",
        help="Upper bound date (inclusive), e.g. '2016-08-15'. Empty = no upper bound.",
    )
    parser.add_argument(
        "--log_fn",
        type=str,
        default="",
        help="Path to save log file (relative to project root)",
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
    """Main preprocessing function."""
    args = parse_args()
    setup_logging(Path(args.log_fn), args.log_level)

    # Convert paths to absolute paths relative to project root
    data_fn = Path(args.data_fn).resolve()
    output_fn = (
        Path(args.output_fn).resolve()
        if args.output_fn
        else Path("preprocessed.csv").resolve()
    )
    item_fn = Path(args.item_fn).resolve()
    store_fn = Path(args.store_fn).resolve()

    try:
        # ------------------------------------------------------------------
        # Log configuration
        # ------------------------------------------------------------------
        logger.info("Starting data preprocessing with configuration:")
        logger.info(f"  Data fn: {data_fn}")
        logger.info(f"  Item fn: {item_fn}")
        logger.info(f"  Store fn: {store_fn}")
        logger.info(f"  Output fn: {output_fn}")
        logger.info(f"  Log fn: {args.log_fn}")
        logger.info(f"  Log level: {args.log_level}")
        logger.info(f"  Nrows: {args.nrows}")
        logger.info(f"  Start date: {args.start_date}")
        logger.info(f"  End date: {args.end_date}")
        logger.info(f"  Store top n: {args.store_top_n}")
        logger.info(f"  Store med n: {args.store_med_n}")
        logger.info(f"  Store bottom n: {args.store_bottom_n}")
        logger.info(f"  Item top n: {args.item_top_n}")
        logger.info(f"  Item med n: {args.item_med_n}")
        logger.info(f"  Item bottom n: {args.item_bottom_n}")

        # ------------------------------------------------------------------
        # Load core train data
        # ------------------------------------------------------------------
        read_kwargs = {"low_memory": False}
        if args.nrows > 0:
            read_kwargs["nrows"] = args.nrows

        data_df = pd.read_csv(data_fn, **read_kwargs)
        data_df.rename(
            columns={"store_nbr": "store", "item_nbr": "item"}, inplace=True
        )
        if "id" in data_df.columns:
            data_df.drop(["id"], axis=1, inplace=True)

        data_df["date"] = pd.to_datetime(data_df["date"])

        # Optional date filtering
        if args.start_date:
            start_date = pd.to_datetime(args.start_date)
            data_df = data_df[data_df["date"] >= start_date]

        if args.end_date:
            end_date = pd.to_datetime(args.end_date)
            data_df = data_df[data_df["date"] <= end_date]

        logger.info(f"Stores (initial): {data_df['store'].nunique()}")
        logger.info(f"Items (initial): {data_df['item'].nunique()}")
        logger.info(f"Rows after date filter: {len(data_df)}")

        # ------------------------------------------------------------------
        # Select subset of stores (top/median/bottom by unit_sales)
        # ------------------------------------------------------------------
        store_ids = select_extreme_and_median_neighbors(
            data_df,
            n_col="unit_sales",
            group_column="store",
            M=args.store_top_n,
            m=args.store_bottom_n,
            med=args.store_med_n,
        )
        if store_ids:
            data_df = data_df[data_df["store"].isin(store_ids)]
        logger.info(f"Stores after selection: {data_df['store'].nunique()}")

        # ------------------------------------------------------------------
        # Select subset of items (top/median/bottom by unit_sales)
        # ------------------------------------------------------------------
        item_ids = select_extreme_and_median_neighbors(
            data_df,
            n_col="unit_sales",
            group_column="item",
            M=args.item_top_n,
            m=args.item_bottom_n,
            med=args.item_med_n,
        )
        if item_ids:
            data_df = data_df[data_df["item"].isin(item_ids)]
        logger.info(f"Items after selection: {data_df['item'].nunique()}")

        # ------------------------------------------------------------------
        # Merge transactions (e.g., oil prices / transactions.csv)
        # ------------------------------------------------------------------
        trans_df = pd.read_csv("./data/transactions.csv")
        trans_df.rename(columns={"store_nbr": "store"}, inplace=True)
        trans_df["date"] = pd.to_datetime(trans_df["date"])
        data_df = data_df.merge(trans_df, on=["store", "date"], how="left")
        logger.info("Merged transactions")

        # ------------------------------------------------------------------
        # Load stores metadata (used for holidays + static features)
        # ------------------------------------------------------------------
        stores = pd.read_csv(store_fn)
        stores.rename(columns={"store_nbr": "store"}, inplace=True)
        logger.info(f"Store metadata rows: {len(stores)}")

        # ------------------------------------------------------------------
        # Holidays: build per-(store,date) holiday flags
        # ------------------------------------------------------------------
        hol = pd.read_csv("./data/holidays_events.csv")
        hol["date"] = pd.to_datetime(hol["date"])

        # Drop transferred rows (keep the effective holiday dates)
        if "transferred" in hol.columns:
            hol = hol[hol["transferred"] == False].copy()

        # National holidays -> all stores
        nat = hol[hol["locale"] == "National"].copy()
        if not nat.empty:
            nat["key"] = 1
            stores_tmp = stores[["store"]].assign(key=1)
            nat = nat.merge(stores_tmp, on="key").drop(columns="key")

        # Regional holidays -> stores in matching state/province
        reg = hol[hol["locale"] == "Regional"].merge(
            stores[["store", "state"]],
            left_on="locale_name",
            right_on="state",
            how="left",
        )

        # Local holidays -> stores in matching city
        loc = hol[hol["locale"] == "Local"].merge(
            stores[["store", "city"]],
            left_on="locale_name",
            right_on="city",
            how="left",
        )

        hol_store = pd.concat([nat, reg, loc], ignore_index=True)

        # Keep only what we need
        hol_store = hol_store[
            ["date", "store", "locale", "type"]
        ].drop_duplicates()

        # Encode holiday flags
        hol_store["is_holiday_any"] = 1
        hol_store["is_holiday_national"] = (
            hol_store["locale"] == "National"
        ).astype(int)
        hol_store["is_holiday_regional"] = (
            hol_store["locale"] == "Regional"
        ).astype(int)
        hol_store["is_holiday_local"] = (
            hol_store["locale"] == "Local"
        ).astype(int)

        # Merge holidays into main panel
        data_df = data_df.merge(
            hol_store[
                [
                    "date",
                    "store",
                    "is_holiday_any",
                    "is_holiday_national",
                    "is_holiday_regional",
                    "is_holiday_local",
                ]
            ],
            on=["store", "date"],
            how="left",
        )

        holiday_cols = [
            "is_holiday_any",
            "is_holiday_national",
            "is_holiday_regional",
            "is_holiday_local",
        ]
        data_df[holiday_cols] = data_df[holiday_cols].fillna(0).astype(int)
        logger.info("Merged holidays")

        # ------------------------------------------------------------------
        # Store static features (type, cluster, etc.)
        # ------------------------------------------------------------------
        # We already have `stores`; now encode type as one-hot
        stores_for_merge = stores.copy()
        # Drop raw city/state after holidays step if you do not need them downstream
        # (optional – keep if you still want them)
        # stores_for_merge = stores_for_merge.drop(columns=["city", "state"])

        if "type" in stores_for_merge.columns:
            type_encoded = pd.get_dummies(
                stores_for_merge["type"], prefix="type", drop_first=True
            ).astype(int)
            stores_for_merge = pd.concat(
                [stores_for_merge.drop("type", axis=1), type_encoded], axis=1
            )

        logger.info(
            f"Initial distinct stores in metadata: {stores_for_merge['store'].nunique()}"
        )

        data_df = data_df.merge(stores_for_merge, on="store", how="left")
        logger.info(
            f"Stores after metadata merge: {data_df['store'].nunique()}"
        )

        # ------------------------------------------------------------------
        # Item static features (family, class, perishable)
        # ------------------------------------------------------------------
        items = pd.read_csv(item_fn)
        items.rename(columns={"item_nbr": "item"}, inplace=True)
        items = items[["item", "family", "class", "perishable"]]

        data_df = data_df.merge(items, on="item", how="left")
        logger.info(f"Items after metadata merge: {data_df['item'].nunique()}")

        # ------------------------------------------------------------------
        # Create store_item identifier and growth_rate
        # ------------------------------------------------------------------
        store_item_values = (
            data_df["store"].astype(str) + "_" + data_df["item"].astype(str)
        )
        data_df.insert(1, "store_item", store_item_values)

        data_df.sort_values(["store_item", "date"], inplace=True)
        data_df.reset_index(drop=True, inplace=True)

        data_df["growth_rate"] = (
            data_df.groupby("store_item")["unit_sales"]
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        # ------------------------------------------------------------------
        # Save result
        # ------------------------------------------------------------------
        save_csv_or_parquet(data_df, output_fn)
        logger.info(
            f"Data preprocessing completed successfully. Output: {output_fn}"
        )

    except Exception as e:
        logger.error(f"Error creating data preprocessing features: {e}")
        raise


if __name__ == "__main__":
    main()
