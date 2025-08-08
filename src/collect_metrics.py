#!/usr/bin/env python3
"""
Collect and concatenate Lightning metrics.csv files into a single CSV.

Given a logs root like:
  output/logs/model_logger_2014_2015_top_53_store_2000_item/
which contains per-model folders like:
  7_13_ResidualMLP_7_13/version_0/metrics.csv

This script will recursively find all files named `metrics.csv`, prepend a
`model_name` column derived from the folder name just above the `version_*`
folder (e.g., `7_13_ResidualMLP_7_13`), and write a combined CSV.

Usage:
  python -m src.collect_metrics \
    --root output/logs/model_logger_2014_2015_top_53_store_2000_item \
    --out  output/logs/model_logger_2014_2015_top_53_store_2000_item/combined_metrics.csv

Notes:
- The script preserves the order of columns from the first encountered metrics.csv.
  If later files contain additional columns, they are appended to the end.
- Non-existent or empty metrics files are skipped.
- The output header will be: [model_name, <metrics columns...>]
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from collections import OrderedDict
from typing import List, Dict, Iterable

import logging

# Set up logger
logger = logging.getLogger(__name__)


def infer_model_name(metrics_path: Path) -> str:
    """Infer model name from a metrics.csv path.

    Expected layout: <root>/<model_name>/version_*/metrics.csv
    If immediate parent is "version_*", return the parent of that; otherwise
    return the immediate parent name.
    """
    parent = metrics_path.parent
    if parent.name.startswith("version_"):
        return parent.parent.name
    return parent.name


def iter_metrics_files(root: Path) -> Iterable[Path]:
    """Yield all metrics.csv files under root recursively."""
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn == "metrics.csv":
                yield Path(dirpath) / fn


def collect_metrics(root: Path) -> tuple[List[Dict[str, str]], List[str]]:
    """Collect rows from all metrics.csv under root and return (rows, fieldnames).

    - Prepends model_name to each row (value inferred from path).
    - Unions fieldnames across files preserving the order of the first file.
    """
    all_rows: List[Dict[str, str]] = []
    field_order = OrderedDict()  # type: OrderedDict[str, None]

    files = list(iter_metrics_files(root))
    if not files:
        logger.info(f"No metrics.csv files found under: {root}")
        return [], []

    logger.info(f"Found {len(files)} metrics.csv files under: {root}")

    for fp in sorted(files):
        model_name = infer_model_name(fp)
        try:
            with fp.open("r", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    logger.info(f"Skipping (no header): {fp}")
                    continue

                # Update global field order (preserve first-seen order)
                for col in reader.fieldnames:
                    if col not in field_order and col != "model_name":
                        field_order[col] = None

                rows_added = 0
                for row in reader:
                    if not row:  # skip empty lines
                        continue
                    row_out = {**row}
                    row_out["model_name"] = model_name
                    all_rows.append(row_out)
                    rows_added += 1

                logger.info(f"  + {rows_added:6d} rows from {fp}")
        except Exception as e:
            logger.info(f"Failed to read {fp}: {e}")

    # Compose final header: model_name first, then the collected fields
    header = ["model_name"] + list(field_order.keys())
    return all_rows, header


def write_combined_csv(
    rows: List[Dict[str, str]], header: List[str], out_path: Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logger.info(f"Wrote {len(rows)} rows to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Lightning metrics.csv files")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing per-model folders with version_*/metrics.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=False,
        help="Output CSV path (default: <root>/combined_metrics.csv)",
    )
    args = parser.parse_args()

    root: Path = args.root
    out: Path = args.out if args.out else (root / "combined_metrics.csv")

    rows, header = collect_metrics(root)
    if not rows:
        # still write empty file with header if any
        if header:
            write_combined_csv([], header, out)
        else:
            logger.info("Nothing to write.")
        return

    write_combined_csv(rows, header, out)


if __name__ == "__main__":
    main()
