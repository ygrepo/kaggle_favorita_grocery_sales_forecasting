from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pathlib import Path


def compare_metrics_by_model(
    paths: Dict[str, Path],
    metrics: Optional[List[str]] = None,
    model_filters: Optional[Dict[str, List[str]]] = None,
    output_path: Optional[Path] = None,
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Compare metrics by Model across multiple experiment settings.

    Args:
        paths: dict mapping setting_name -> Path to CSV file.
        metrics: list of metric columns to aggregate.
                 Default = ["RMSSE", "SMAPE", "MARRE"].
        model_filters: optional dict mapping setting_name -> list of models
                       to keep for that file (e.g., {"dl_future": ["TSMIXER"]})
        output_path: optional path to save output CSV.
        agg: aggregation function ("mean", "median", etc.)

    Returns:
        DataFrame with columns: Setting, Model, <metrics...>
    """

    if metrics is None:
        metrics = ["RMSSE", "SMAPE", "MARRE"]

    # Load all CSVs
    dfs = {}
    for key, path in paths.items():
        df = pd.read_csv(path)

        # Apply model filtering if provided
        if model_filters and key in model_filters:
            df = df[df["Model"].isin(model_filters[key])]

        dfs[key] = df

    # Aggregate by Model inside each setting
    summary_rows = []
    for setting, df in dfs.items():

        # Handle empty files gracefully
        if df.empty:
            continue

        grouped = df.groupby("Model")[metrics]

        if agg == "mean":
            agg_df = grouped.mean()
        elif agg == "median":
            agg_df = grouped.median()
        else:
            raise ValueError(f"Unsupported agg function: {agg}")

        # Convert to rows with Setting column
        agg_df = agg_df.reset_index()
        agg_df.insert(0, "Setting", setting)

        summary_rows.append(agg_df)

    # Combine all settings
    comparison_df = pd.concat(summary_rows, ignore_index=True)
    print(comparison_df)

    # Save if requested
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(output_path, index=False)
        print(f"Comparison saved to {output_path}")

    return comparison_df


def main():

    paths = {
        "ml": Path(
            "output/metrics/20251128_2013_2014_store_2000_item_cyc_features_ml_metrics.csv"
        ),
        "dl_no_covs": Path(
            "output/metrics/20251125_2013_2014_store_2000_item_cyc_features_dl_no_covs_metrics.csv"
        ),
        "dl_past_future": Path(
            "output/metrics/20251129_2013_2014_store_2000_item_cyc_features_dl_past_future_covs_metrics.csv"
        ),
        "dl_future": Path(
            "output/metrics/20251129_2013_2014_store_2000_item_cyc_features_dl_future_covs_metrics.csv"
        ),
        "dl_past": Path(
            "output/metrics/20251129_2013_2014_store_2000_item_cyc_features_dl_past_covs_metrics.csv"
        ),
    }

    model_filters = {
        # "dl_no_covs": ["TFT"],
        # "dl_past_future": ["BLOCK_RNN"],
        # "dl_future": ["TSMixer"],
        # "dl_past": ["BLOCK_RNN"],
    }
    output_path = Path(
        "output/metrics/20251129_compare_with_dl_time_series_models.csv"
    )

    compare_metrics_by_model(
        paths, model_filters=model_filters, output_path=output_path
    )


if __name__ == "__main__":
    main()
