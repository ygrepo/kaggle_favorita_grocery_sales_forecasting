import heapq
import numpy as np
import pandas as pd


# def generate_cyclical_features(df: pd.DataFrame, window_size: int = 7) -> pd.DataFrame:
#     """
#     Generate cyclical features for day of week and week of month.
#     Pads partial windows with 0s and handles empty input gracefully.

#     Args:
#         df: Input DataFrame with 'date', 'store', 'item' columns (optionally 'store_item')
#         window_size: Number of days in each window (including last partial)

#     Returns:
#         A DataFrame with cyclical features for each window
#     """
#     df = df.copy()

#     if df.empty:
#         cols = ["start_date", "store_item", "store", "item"]
#         for i in range(1, window_size + 1):
#             for feature in ["dayofweek", "weekofmonth"]:
#                 for trig in ["sin", "cos"]:
#                     cols.append(f"{feature}_{trig}_{i}")
#         return pd.DataFrame(columns=cols)

#     df["date"] = pd.to_datetime(df["date"])

#     # If store_item not provided, create it
#     if "store_item" not in df.columns:
#         df["store_item"] = df["store"].astype(str) + "_" + df["item"].astype(str)

#     # Cyclical features
#     df["dayofweek"] = df["date"].dt.dayofweek
#     df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
#     df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

#     df["weekofmonth"] = df["date"].apply(lambda d: (d.day - 1) // 7 + 1)
#     df["weekofmonth_sin"] = np.sin(2 * np.pi * df["weekofmonth"] / 5)
#     df["weekofmonth_cos"] = np.cos(2 * np.pi * df["weekofmonth"] / 5)

#     results = []

#     for store_item, group in df.groupby("store_item"):
#         group = group.sort_values("date").reset_index(drop=True)
#         total_rows = len(group)
#         num_windows = (total_rows + window_size - 1) // window_size  # ceil division

#         for w in range(num_windows):
#             start_idx = w * window_size
#             window_df = group.iloc[start_idx : start_idx + window_size]
#             row = {
#                 "start_date": window_df["date"].iloc[0],
#                 "store_item": store_item,
#                 "store": window_df["store"].iloc[0],
#                 "item": window_df["item"].iloc[0],
#             }

#             for day_idx in range(window_size):
#                 if day_idx < len(window_df):
#                     day_row = window_df.iloc[day_idx]
#                     for feature in ["dayofweek", "weekofmonth"]:
#                         for trig in ["sin", "cos"]:
#                             row[f"{feature}_{trig}_{day_idx + 1}"] = day_row[
#                                 f"{feature}_{trig}"
#                             ]
#                 else:
#                     # Pad with 0
#                     for feature in ["dayofweek", "weekofmonth"]:
#                         for trig in ["sin", "cos"]:
#                             row[f"{feature}_{trig}_{day_idx + 1}"] = 0.0

#             results.append(row)

#     # Ensure consistent column ordering
#     cols = ["start_date", "store_item", "store", "item"]
#     for i in range(1, window_size + 1):
#         for feature in ["dayofweek", "weekofmonth"]:
#             for trig in ["sin", "cos"]:
#                 cols.append(f"{feature}_{trig}_{i}")

#     return pd.DataFrame(results, columns=cols)


# def generate_nonoverlap_window_features(
#     df: pd.DataFrame, window_size: int = 5
# ) -> pd.DataFrame:
#     """
#     Splits the dates in train_df into non-overlapping windows of length `window_size`,
#     then for each (store, item) within each window computes:
#       - total sales on each day
#       - median sales per store on each day
#       - median sales per item on each day

#     Returns a DataFrame with columns organized in this order:
#       - store_item
#       - store
#       - item
#       - start_date
#       - sales_day_1 ... sales_day_{window_size}
#       - store_med_day_1 ... store_med_day_{window_size}
#       - item_med_day_1 ... item_med_day_{window_size}
#     """
#     # 1) Ensure datetime
#     df = df.copy()
#     df["date"] = pd.to_datetime(df["date"])

#     # 2) Build non-overlapping windows (including partial window)
#     unique_dates = df["date"].sort_values().unique()
#     chunked_windows = [
#         unique_dates[i : i + window_size]
#         for i in range(0, len(unique_dates), window_size)
#     ]

#     records = []
#     for window_dates in chunked_windows:
#         w_df = df[df["date"].isin(window_dates)]

#         # precompute medians & sums
#         store_med = (
#             w_df.groupby(["store", "date"])["unit_sales"].median().unstack(fill_value=0)
#         )
#         item_med = (
#             w_df.groupby(["item", "date"])["unit_sales"].median().unstack(fill_value=0)
#         )
#         sales = (
#             w_df.groupby(["store", "item", "date"])["unit_sales"]
#             .sum()
#             .unstack(fill_value=0)
#         )

#         for (store, item), sales_vals in sales.iterrows():
#             row = {
#                 "store_item": f"{store}_{item}",
#                 "store": store,
#                 "item": item,
#                 "start_date": window_dates[0],
#             }

#             # sales_day_i
#             for i in range(1, window_size + 1):
#                 try:
#                     d = window_dates[i - 1]
#                     row[f"sales_day_{i}"] = sales_vals.get(d, 0)
#                 except IndexError:
#                     row[f"sales_day_{i}"] = 0

#             # store_med_day_i
#             sm = (
#                 store_med.loc[store]
#                 if store in store_med.index
#                 else pd.Series(0, index=window_dates)
#             )
#             for i in range(1, window_size + 1):
#                 try:
#                     d = window_dates[i - 1]
#                     row[f"store_med_day_{i}"] = sm.get(d, 0)
#                 except IndexError:
#                     row[f"store_med_day_{i}"] = 0

#             # item_med_day_i
#             im = (
#                 item_med.loc[item]
#                 if item in item_med.index
#                 else pd.Series(0, index=window_dates)
#             )
#             for i in range(1, window_size + 1):
#                 try:
#                     d = window_dates[i - 1]
#                     row[f"item_med_day_{i}"] = im.get(d, 0)
#                 except IndexError:
#                     row[f"item_med_day_{i}"] = 0

#             records.append(row)

#     # Assemble DataFrame
#     cols = ["start_date", "store_item", "store", "item"]
#     sales_cols = [f"sales_day_{i}" for i in range(1, window_size + 1)]
#     store_med_cols = [f"store_med_day_{i}" for i in range(1, window_size + 1)]
#     item_med_cols = [f"item_med_day_{i}" for i in range(1, window_size + 1)]

#     cols.extend(sales_cols + store_med_cols + item_med_cols)

#     if not records:
#         return pd.DataFrame(columns=cols)

#     df = pd.DataFrame(records)
#     return df[cols]

# import pandas as pd
# import numpy as np


def generate_aligned_windows(df, window_size):
    """
    Returns a list of aligned non-overlapping windows (including partial at end),
    shared by both feature functions.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    unique_dates = np.sort(df["date"].unique())
    return [
        unique_dates[i : i + window_size]
        for i in range(0, len(unique_dates), window_size)
    ]


def generate_cyclical_features(df: pd.DataFrame, window_size: int = 7) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if "store_item" not in df.columns:
        df["store_item"] = df["store"].astype(str) + "_" + df["item"].astype(str)

    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["weekofmonth"] = df["date"].apply(lambda d: (d.day - 1) // 7 + 1)
    df["weekofmonth_sin"] = np.sin(2 * np.pi * df["weekofmonth"] / 5)
    df["weekofmonth_cos"] = np.cos(2 * np.pi * df["weekofmonth"] / 5)

    windows = generate_aligned_windows(df, window_size)
    results = []

    for store_item, group in df.groupby("store_item"):
        group = group.sort_values("date").reset_index(drop=True)

        for window_dates in windows:
            window_df = group[group["date"].isin(window_dates)]
            if window_df.empty:
                continue

            row = {
                "start_date": window_dates[0],
                "store_item": store_item,
                "store": window_df["store"].iloc[0],
                "item": window_df["item"].iloc[0],
            }

            for i in range(window_size):
                if i < len(window_df):
                    r = window_df.iloc[i]
                    for f in ["dayofweek", "weekofmonth"]:
                        for t in ["sin", "cos"]:
                            row[f"{f}_{t}_{i+1}"] = r[f"{f}_{t}"]
                else:
                    for f in ["dayofweek", "weekofmonth"]:
                        for t in ["sin", "cos"]:
                            row[f"{f}_{t}_{i+1}"] = 0.0

            results.append(row)

    columns = ["start_date", "store_item", "store", "item"] + [
        f"{f}_{t}_{i}"
        for i in range(1, window_size + 1)
        for f in ["dayofweek", "weekofmonth"]
        for t in ["sin", "cos"]
    ]

    return pd.DataFrame(results, columns=columns)


def generate_nonoverlap_window_features(
    df: pd.DataFrame, window_size: int = 5
) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    windows = generate_aligned_windows(df, window_size)
    records = []

    for window_dates in windows:
        w_df = df[df["date"].isin(window_dates)]

        store_med = (
            w_df.groupby(["store", "date"])["unit_sales"].median().unstack(fill_value=0)
        )
        item_med = (
            w_df.groupby(["item", "date"])["unit_sales"].median().unstack(fill_value=0)
        )
        sales = (
            w_df.groupby(["store", "item", "date"])["unit_sales"]
            .sum()
            .unstack(fill_value=0)
        )

        for (store, item), sales_vals in sales.iterrows():
            row = {
                "store_item": f"{store}_{item}",
                "store": store,
                "item": item,
                "start_date": window_dates[0],
            }

            for i in range(1, window_size + 1):
                try:
                    d = window_dates[i - 1]
                    row[f"sales_day_{i}"] = sales_vals.get(d, 0)
                    row[f"store_med_day_{i}"] = (
                        store_med.loc[store].get(d, 0)
                        if store in store_med.index
                        else 0
                    )
                    row[f"item_med_day_{i}"] = (
                        item_med.loc[item].get(d, 0) if item in item_med.index else 0
                    )
                except IndexError:
                    row[f"sales_day_{i}"] = 0
                    row[f"store_med_day_{i}"] = 0
                    row[f"item_med_day_{i}"] = 0

            records.append(row)

    cols = ["start_date", "store_item", "store", "item"] + [
        f"{prefix}_day_{i}"
        for i in range(1, window_size + 1)
        for prefix in ["sales", "store_med", "item_med"]
    ]

    return pd.DataFrame(records, columns=cols)


# def generate_nonoverlap_window_features(
#     df: pd.DataFrame, window_size: int = 5
# ) -> pd.DataFrame:
#     """
#     Splits the dates in train_df into non-overlapping windows of length `window_size`,
#     then for each (store, item) within each window computes:
#       - total sales on each day
#       - median sales per store on each day
#       - median sales per item on each day
#       - day_sin and day_cos features to capture cyclical patterns
#       - seasonal sales patterns (means, medians, ratios)

#     Returns a DataFrame with columns:
#       - id = '{store}_{item}_{window_start:%Y-%m-%d}'
#       - sales_day_1 ... sales_day_{window_size}
#       - store_med_day_1 ... store_med_day_{window_size}
#       - item_med_day_1 ... item_med_day_{window_size}
#       - yearly_sin_1 ... yearly_sin_{window_size}
#       - yearly_cos_1 ... yearly_cos_{window_size}
#       - quarterly_sin_1 ... quarterly_sin_{window_size}
#       - quarterly_cos_1 ... quarterly_cos_{window_size}
#       - seasonal_sin_1 ... seasonal_sin_{window_size}
#       - seasonal_cos_1 ... seasonal_cos_{window_size}
#       - monthly_sin_1 ... monthly_sin_{window_size}
#       - monthly_cos_1 ... monthly_cos_{window_size}
#       - weekly_sin_1 ... weekly_sin_{window_size}
#       - weekly_cos_1 ... weekly_cos_{window_size}
#       - weekly_sales_sin_1 ... weekly_sales_sin_{window_size}
#       - weekly_sales_cos_1 ... weekly_sales_cos_{window_size}
#       - monthly_sales_sin_1 ... monthly_sales_sin_{window_size}
#       - monthly_sales_cos_1 ... monthly_sales_cos_{window_size}
#       - quarterly_sales_sin_1 ... quarterly_sales_sin_{window_size}
#       - quarterly_sales_cos_1 ... quarterly_sales_cos_{window_size}
#       - yearly_sales_sin_1 ... yearly_sales_sin_{window_size}
#       - yearly_sales_cos_1 ... yearly_sales_cos_{window_size}
#       - seasonal_mean_1 ... seasonal_mean_{window_size}
#       - seasonal_median_1 ... seasonal_median_{window_size}
#       - seasonal_ratio_1 ... seasonal_ratio_{window_size}
#       - seasonal_rolling_mean_1 ... seasonal_rolling_mean_{window_size}
#     """
#     # 1) Ensure datetime
#     df = df.copy()
#     df["date"] = pd.to_datetime(df["date"])

#     # 2) Build non-overlapping windows
#     unique_dates = df["date"].sort_values().unique()
#     chunked_windows = [
#         unique_dates[i : i + window_size]
#         for i in range(0, len(unique_dates), window_size)
#         if len(unique_dates[i : i + window_size]) == window_size
#     ]

#     records = []
#     for window_dates in chunked_windows:
#         window_start = pd.to_datetime(window_dates[0])
#         window_str = window_start.strftime("%Y-%m-%d")

#         # subset to this window
#         w_df = df[df["date"].isin(window_dates)]

#         # precompute medians & sums
#         store_med = (
#             w_df.groupby(["store", "date"])["unit_sales"].median().unstack(fill_value=0)
#         )
#         item_med = (
#             w_df.groupby(["item", "date"])["unit_sales"].median().unstack(fill_value=0)
#         )
#         sales = (
#             w_df.groupby(["store", "item", "date"])["unit_sales"]
#             .sum()
#             .unstack(fill_value=0)
#         )

#         # Calculate sales cycle features
#         # First, get the full history for each store_item
#         store_item_history = df.groupby(["store", "item"])
#         for (store, item), group in store_item_history:
#             # Calculate sales cycle features for this store_item
#             if (store, item) not in sales.index:
#                 continue

#             # Get the sales history for this store_item
#             sales_history = group["unit_sales"]

#             # Calculate seasonal sales patterns
#             # Create seasonal groups (Spring: 3-5, Summer: 6-8, Fall: 9-11, Winter: 12,1,2)
#             group['season'] = pd.cut(
#                 group['date'].dt.month,
#                 bins=[0, 3, 6, 9, 12],
#                 labels=['Winter', 'Spring', 'Summer', 'Fall'],
#                 right=False
#             )

#             # Calculate seasonal means
#             seasonal_means = group.groupby("season")["unit_sales"].mean()
#             seasonal_medians = group.groupby("season")["unit_sales"].median()

#             # Calculate seasonal ratios
#             seasonal_ratios = group["unit_sales"] / group["season"].map(seasonal_means)
#             seasonal_ratios = seasonal_ratios.fillna(1).replace([np.inf, -np.inf], 1)

#             # Calculate weekly sales cycle (7 days)
#             weekly_sales = sales_history.rolling(7).mean()
#             weekly_sales_cycle = weekly_sales / sales_history
#             weekly_sales_radians = (weekly_sales_cycle * 2 * np.pi).fillna(0)
#             sales.loc[(store, item), "weekly_sales_sin"] = np.sin(weekly_sales_radians.values)
#             sales.loc[(store, item), "weekly_sales_cos"] = np.cos(weekly_sales_radians.values)

#             # Calculate monthly sales cycle (30 days)
#             monthly_sales = sales_history.rolling(30).mean()
#             monthly_sales_cycle = monthly_sales / sales_history
#             monthly_sales_radians = (monthly_sales_cycle * 2 * np.pi).fillna(0)
#             sales.loc[(store, item), "monthly_sales_sin"] = np.sin(monthly_sales_radians.values)
#             sales.loc[(store, item), "monthly_sales_cos"] = np.cos(monthly_sales_radians.values)

#             # Calculate quarterly sales cycle (90 days)
#             quarterly_sales = sales_history.rolling(90).mean()
#             quarterly_sales_cycle = quarterly_sales / sales_history
#             quarterly_sales_radians = (quarterly_sales_cycle * 2 * np.pi).fillna(0)
#             sales.loc[(store, item), "quarterly_sales_sin"] = np.sin(quarterly_sales_radians.values)
#             sales.loc[(store, item), "quarterly_sales_cos"] = np.cos(quarterly_sales_radians.values)

#             # Calculate yearly sales cycle (365 days)
#             yearly_sales = sales_history.rolling(365).mean()
#             yearly_sales_cycle = yearly_sales / sales_history
#             yearly_sales_radians = (yearly_sales_cycle * 2 * np.pi).fillna(0)
#             sales.loc[(store, item), "yearly_sales_sin"] = np.sin(yearly_sales_radians.values)
#             sales.loc[(store, item), "yearly_sales_cos"] = np.cos(yearly_sales_radians.values)

#             # Normalize cycles to be between 0 and 1
#             weekly_sales_cycle = (weekly_sales_cycle - weekly_sales_cycle.min()) / (
#                 weekly_sales_cycle.max() - weekly_sales_cycle.min()
#             )
#             monthly_sales_cycle = (monthly_sales_cycle - monthly_sales_cycle.min()) / (
#                 monthly_sales_cycle.max() - monthly_sales_cycle.min()
#             )
#             quarterly_sales_cycle = (
#                 quarterly_sales_cycle - quarterly_sales_cycle.min()
#             ) / (quarterly_sales_cycle.max() - quarterly_sales_cycle.min())
#             yearly_sales_cycle = (yearly_sales_cycle - yearly_sales_cycle.min()) / (
#                 yearly_sales_cycle.max() - yearly_sales_cycle.min()
#             )

#             # Convert to radians (0 to 2pi)
#             weekly_radians = weekly_sales_cycle * 2 * np.pi
#             monthly_radians = monthly_sales_cycle * 2 * np.pi
#             quarterly_radians = quarterly_sales_cycle * 2 * np.pi
#             yearly_radians = yearly_sales_cycle * 2 * np.pi

#             # Store the cycles and seasonal features in the sales DataFrame
#             sales.loc[(store, item), "weekly_sales_sin"] = np.sin(weekly_radians)
#             sales.loc[(store, item), "weekly_sales_cos"] = np.cos(weekly_radians)
#             sales.loc[(store, item), "monthly_sales_sin"] = np.sin(monthly_radians)
#             sales.loc[(store, item), "monthly_sales_cos"] = np.cos(monthly_radians)
#             sales.loc[(store, item), "quarterly_sales_sin"] = np.sin(quarterly_radians)
#             sales.loc[(store, item), "quarterly_sales_cos"] = np.cos(quarterly_radians)
#             sales.loc[(store, item), "yearly_sales_sin"] = np.sin(yearly_radians)
#             sales.loc[(store, item), "yearly_sales_cos"] = np.cos(yearly_radians)

#             # Store seasonal features
#             sales.loc[(store, item), "seasonal_mean"] = seasonal_means
#             sales.loc[(store, item), "seasonal_median"] = seasonal_medians
#             sales.loc[(store, item), "seasonal_ratio"] = seasonal_ratios
#             sales.loc[(store, item), "seasonal_rolling_mean"] = seasonal_rolling

#         for (store, item), sales_vals in sales.iterrows():
#             row = {"id": f"{store}_{item}_{window_str}"}

#             # Add multiple cyclical features
#             for i, d in enumerate(window_dates, start=1):
#                 # Yearly cycle (day of year)
#                 day_of_year = d.timetuple().tm_yday
#                 yearly_radians = (day_of_year - 1) * (2 * np.pi / 365)
#                 row[f"yearly_sin_{i}"] = np.sin(yearly_radians)
#                 row[f"yearly_cos_{i}"] = np.cos(yearly_radians)

#                 # Quarterly cycle (1-4)
#                 quarter = (d.month - 1) // 3 + 1
#                 quarterly_radians = (quarter - 1) * (2 * np.pi / 4)
#                 row[f"quarterly_sin_{i}"] = np.sin(quarterly_radians)
#                 row[f"quarterly_cos_{i}"] = np.cos(quarterly_radians)

#                 # Seasonal cycle (1-4, Spring, Summer, Fall, Winter)
#                 # Spring: March-May (3-5), Summer: June-Aug (6-8), Fall: Sep-Nov (9-11), Winter: Dec-Feb (12,1,2)
#                 month = d.month
#                 if month in [12, 1, 2]:  # Winter
#                     season = 4
#                 elif month in [3, 4, 5]:  # Spring
#                     season = 1
#                 elif month in [6, 7, 8]:  # Summer
#                     season = 2
#                 else:  # Fall
#                     season = 3
#                 seasonal_radians = (season - 1) * (2 * np.pi / 4)
#                 row[f"seasonal_sin_{i}"] = np.sin(seasonal_radians)
#                 row[f"seasonal_cos_{i}"] = np.cos(seasonal_radians)

#                 # Monthly cycle (1-12)
#                 monthly_radians = (d.month - 1) * (2 * np.pi / 12)
#                 row[f"monthly_sin_{i}"] = np.sin(monthly_radians)
#                 row[f"monthly_cos_{i}"] = np.cos(monthly_radians)

#                 # Day of week cycle (0-6)
#                 day_of_week = d.weekday()
#                 weekly_radians = day_of_week * (2 * np.pi / 7)
#                 row[f"weekly_sin_{i}"] = np.sin(weekly_radians)
#                 row[f"weekly_cos_{i}"] = np.cos(weekly_radians)

#             # sales_day_i and sales cycle features
#             for i, d in enumerate(window_dates, start=1):
#                 row[f"sales_day_{i}"] = sales_vals.get(d, 0)

#                 # Add sales cycle and seasonal features
#             if d in sales.columns:
#                 row[f"weekly_sales_sin_{i}"] = sales.loc[
#                     (store, item), "weekly_sales_sin", d
#                 ]
#                 row[f"weekly_sales_cos_{i}"] = sales.loc[
#                     (store, item), "weekly_sales_cos", d
#                 ]
#                 row[f"monthly_sales_sin_{i}"] = sales.loc[
#                     (store, item), "monthly_sales_sin", d
#                 ]
#                 row[f"monthly_sales_cos_{i}"] = sales.loc[
#                     (store, item), "monthly_sales_cos", d
#                 ]
#                 row[f"quarterly_sales_sin_{i}"] = sales.loc[
#                     (store, item), "quarterly_sales_sin", d
#                 ]
#                 row[f"quarterly_sales_cos_{i}"] = sales.loc[
#                     (store, item), "quarterly_sales_cos", d
#                 ]
#                 row[f"yearly_sales_sin_{i}"] = sales.loc[
#                     (store, item), "yearly_sales_sin", d
#                 ]
#                 row[f"yearly_sales_cos_{i}"] = sales.loc[
#                     (store, item), "yearly_sales_cos", d
#                 ]

#                 # Add seasonal features
#                 row[f"seasonal_mean_{i}"] = sales.loc[(store, item), "seasonal_mean", d]
#                 row[f"seasonal_median_{i}"] = sales.loc[
#                     (store, item), "seasonal_median", d
#                 ]
#                 row[f"seasonal_ratio_{i}"] = sales.loc[
#                     (store, item), "seasonal_ratio", d
#                 ]
#                 row[f"seasonal_rolling_mean_{i}"] = sales.loc[
#                     (store, item), "seasonal_rolling_mean", d
#                 ]

#             # store_med_day_i
#             if store in store_med.index:
#                 sm = store_med.loc[store]
#             else:
#                 sm = pd.Series(0, index=window_dates)
#             for i, d in enumerate(window_dates, start=1):
#                 row[f"store_med_day_{i}"] = sm.get(d, 0)

#             # item_med_day_i
#             if item in item_med.index:
#                 im = item_med.loc[item]
#             else:
#                 im = pd.Series(0, index=window_dates)
#             for i, d in enumerate(window_dates, start=1):
#                 row[f"item_med_day_{i}"] = im.get(d, 0)

#             records.append(row)

#     # Reorder columns to group related features together
#     columns = ["id", "store", "item"]

#     # Add sales and seasonal features first
#     for i in range(1, window_size + 1):
#         columns.extend(
#             [
#                 f"sales_day_{i}",
#                 f"yearly_sales_sin_{i}",
#                 f"yearly_sales_cos_{i}",
#                 f"quarterly_sales_sin_{i}",
#                 f"quarterly_sales_cos_{i}",
#                 f"monthly_sales_sin_{i}",
#                 f"monthly_sales_cos_{i}",
#                 f"weekly_sales_sin_{i}",
#                 f"weekly_sales_cos_{i}",
#                 f"seasonal_mean_{i}",
#                 f"seasonal_median_{i}",
#                 f"seasonal_ratio_{i}",
#                 f"seasonal_rolling_mean_{i}",
#             ]
#         )

#     # Add store and item median features
#     for i in range(1, window_size + 1):
#         columns.extend(
#             [
#                 f"store_med_day_{i}",
#                 f"item_med_day_{i}",
#             ]
#         )

#     # Add time-based cyclical features
#     for i in range(1, window_size + 1):
#         columns.extend(
#             [
#                 f"yearly_sin_{i}",
#                 f"yearly_cos_{i}",
#                 f"quarterly_sin_{i}",
#                 f"quarterly_cos_{i}",
#                 f"seasonal_sin_{i}",
#                 f"seasonal_cos_{i}",
#                 f"monthly_sin_{i}",
#                 f"monthly_cos_{i}",
#                 f"weekly_sin_{i}",
#                 f"weekly_cos_{i}",
#             ]
#         )

#     return pd.DataFrame.from_records(records)[columns]

#     return pd.DataFrame.from_records(records)


# Redefine RunningMedian class
class RunningMedian:
    def __init__(self):
        self.low = []  # max-heap
        self.high = []  # min-heap

    def add(self, num):
        heapq.heappush(self.low, -heapq.heappushpop(self.high, num))
        if len(self.low) > len(self.high):
            heapq.heappush(self.high, -heapq.heappop(self.low))

    def median(self) -> float:
        if self.low or self.high:
            if len(self.high) > len(self.low):
                return float(self.high[0])
            return (self.high[0] - self.low[0]) / 2.0
        return 0.0


class SmoothOnlineMedian:
    def __init__(self, s: float, eta: float, debug: bool = False):
        self.estimate = s
        self.eta = eta
        self.debug = debug
        if self.debug:
            self.history = []

    def update(self, x: float) -> float:
        delta = self.eta * self._sgn(x - self.estimate)
        self.estimate += delta
        if self.debug:
            self.history.append(self.estimate)
        return self.estimate

    def _sgn(self, value: float) -> int:
        return 1 if value > 0 else -1 if value < 0 else 0

    def get_estimate(self) -> float:
        return self.estimate

    def get_history(self):
        if self.debug:
            return self.history
        else:
            raise AttributeError(
                "History tracking is disabled. Enable debug mode to track history."
            )
