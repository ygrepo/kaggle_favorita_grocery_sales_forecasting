import heapq
import numpy as np
import pandas as pd


def generate_aligned_windows(df, window_size):
    """
    Returns a list of exact non-overlapping date-aligned windows ending at the last date in the data,
    stepping backwards from the latest date.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    last_date = df["date"].max()
    first_date = df["date"].min()

    start_dates = []
    current = last_date
    while current >= first_date + pd.Timedelta(days=window_size - 1):
        start_dates.append(current - pd.Timedelta(days=window_size - 1))
        current -= pd.Timedelta(days=window_size)

    return [
        pd.date_range(start, periods=window_size).to_pydatetime()
        for start in reversed(start_dates)
    ]


def generate_cyclical_features(df: pd.DataFrame, window_size: int = 7) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Ensure store_item column exists if needed
    if "store_item" not in df.columns:
        df["store_item"] = df["store"].astype(str) + "_" + df["item"].astype(str)

    # Prepare full column structure regardless of input size
    cols = ["start_date", "store_item", "store", "item"]
    for i in range(1, window_size + 1):
        for feature in ["dayofweek", "weekofmonth", "monthofyear"]:
            for trig in ["sin", "cos"]:
                cols.append(f"{feature}_{trig}_{i}")

    if df.empty:
        return pd.DataFrame(columns=cols)

    # Add cyclical features
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["weekofmonth"] = df["date"].apply(lambda d: (d.day - 1) // 7 + 1)
    df["weekofmonth_sin"] = np.sin(2 * np.pi * df["weekofmonth"] / 5)
    df["weekofmonth_cos"] = np.cos(2 * np.pi * df["weekofmonth"] / 5)
    df["monthofyear"] = df["date"].dt.month
    df["monthofyear_sin"] = np.sin(2 * np.pi * df["monthofyear"] / 12)
    df["monthofyear_cos"] = np.cos(2 * np.pi * df["monthofyear"] / 12)

    windows = generate_aligned_windows(df, window_size)
    results = []

    for store_item, group in df.groupby("store_item"):
        group = group.sort_values("date").reset_index(drop=True)

        for window_dates in windows:
            window_df = group[group["date"].isin(window_dates)]
            if window_df.empty:
                continue

            row = {
                "start_date": window_df["date"].min(),
                "store_item": store_item,
                "store": window_df["store"].iloc[0],
                "item": window_df["item"].iloc[0],
            }

            for i in range(window_size):
                if i < len(window_df):
                    r = window_df.iloc[i]
                    for f in ["dayofweek", "weekofmonth", "monthofyear"]:
                        for t in ["sin", "cos"]:
                            row[f"{f}_{t}_{i+1}"] = r[f"{f}_{t}"]
                else:
                    for f in ["dayofweek", "weekofmonth", "monthofyear"]:
                        for t in ["sin", "cos"]:
                            row[f"{f}_{t}_{i+1}"] = 0.0

            results.append(row)

    return pd.DataFrame(results, columns=cols)


def generate_sales_features(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
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

    # Assemble DataFrame
    cols = ["start_date", "store_item", "store", "item"]
    sales_cols = [f"sales_day_{i}" for i in range(1, window_size + 1)]
    store_med_cols = [f"store_med_day_{i}" for i in range(1, window_size + 1)]
    item_med_cols = [f"item_med_day_{i}" for i in range(1, window_size + 1)]

    cols.extend(sales_cols + store_med_cols + item_med_cols)

    if not records:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(records)
    return df[cols]


def add_y_targets_from_shift(df, window_size=16):
    df = df.sort_values(["store_item", "start_date"]).reset_index(drop=True)
    result = []

    for _, group in df.groupby("store_item"):
        group = group.sort_values("start_date").reset_index(drop=True)

        for i in range(len(group) - 1):
            cur_row = group.iloc[i]
            next_row = group.iloc[i + 1]

            # Check if next start_date is exactly +16 days
            if (next_row["start_date"] - cur_row["start_date"]).days != window_size:
                continue

            row = cur_row.copy()
            for col in group.columns:
                if any(
                    col.startswith(prefix)
                    for prefix in [
                        "sales_day_",
                        "store_med_day_",
                        "item_med_day_",
                        "dayofweek_",
                        "weekofmonth_",
                        "monthofyear_",
                    ]
                ):
                    row[f"y_{col}"] = next_row[col]

            result.append(row)

    return pd.DataFrame(result)


def prepare_training_data_from_raw_df(df, window_size=16):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "store_item" not in df.columns:
        df["store_item"] = df["store"].astype(str) + "_" + df["item"].astype(str)

    # Removed unused variable
    cyc_df = generate_cyclical_features(df, window_size)
    print(f"cyc_df.shape: {cyc_df.shape}")
    sales_df = generate_sales_features(df, window_size)
    print(f"sales_df.shape: {sales_df.shape}")

    merged_df = pd.merge(
        cyc_df, sales_df, on=["start_date", "store_item", "store", "item"], how="inner"
    )
    print(f"merged_df.shape: {merged_df.shape}")
    merged_df = add_y_targets_from_shift(merged_df, window_size)
    print(f"merged_df.shape: {merged_df.shape}")

    return merged_df


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
