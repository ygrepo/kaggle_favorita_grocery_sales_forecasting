import torch
import torch.nn as nn
import heapq
import pandas as pd
from collections import defaultdict, deque


class ShallowNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=None):
        super().__init__()
        output_dim = output_dim or 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class RunningMedian:
    def __init__(self):
        self.low = []  # max-heap
        self.high = []  # min-heap

    def add(self, num):
        heapq.heappush(self.low, -heapq.heappushpop(self.high, num))
        if len(self.low) > len(self.high):
            heapq.heappush(self.high, -heapq.heappop(self.low))

    def median(self):
        if not self.low and not self.high:
            return 0.0
        if len(self.high) > len(self.low):
            return float(self.high[0])
        return (self.high[0] - self.low[0]) / 2.0


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    sid = checkpoint["sid"]
    model_state = checkpoint["model_state_dict"]
    feature_cols = checkpoint["feature_cols"]
    input_dim = len(feature_cols)
    model = ShallowNN(input_dim=input_dim)
    model.load_state_dict(model_state)
    model.eval()
    return sid, model, feature_cols


class StreamPredictor:
    def __init__(self, model_dict):
        self.model_dict = model_dict
        self.store_sales_history = defaultdict(lambda: deque(maxlen=7))
        self.item_sales_history = defaultdict(lambda: deque(maxlen=7))
        self.store_medians = defaultdict(RunningMedian)
        self.item_medians = defaultdict(RunningMedian)

    def predict_stream(self, df):
        predictions = []
        for idx, row in df.iterrows():
            store = row["store"]
            item = row["item"]
            sid = f"{store}_{item}"

            if sid not in self.model_dict:
                predictions.append(0.0)
                continue

            model_info = self.model_dict[sid]
            model = model_info["model"]
            feature_cols = model_info["feature_cols"]

            store_hist = list(self.store_sales_history[store])
            item_hist = list(self.item_sales_history[item])
            store_hist = [0] * (7 - len(store_hist)) + store_hist
            item_hist = [0] * (7 - len(item_hist)) + item_hist

            input_features = row[feature_cols].astype("float32").values
            input_tensor = torch.tensor(input_features).unsqueeze(0)

            with torch.no_grad():
                pred = model(input_tensor).item()

            predictions.append(pred)

            self.store_sales_history[store].append(pred)
            self.item_sales_history[item].append(pred)
            self.store_medians[store].add(pred)
            self.item_medians[item].add(pred)

        return predictions
