import torch
from torch import nn
from torch.utils.data import TensorDataset, Dataset, DataLoader, Subset


class StoreItemDataset(Dataset):
    def __init__(self, df, store_item_id, feature_cols, target_col, weight_col):
        self.store_df = df[df["store_item"] == store_item_id].reset_index(drop=True)
        self.X = torch.tensor(self.store_df[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(
            self.store_df[target_col].values, dtype=torch.float32
        ).unsqueeze(1)
        self.w = torch.tensor(
            self.store_df[weight_col].values, dtype=torch.float32
        ).unsqueeze(1)

    def __len__(self):
        return len(self.store_df)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx]


# 2) Define the Shallow NN with tanh activations
class ShallowNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=None):
        super().__init__()
        output_dim = output_dim or input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),  # outputs in (0,1)
        )

    def forward(self, x):
        return self.net(x)


# 3) Implement NWRMSLE loss
class NWRMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, w):
        # add 1 to avoid log(0), clamp to eps to keep preds positive
        eps = 1e-6
        y_pred = torch.clamp(y_pred, min=eps)
        log_diff = torch.log(y_pred + 1.0) - torch.log(y_true + 1.0)
        num = torch.sum(w * log_diff**2)
        den = torch.sum(w)
        return torch.sqrt(num / den)
