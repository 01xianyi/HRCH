import torch.nn as nn
import torch.nn.functional as F


class TextNet_cluster(nn.Module):
    """Text encoder that exposes four hash levels for hierarchical training."""

    def __init__(self, y_dim, bit, data_name, norm=True, mid_num1=1024 * 6, mid_num2=1024 * 8, droprate=0.15):
        super().__init__()
        self.norm = norm
        self.data_name = data_name.lower()

        if "mir" in self.data_name:
            hidden_1 = mid_num1
            hidden_2 = mid_num2
            dropout = droprate
        else:
            hidden_1 = 1024 * 8
            hidden_2 = 1024 * 12
            dropout = 0.1

        self.l0 = nn.Sequential(nn.Linear(y_dim, hidden_1), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.l1 = nn.Sequential(nn.Linear(hidden_1, hidden_2), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.cluster0 = nn.Sequential(nn.Linear(hidden_1, bit), nn.Tanh())
        self.cluster1 = nn.Sequential(nn.Linear(hidden_2, bit), nn.Tanh())

    def forward(self, x):
        level0_features = self.l0(x)
        level1_features = self.l1(level0_features)
        cluster0 = F.normalize(self.cluster0(level0_features), p=2, dim=1)
        cluster1 = F.normalize(self.cluster1(level1_features), p=2, dim=1)
        hash_results = {
            "level0": cluster0,
            "level1": cluster0,
            "level2": cluster1,
            "level3": cluster1,
        }
        return {"hash_codes": hash_results, "output": cluster1}
