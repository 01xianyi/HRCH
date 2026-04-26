import torch
import torch.nn as nn
import torch.nn.functional as F


class cluster_mlp(nn.Module):
    """Prototype classifier initialized from sampled hash codes."""

    def __init__(self, hash_dim, cluster_num, prototypes):
        super().__init__()
        self.fc = nn.Linear(hash_dim, cluster_num)
        self.fc.weight = nn.Parameter(prototypes.clone().detach().float(), requires_grad=True)

    def forward(self, x):
        return F.normalize(self.fc(x), p=2, dim=1)
