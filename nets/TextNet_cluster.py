import torch
from torch import nn
from torch.nn import functional as F
from utils.config import args

class TextNet_cluster(nn.Module):
    def __init__(self, y_dim, bit, norm=True, mid_num1=1024*8, mid_num2=1024*8,droprate=0.0):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(TextNet_cluster, self).__init__()
        self.norm = norm

        self.l0 = nn.Sequential(nn.Linear(y_dim, mid_num1), nn.ReLU(inplace=True), nn.Dropout(droprate))
        self.l1 = nn.Sequential(nn.Linear(mid_num1, mid_num2), nn.ReLU(inplace=True), nn.Dropout(droprate))
        self.cluster0 = nn.Sequential(nn.Linear(mid_num1, bit), nn.Tanh())
        self.cluster1 = nn.Sequential(nn.Linear(mid_num2, bit), nn.Tanh())

    def forward(self, x):
        l0 = self.l0(x)
        l1 = self.l1(l0)
        cluster0 = self.cluster0(l0)
        cluster1 = self.cluster1(l1)

        cluster0=F.normalize(cluster0, p=2, dim=1)
        cluster1=F.normalize(cluster1, p=2, dim=1)


        self.hash_results = {
            'level0': cluster0,
            'level1': cluster0,
            'level2': cluster1,
            'level3': cluster1
        }

        return {'hash_codes': self.hash_results}

