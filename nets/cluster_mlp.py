import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class cluster_mlp(nn.Module):
    def __init__(self, hash_dim, cluster_num, prototypes):
        super(cluster_mlp, self).__init__()

        self.fc = nn.Linear(hash_dim, cluster_num)
        self.prototypes = prototypes
        self.fc.weight = nn.Parameter(self.prototypes.clone().detach().float(), requires_grad=True)

    def forward(self, x):
        out=F.normalize(self.fc(x), p=2, dim=1)
        # out=self.fc(x)
        return out



if __name__ == '__main__':
    hash_dim = 128
    cluster_num = 1000
    prototypes = torch.randn(cluster_num, hash_dim)

    model = cluster_mlp(hash_dim, cluster_num, prototypes)
    sample_input = torch.randn(5, hash_dim)
    output = model(sample_input)
    _, min_indice=torch.min(output,dim=1)
    print(min_indice)
    print("Output shape:", output.shape)
