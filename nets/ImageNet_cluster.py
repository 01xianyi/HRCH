import torch
import torch.nn as nn
from models.modules import ConvNextBlock, Decoder, LayerNorm, SimDecoder, UpSampleConvnext
import torch.distributed as dist
from models.revcol_function import ReverseFunction
from timm.models.layers import trunc_normal_
from torch.nn import functional as F
from utils.config import args




class Fusion(nn.Module):
    def __init__(self, level, channels) -> None:
        super().__init__()
        self.level = level
        self.down = nn.Sequential(
            nn.Conv2d(channels[level - 1], channels[level], kernel_size=2, stride=2),
            LayerNorm(channels[level], eps=1e-6, data_format="channels_first"),  # 向下采样，缩小特征图尺寸
        ) if level in [1, 2, 3] else nn.Identity()
        self.up = UpSampleConvnext(1, channels[level + 1], channels[level]) if level in [0, 1, 2] else nn.Identity()

    def forward(self, *args):
        c_down, c_up = args
        if self.level == 3:
            x = self.down(c_down)
        else:
            x = self.up(c_up) + self.down(c_down)
        return x

class Level(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, dp_rate=0.0) -> None:
        super().__init__()
        countlayer = sum(layers[:level])
        expansion = 4
        self.fusion = Fusion(level, channels)
        # modules = [ConvNextBlockWithBN(channels[level], expansion * channels[level], channels[level], kernel_size=kernel_size,
        #                          layer_scale_init_value=1e-6, drop_path=dp_rate[countlayer + i]) for i in
        #            range(layers[level])]
        modules = [ConvNextBlock(channels[level], expansion * channels[level], channels[level], kernel_size=kernel_size,
                                 layer_scale_init_value=1e-6, drop_path=dp_rate[countlayer + i]) for i in
                   range(layers[level])]

        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x = self.fusion(*args)
        x = self.blocks(x)
        return x

class ConvNextBlockWithBN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, layer_scale_init_value=1e-6,
                 drop_path=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.GELU()

        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((out_channels, 1, 1)), requires_grad=True)
        self.dropout = nn.Dropout2d(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.layer_scale * x
        x = self.dropout(x)
        return x


class train_column(nn.Module):
    def __init__(self, channels, layers, kernel_size, dp_rates,bit=128,infeatures=None,num_clusters=None,droprate=0.1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5

        self.apply(self._init_weights)

        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None

        self.level0 = Level(0, channels, layers, kernel_size, dp_rates)

        self.level1 = Level(1, channels, layers, kernel_size, dp_rates)

        self.level2 = Level(2, channels, layers, kernel_size, dp_rates)

        self.level3 = Level(3, channels, layers, kernel_size, dp_rates)

        self.hash_projections = nn.ModuleDict({
            'level0': HashProjection(infeatures[0],bit,droprate),
            'level1': HashProjection(infeatures[1],bit,droprate),
            'level2': HashProjection(infeatures[2],bit,droprate),
            'level3': HashProjection(infeatures[3],bit,droprate)
        })

    def forward(self, *args):
        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        args=args[0]
        # print(len(args))
        _, c0, c1, c2, c3 = ReverseFunction.apply(
            local_funs, alpha, *args)
        self.hash_results = {
            'level0': self.hash_projections['level0'](c0),
            'level1': self.hash_projections['level1'](c1),
            'level2': self.hash_projections['level2'](c2),
            'level3': self.hash_projections['level3'](c3)
        }
        return {'hash_codes': self.hash_results}

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=.02)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            nn.init.constant_(module.bias, 0)

class HashProjection(nn.Module):
    def __init__(self, in_feature ,hash_dim,level=None,droprate=0.1):
        super().__init__()
        self.level = level
        self.hash_dim = hash_dim
        self.project = nn.Sequential(
                    nn.Linear(in_features=in_feature, out_features=self.hash_dim),
                    nn.Tanh()
                ).cuda()


    def forward(self, x):
        x=x.flatten(1)
        x=self.project(x)

        norm_x = torch.norm(x, dim=1, keepdim=True)
        x = x / norm_x
        return x

def ImageNet_cluster (drop_path=0.1, kernel_size = 3 , bit=128 ,layers=[2,2,4,2],droprate=0.1):
    channels = [64, 128, 256, 512]

    layers = layers
    dp_rate = [x.item() for x in torch.linspace(0, drop_path, sum(layers))]
    infeatures=[64*56*56,128*28*28,256*14*14,512*7*7]

    return train_column(channels, layers,dp_rates = dp_rate,kernel_size=kernel_size,bit=bit,infeatures=infeatures,droprate=droprate)
