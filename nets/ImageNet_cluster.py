import torch
import torch.nn as nn

from models.modules import ConvNextBlock, LayerNorm, UpSampleConvnext
from models.revcol_function import ReverseFunction
from timm.models.layers import trunc_normal_


class Fusion(nn.Module):
    def __init__(self, level, channels):
        super().__init__()
        self.level = level
        self.down = (
            nn.Sequential(
                nn.Conv2d(channels[level - 1], channels[level], kernel_size=2, stride=2),
                LayerNorm(channels[level], eps=1e-6, data_format="channels_first"),
            )
            if level in [1, 2, 3]
            else nn.Identity()
        )
        self.up = UpSampleConvnext(1, channels[level + 1], channels[level]) if level in [0, 1, 2] else nn.Identity()

    def forward(self, lower_feature, upper_feature):
        if self.level == 3:
            return self.down(lower_feature)
        return self.up(upper_feature) + self.down(lower_feature)


class Level(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, drop_path_rates):
        super().__init__()
        start = sum(layers[:level])
        expansion = 4
        self.fusion = Fusion(level, channels)
        blocks = [
            ConvNextBlock(
                channels[level],
                expansion * channels[level],
                channels[level],
                kernel_size=kernel_size,
                layer_scale_init_value=1e-6,
                drop_path=drop_path_rates[start + block_index],
            )
            for block_index in range(layers[level])
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, lower_feature, upper_feature):
        return self.blocks(self.fusion(lower_feature, upper_feature))


class HashProjection(nn.Module):
    def __init__(self, in_feature, hash_dim):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(in_feature, hash_dim), nn.Tanh())

    def forward(self, x):
        return nn.functional.normalize(self.project(x.flatten(1)), dim=1)


class train_column(nn.Module):
    def __init__(self, channels, layers, kernel_size, drop_path_rates, bit=128, infeatures=None):
        super().__init__()
        shortcut_scale = 0.5

        self.alpha0 = nn.Parameter(shortcut_scale * torch.ones((1, channels[0], 1, 1)), requires_grad=True)
        self.alpha1 = nn.Parameter(shortcut_scale * torch.ones((1, channels[1], 1, 1)), requires_grad=True)
        self.alpha2 = nn.Parameter(shortcut_scale * torch.ones((1, channels[2], 1, 1)), requires_grad=True)
        self.alpha3 = nn.Parameter(shortcut_scale * torch.ones((1, channels[3], 1, 1)), requires_grad=True)

        self.level0 = Level(0, channels, layers, kernel_size, drop_path_rates)
        self.level1 = Level(1, channels, layers, kernel_size, drop_path_rates)
        self.level2 = Level(2, channels, layers, kernel_size, drop_path_rates)
        self.level3 = Level(3, channels, layers, kernel_size, drop_path_rates)

        self.hash_projections = nn.ModuleDict(
            {
                "level0": HashProjection(infeatures[0], bit),
                "level1": HashProjection(infeatures[1], bit),
                "level2": HashProjection(infeatures[2], bit),
                "level3": HashProjection(infeatures[3], bit),
            }
        )
        self.apply(self._init_weights)

    def forward(self, features):
        local_functions = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunction.apply(local_functions, alpha, *features)
        hash_results = {
            "level0": self.hash_projections["level0"](c0),
            "level1": self.hash_projections["level1"](c1),
            "level2": self.hash_projections["level2"](c2),
            "level3": self.hash_projections["level3"](c3),
        }
        clusters = {
            "level0": self.hash_projections["level0"](features[1]),
            "level1": self.hash_projections["level1"](features[2]),
            "level2": self.hash_projections["level2"](features[3]),
            "level3": self.hash_projections["level3"](features[4]),
        }
        return {"hash_codes": hash_results, "output": hash_results["level3"], "clusters": clusters}

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)


def ImageNet_cluster(drop_path=0.1, kernel_size=3, bit=128, layers=None, droprate=0.1):
    del droprate
    channels = [64, 128, 256, 512]
    layers = layers or [2, 2, 4, 2]
    drop_path_rates = [value.item() for value in torch.linspace(0, drop_path, sum(layers))]
    infeatures = [64 * 56 * 56, 128 * 28 * 28, 256 * 14 * 14, 512 * 7 * 7]
    return train_column(channels, layers, kernel_size=kernel_size, drop_path_rates=drop_path_rates, bit=bit, infeatures=infeatures)
