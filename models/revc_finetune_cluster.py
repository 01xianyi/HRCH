import torch
import torch.nn as nn

from models.modules import ConvNextBlock, LayerNorm, UpSampleConvnext
from models.revcol_function import ReverseFunction
from timm.models.layers import trunc_normal_


class Fusion(nn.Module):
    def __init__(self, level, channels, first_column):
        super().__init__()
        self.level = level
        self.first_column = first_column
        self.down = (
            nn.Sequential(
                nn.Conv2d(channels[level - 1], channels[level], kernel_size=2, stride=2),
                LayerNorm(channels[level], eps=1e-6, data_format="channels_first"),
            )
            if level in [1, 2, 3]
            else nn.Identity()
        )
        self.up = (
            UpSampleConvnext(1, channels[level + 1], channels[level])
            if (not first_column and level in [0, 1, 2])
            else nn.Identity()
        )

    def forward(self, lower_feature, upper_feature):
        if self.first_column or self.level == 3:
            return self.down(lower_feature)
        return self.up(upper_feature) + self.down(lower_feature)


class Level(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_column):
        super().__init__()
        self.fusion = Fusion(level, channels, first_column)
        blocks = [
            ConvNextBlock(
                channels[level],
                4 * channels[level],
                channels[level],
                kernel_size=kernel_size,
                layer_scale_init_value=1e-6,
                drop_path=0.0,
            )
            for _ in range(layers[level])
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, lower_feature, upper_feature):
        return self.blocks(self.fusion(lower_feature, upper_feature))


class SubNet(nn.Module):
    def __init__(self, channels, layers, kernel_size, first_column):
        super().__init__()
        shortcut_scale = 0.5
        self.alpha0 = nn.Parameter(shortcut_scale * torch.ones((1, channels[0], 1, 1)), requires_grad=True)
        self.alpha1 = nn.Parameter(shortcut_scale * torch.ones((1, channels[1], 1, 1)), requires_grad=True)
        self.alpha2 = nn.Parameter(shortcut_scale * torch.ones((1, channels[2], 1, 1)), requires_grad=True)
        self.alpha3 = nn.Parameter(shortcut_scale * torch.ones((1, channels[3], 1, 1)), requires_grad=True)

        self.level0 = Level(0, channels, layers, kernel_size, first_column)
        self.level1 = Level(1, channels, layers, kernel_size, first_column)
        self.level2 = Level(2, channels, layers, kernel_size, first_column)
        self.level3 = Level(3, channels, layers, kernel_size, first_column)

    def forward(self, *features):
        local_functions = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunction.apply(local_functions, alpha, *features)
        return c0, c1, c2, c3


class FullNet(nn.Module):
    def __init__(self, channels, layers, num_subnet=4, kernel_size=3, drop_path=0.0):
        super().__init__()
        del drop_path
        self.num_subnet = num_subnet
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=4, stride=4),
            LayerNorm(channels[0], eps=1e-6, data_format="channels_first"),
        )

        for index in range(num_subnet):
            self.add_module(f"subnet{index}", SubNet(channels, layers, kernel_size, first_column=index == 0))

        self.apply(self._init_weights)

    def forward(self, x):
        x = self.stem(x)
        c0, c1, c2, c3 = 0, 0, 0, 0
        for index in range(self.num_subnet):
            c0, c1, c2, c3 = getattr(self, f"subnet{index}")(x, c0, c1, c2, c3)
        return x, c0, c1, c2, c3

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)


def revcol_tiny_fintune_cluster(save_memory, inter_supv=True, drop_path=0.1, num_classes=1000, kernel_size=3):
    del save_memory, inter_supv, num_classes
    channels = [64, 128, 256, 512]
    layers = [2, 2, 4, 2]
    return FullNet(channels=channels, layers=layers, num_subnet=4, kernel_size=kernel_size, drop_path=drop_path)
