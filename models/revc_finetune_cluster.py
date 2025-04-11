import torch
import torch.nn as nn
from models.modules import ConvNextBlock, Decoder, LayerNorm, SimDecoder, UpSampleConvnext
import torch.distributed as dist
from models.revcol_function import ReverseFunction
from timm.models.layers import trunc_normal_


class Fusion(nn.Module):
    def __init__(self, level, channels, first_col) -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        self.down = nn.Sequential(
            nn.Conv2d(channels[level - 1], channels[level], kernel_size=2, stride=2),
            LayerNorm(channels[level], eps=1e-6, data_format="channels_first"),
        ) if level in [1, 2, 3] else nn.Identity()
        if not first_col:
            self.up = UpSampleConvnext(1, channels[level + 1], channels[level]) if level in [0, 1, 2] else nn.Identity()

    def forward(self, *args):
        c_down, c_up = args
        if self.first_col:
            x = self.down(c_down)
            return x
        if self.level == 3:
            x = self.down(c_down)
        else:
            x = self.up(c_up) + self.down(c_down)
        return x

class Level(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0) -> None:
        super().__init__()
        countlayer = sum(layers[:level])
        expansion = 4
        self.fusion = Fusion(level, channels, first_col)
        modules = [ConvNextBlock(channels[level], expansion * channels[level], channels[level], kernel_size=kernel_size,
                                 layer_scale_init_value=1e-6, drop_path=0.0) for i in
                   range(layers[level])]

        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x = self.fusion(*args)
        x = self.blocks(x)
        return x

class SubNet(nn.Module):
    def __init__(self, channels, layers, kernel_size, first_col, dp_rates, save_memory) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        self.save_memory = save_memory
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None

        self.level0 = Level(0, channels, layers, kernel_size, first_col, dp_rates)

        self.level1 = Level(1, channels, layers, kernel_size, first_col, dp_rates)

        self.level2 = Level(2, channels, layers, kernel_size, first_col, dp_rates)

        self.level3 = Level(3, channels, layers, kernel_size, first_col, dp_rates)

    def forward(self, *args):
        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunction.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3

class FullNet(nn.Module):
    def __init__(self, channels=[32, 64, 96, 128], layers=[2, 3, 6, 3], num_subnet=5, kernel_size=3,
                 drop_path=0.0,save_memory=True, inter_supv=True, head_init_scale=None) -> None:
        # super().__init__()
        super(FullNet, self).__init__()
        self.num_subnet = num_subnet
        self.inter_supv = inter_supv
        self.channels = channels
        self.layers = layers

        self.stem = nn.Sequential(
                nn.Conv2d(3, channels[0], kernel_size=4,stride=4),
                LayerNorm(channels[0], eps=1e-6, data_format="channels_first"))

        dp_rate = [x.item() for x in torch.linspace(0, drop_path, sum(layers))]

        for i in range(num_subnet):
            first_col = True if i == 0 else False
            self.add_module(f'subnet{str(i)}', SubNet(
                channels, layers, kernel_size, first_col, dp_rates=dp_rate, save_memory=save_memory))

        self.apply(self._init_weights)

        if head_init_scale:
            print(f'Head_init_scale: {head_init_scale}')
            self.cls.classifier._modules['1'].weight.data.mul_(head_init_scale)
            self.cls.classifier._modules['1'].bias.data.mul_(head_init_scale)

    def forward(self, x):
        c0, c1, c2, c3 = 0, 0, 0, 0
        x = self.stem(x)
        # print("x.shape",x.shape)
        for i in range(self.num_subnet):
            c0, c1, c2, c3 = getattr(self, f'subnet{str(i)}')(x, c0, c1, c2, c3)
            if i==3:
                return x,c0,c1,c2,c3

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=.02)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            nn.init.constant_(module.bias, 0)

def revcol_tiny_fintune_cluster(save_memory, inter_supv=True, drop_path=0.1, num_classes=1000, kernel_size = 3):
    channels = [64, 128, 256, 512]
    layers = [2, 2, 4, 2]
    num_subnet = 4
    return FullNet(channels, layers, num_subnet, drop_path = drop_path, save_memory=save_memory, inter_supv=inter_supv, kernel_size=kernel_size)
