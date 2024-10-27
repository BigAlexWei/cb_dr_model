import math
from collections import OrderedDict

import torch
from torch import nn
from torchvision.ops.misc import Conv3dNormActivation

param_dict = {4: {"depth": 16, "w_0": 48, "w_a": 27.89, "w_m": 2.09, "group_width": 8},
              8: {"depth": 14, "w_0": 56, "w_a": 38.84, "w_m": 2.4, "group_width": 16},
              16: {"depth": 27, "w_0": 48, "w_a": 20.71, "w_m": 2.65, "group_width": 24},
              32: {"depth": 21, "w_0": 80, "w_a": 42.63, "w_m": 2.66, "group_width": 24},
              64: {"depth": 25, "w_0": 112, "w_a": 33.36, "w_m": 2.28, "group_width": 72}}


def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


class SqueezeExcitation3d(nn.Module):
    def __init__(self, in_chs, squeeze_chs):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool3d(1)

        self.fc1 = nn.Conv3d(in_chs, squeeze_chs, 1)
        self.fc2 = nn.Conv3d(squeeze_chs, in_chs, 1)
        self.activation = nn.ReLU()

        self.scale_activation = nn.Sigmoid()

    def _scale(self, x):
        scale = self.avgpool(x)

        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)

        return self.scale_activation(scale)

    def forward(self, x):
        scale = self._scale(x)

        return scale * x


class SimpleStemIN(Conv3dNormActivation):
    def __init__(self, in_chs):
        super().__init__(in_chs, 32, kernel_size=3, stride=2)


class BottleneckTransform(nn.Sequential):
    def __init__(self, in_chs, out_chs, stride, group_width, se_ratio=0.25):
        layers: OrderedDict[str, nn.Module] = OrderedDict()

        w_b = out_chs
        g = w_b // group_width

        layers["a"] = Conv3dNormActivation(in_chs, w_b, kernel_size=1)

        layers["b"] = Conv3dNormActivation(w_b, w_b, kernel_size=3, stride=stride, groups=g)

        width_se_out = int(round(se_ratio * in_chs))
        layers["se"] = SqueezeExcitation3d(in_chs=w_b, squeeze_chs=width_se_out)

        layers["c"] = Conv3dNormActivation(w_b, out_chs, kernel_size=1, activation_layer=None)

        super().__init__(layers)


class BottleneckBlock(nn.Module):
    def __init__(self, in_chs, out_chs, stride, group_width):
        super().__init__()

        self.proj = None
        if stride != 1:
            self.proj = Conv3dNormActivation(in_chs, out_chs, kernel_size=3, stride=stride, activation_layer=None)

        self.f = BottleneckTransform(in_chs, out_chs, stride, group_width)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.proj is not None:
            x = self.proj(x) + self.f(x)
        else:
            x = x + self.f(x)

        return self.activation(x)


class AnyStage(nn.Sequential):
    def __init__(self, in_chs, out_chs, depth, group_width, stage_index):
        super().__init__()

        for i in range(depth):
            block = BottleneckBlock(in_chs=in_chs if i == 0 else out_chs,
                                    out_chs=out_chs,
                                    stride=2 if i == 0 else 1,
                                    group_width=group_width)

            self.add_module(f"block{stage_index}-{i}", block)


class BlockParams:
    def __init__(self, depths, widths, group_widths):
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths

    @classmethod
    def from_init_params(cls, depth, w_0, w_a, w_m, group_width):
        # Compute the block widths. Each stage has one unique block width
        widths_cont = torch.arange(depth) * w_a + w_0
        block_capacity = torch.round(torch.log(widths_cont / w_0) / math.log(w_m))
        block_widths = (torch.round(torch.divide(w_0 * torch.pow(w_m, block_capacity), 8)) * 8).int().tolist()
        num_stages = len(set(block_widths))

        # Convert to per stage parameters
        split_helper = zip(block_widths + [0], [0] + block_widths, block_widths + [0], [0] + block_widths)
        splits = [w != wp or r != rp for w, wp, r, rp in split_helper]

        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = torch.diff(torch.tensor([d for d, t in enumerate(splits) if t])).int().tolist()

        group_widths = [group_width] * num_stages

        # Adjust the compatibility of stage widths and group widths
        stage_widths, group_widths = cls._adjust_widths_groups_compatibilty(stage_widths, group_widths)

        return zip(stage_widths, stage_depths, group_widths)

    @staticmethod
    def _adjust_widths_groups_compatibilty(stage_widths, group_widths):
        # Compute all widths for the current settings
        group_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, stage_widths)]

        # Compute the adjusted widths so that stage and group widths fit
        stage_widths = [make_divisible(w_bot, g) for w_bot, g in zip(stage_widths, group_widths_min)]

        return stage_widths, group_widths_min


class RegNetEncoder(nn.Module):
    def __init__(self, stem_in_chs, block_params):
        super().__init__()

        self.stem = SimpleStemIN(stem_in_chs)

        current_width = self.stem.out_channels

        blocks = []
        for i, (out_chs, depth, group_width) in enumerate(block_params):
            current_stage = AnyStage(current_width, out_chs, depth, group_width, i + 1)

            blocks.append((f"block{i + 1}", current_stage))

            current_width = out_chs

        self.trunk_output = nn.Sequential(OrderedDict(blocks))

        self.avgpool = nn.AdaptiveAvgPool3d(1)

        self.out_features = current_width

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.trunk_output(x)

        x = self.avgpool(x)

        return x.flatten(start_dim=1)


def regnet_encoder(stem_in_chs, complexity):
    params = BlockParams.from_init_params(**param_dict[complexity])

    return RegNetEncoder(stem_in_chs, params)


class Model(nn.Module):
    def __init__(self, image_in_chs, image_encoder_complexity, extra_feas):
        super().__init__()

        self.m_image_encoder = regnet_encoder(image_in_chs, image_encoder_complexity)

        self.m_decoder = nn.Linear(in_features=self.m_image_encoder.out_features + extra_feas, out_features=1)
        nn.init.normal_(self.m_decoder.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.m_decoder.bias)

    def forward(self, i_x, e_x, mixup_coef=None):
        i_x = self.m_image_encoder(i_x)

        x = torch.concatenate([i_x, e_x], dim=1)

        if mixup_coef is not None:
            x = (x + mixup_coef * x.flip(dims=[0])) / (1 + mixup_coef)

        return self.m_decoder(x)
