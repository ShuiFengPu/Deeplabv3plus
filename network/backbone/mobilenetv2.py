import torch
import torch.nn as nn
import torch.hub as hub

__all__ = ['MobileNetV2']


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes,
                 kernel_size=3, stride=1,
                 groups=1, dilation=1):
        '''in_planes：输入通道数。
            out_planes：输出通道数。
            kernel_size：卷积核大小（默认3）。
            stride：步幅（默认1）。
            padding：填充大小。
            dilation：膨胀率，用于控制卷积核的感受野。
            groups：分组卷积的组数。
            bias=False：不使用偏置，因为批量归一化会处理偏置。
        '''
        padding = (dilation * (kernel_size - 1) + 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size,
                      stride, padding, dilation=dilation,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

        # 深度可分离卷积适配膨胀率
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim,
                       stride=stride, groups=hidden_dim,
                       dilation=dilation),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 output_stride=32,  # 新增参数
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8):
        super().__init__()

        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # 根据output_stride调整膨胀率
        self.dilation = 1
        self.current_stride = 1
        self.output_stride = output_stride

        input_channel = self._make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = self._make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        features = [ConvBNReLU(3, input_channel, stride=2)]
        self.current_stride *= 2  # 首层stride=2

        # 构建中间层（适配output_stride）
        for t, c, n, s in inverted_residual_setting:
            output_channel = self._make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                # 动态调整stride和dilation
                stride = s if i == 0 else 1
                if self.current_stride * stride > self.output_stride:
                    stride = 1
                    self.dilation *= s

                features.append(
                    block(input_channel, output_channel,
                          stride, expand_ratio=t,
                          dilation=self.dilation)
                )
                input_channel = output_channel

                # 更新当前总步长
                if stride > 1:
                    self.current_stride *= stride

        # 最终层
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _make_divisible(self, v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v


def mobilenet_v2(pretrained=False, progress=True, ** kwargs):
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = hub.load_state_dict_from_url(
            'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
            progress=progress
        )
        model.load_state_dict(state_dict, strict=False)  # 允许部分加载
    return model