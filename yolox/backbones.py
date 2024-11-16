# See: https://github.com/open-mmlab/mmyolo/tree/main/configs/yolox
import torch


class ConvModule(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.conv: torch.nn.Conv2d = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn: torch.nn.BatchNorm2d = torch.nn.BatchNorm2d(num_features=out_channels)
        self.act: torch.nn.modules.activation = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DarknetBottleneck(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            add: bool = True,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.conv1x1: ConvModule = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv3x3: ConvModule = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.need_add: bool = add

        self.shortcut: torch.nn.Module = (
            ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            if add and in_channels != out_channels
            else torch.nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv3x3(self.conv1x1(x))

        if self.need_add:
            return out + identity
        return out


# TODO: 所以Focus这样隔像素抽取有啥用?
class FocusBlock(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int = 64,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.conv: ConvModule = ConvModule(
            in_channels=int(in_channels * 4),
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        top_left: torch.Tensor = x[:, :, 0::2, 0::2]  # 偶数行偶数列
        top_right: torch.Tensor = x[:, :, 1::2, 0::2]  # 偶数行奇数列
        bot_left: torch.Tensor = x[:, :, 0::2, 1::2]  # 奇数行偶数列
        bot_right: torch.Tensor = x[:, :, 1::2, 1::2]  # 奇数行奇数列

        out: torch.Tensor = torch.concat([top_left, top_right, bot_left, bot_right], dim=1)
        return self.conv(out)


class CSPLayer(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_layers: int,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        reduce_channel: int = out_channels // 2 + out_channels % 2
        bypass_channel: int = out_channels // 2 + out_channels % 2

        self.conv_reduce: ConvModule = ConvModule(
            in_channels=in_channels,
            out_channels=reduce_channel,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv_bypass: ConvModule = ConvModule(
            in_channels=in_channels,
            out_channels=bypass_channel,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        raise NotImplementedError

        self.layers: torch.nn.Sequential = torch.nn.Sequential(
        )
