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
            num_blocks: int,
            add: bool,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        split_channels = out_channels // 2  # reduce_channel 改为更通用的 split_channels
        main_channels = split_channels  # bypass_channel 可以简化
        shortcut_channels = out_channels - split_channels  # 处理奇数通道更优雅

        # 主路径
        self.main_conv: ConvModule = ConvModule(
            in_channels=in_channels,
            out_channels=main_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # shortcut路径
        self.shortcut_conv: ConvModule = ConvModule(
            in_channels=in_channels,
            out_channels=shortcut_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Bottleneck blocks
        self.blocks: torch.nn.Sequential = torch.nn.Sequential(
            *(
                DarknetBottleneck(
                    in_channels=main_channels,
                    out_channels=main_channels,
                    add=add,
                )
                for _ in range(num_blocks)
            )
        )

        # 融合卷积
        self.final_conv: ConvModule = ConvModule(
            in_channels=out_channels,  # main_channels + shortcut_channels
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shortcut path
        shortcut = self.shortcut_conv(x)

        # main path
        main = self.main_conv(x)
        main = self.blocks(main)

        # concat and final conv
        out = torch.cat([shortcut, main], dim=1)
        return self.final_conv(out)

# TODO: Need Check
class SPPFBottleneck(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        if out_channels % 2 != 0:
            raise ValueError("out_channels must be even.")

        self.conv1: ConvModule = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # 此处将步长(stride)设置为1并配置填充(stride)为1/2核大小(kernel_size)来实现不改变输出大小的池化
        self.max_pool1: torch.nn.MaxPool2d = torch.nn.MaxPool2d(
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.max_pool2: torch.nn.MaxPool2d = torch.nn.MaxPool2d(
            kernel_size=9,
            stride=1,
            padding=4
        )
        self.max_pool3: torch.nn.MaxPool2d = torch.nn.MaxPool2d(
            kernel_size=13,
            stride=1,
            padding=6
        )

        self.conv2: ConvModule = ConvModule(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)

        # 三个maxpool分支
        maxpool1 = self.max_pool1(x)
        maxpool2 = self.max_pool2(x)
        maxpool3 = self.max_pool3(x)

        # concat操作:将原始x和三个maxpool结果在channel维度上拼接
        concat = torch.cat([x, maxpool1, maxpool2, maxpool3], dim=1)

        # 最后一个conv
        out = self.conv2(concat)

        return out
