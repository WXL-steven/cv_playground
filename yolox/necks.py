# See: https://github.com/open-mmlab/mmyolo/tree/main/configs/yolox
from dataclasses import dataclass, field, fields
from typing import Optional

import torch

from yolox.modules import ConvModule, FocusBlock, CSPLayer, SPPFBottleneck
from yolox.backbones import CSPDarknetStageFeatures


@dataclass
class PAFPNStageFeatures:
    out0: Optional[torch.Tensor] = field(default=None)
    out1: Optional[torch.Tensor] = field(default=None)
    out2: Optional[torch.Tensor] = field(default=None)

    def __iter__(self) -> iter:
        self: dataclass  # 这是给若智PyCharm的静态分析看的
        for f in fields(self):
            yield f.name, getattr(self, f.name)


class PAFPN(torch.nn.Module):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.reduce_layer1 = ConvModule(
            in_channels=1024,
            out_channels=512,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.top_down_layer2 = torch.nn.Sequential(
            CSPLayer(
                in_channels=1024,
                out_channels=512,
                add=False,
                num_blocks=3
            ),
            ConvModule(
                in_channels=512,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )

        self.top_down_layer1 = CSPLayer(
            in_channels=512,
            out_channels=256,
            add=False,
            num_blocks=3
        )

        self.out_layer0 = ConvModule(
            in_channels=256,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.down_scamp0 = ConvModule(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.bottom_up_layer0 = CSPLayer(
            in_channels=512,
            out_channels=512,
            num_blocks=3,
            add=False
        )

        self.out_layer1 = ConvModule(
            in_channels=512,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.down_scamp1 = ConvModule(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.bottom_up_layer1 = CSPLayer(
            in_channels=1024,
            out_channels=1024,
            num_blocks=3,
            add=False
        )

        self.out_layer2 = ConvModule(
            in_channels=1024,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.up_sample = torch.nn.Upsample(
            scale_factor=2,
            mode='nearest'  # 使用最邻近上采样
        )

    def forward(self, features: CSPDarknetStageFeatures) -> PAFPNStageFeatures:
        # [N, 1024, 20, 20]
        reduce_out = self.reduce_layer1(features.stage4)
        up_sample2 = self.up_sample(reduce_out)
        top_down_out2 = self.top_down_layer2(
            torch.concat((features.stage3, up_sample2), dim=1)
        )
        up_sample1 = self.up_sample(top_down_out2)
        top_down_out1 = self.top_down_layer1(
            torch.concat((features.stage2, up_sample1), dim=1)
        )

        down_sample_out0 = self.down_scamp0(top_down_out1)
        bottom_up_out0 = self.bottom_up_layer0(
            torch.concat((down_sample_out0, top_down_out2), dim=1)
        )
        down_sample_out1 = self.down_scamp1(bottom_up_out0)
        bottom_up_out1 = self.bottom_up_layer1(
            torch.concat((down_sample_out1, reduce_out), dim=1)
        )

        out0 = self.out_layer0(top_down_out1)
        out1 = self.out_layer1(bottom_up_out0)
        out2 = self.out_layer2(bottom_up_out1)

        return PAFPNStageFeatures(out0, out1, out2)
