from dataclasses import dataclass
from typing import Optional

import torch

from yolox.modules import ConvModule, DarknetBottleneck, FocusBlock, CSPLayer, SPPFBottleneck


@dataclass
class CSPDarknetStageFeatures:
    stage2: torch.Tensor
    stage3: torch.Tensor
    stage4: torch.Tensor


class CSPDarknet(torch.nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            # layer_scale: Optional[dict] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        # if layer_scale is None:
        #     layer_scale = {
        #         1: 1,
        #     }

        self.stem_layer = FocusBlock(
            in_channels=in_channels,
            out_channels=64
        )

        self.stage_layer1 = torch.nn.Sequential(
            ConvModule(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            CSPLayer(
                in_channels=128,
                out_channels=128,
                num_blocks=3,
                add=True
            )
        )

        self.stage_layer2 = torch.nn.Sequential(
            ConvModule(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            CSPLayer(
                in_channels=256,
                out_channels=256,
                num_blocks=9,
                add=True
            )
        )

        self.stage_layer3 = torch.nn.Sequential(
            ConvModule(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            CSPLayer(
                in_channels=512,
                out_channels=512,
                num_blocks=9,
                add=True
            )
        )

        self.stage_layer4 = torch.nn.Sequential(
            ConvModule(
                in_channels=512,
                out_channels=1024,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            SPPFBottleneck(
                in_channels=1024,
                out_channels=1024
            ),
            CSPLayer(
                in_channels=1024,
                out_channels=1024,
                num_blocks=3,
                add=False
            )
        )

    def forward(self, x: torch.Tensor) -> CSPDarknetStageFeatures:
        # [N, C, H, W] / [N, 3, 640, 640]
        x = self.stem_layer(x)

        # [N, 64, H/2, W/2] / [320, 320]
        x = self.stage_layer1(x)
        # [N, 128, H/4, W/4] / [160, 160]
        feature2 = self.stage_layer2(x)
        # [N, 256, H/8, W/8] / [80, 80]
        feature3 = self.stage_layer3(feature2)
        # [N, 512, H/16, W/16] / [40, 40]
        feature4 = self.stage_layer4(feature3)
        # [N, 1024, H/32, W/32] / [20, 20]

        return CSPDarknetStageFeatures(feature2, feature3, feature4)
