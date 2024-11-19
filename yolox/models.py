# See: https://github.com/open-mmlab/mmyolo/tree/main/configs/yolox
from dataclasses import dataclass, field, fields
from typing import Iterable, Optional

import torch

from yolox.backbones import CSPDarknet, CSPDarknetStageFeatures
from yolox.necks import PAFPN, PAFPNOutputFeatures
from yolox.heads import YOLOXHead, BBoxArchResult


@dataclass
class YOLOXOutput:
    small: Optional[BBoxArchResult] = field(default=None)  # 80x80
    medium: Optional[BBoxArchResult] = field(default=None)  # 40x40
    large: Optional[BBoxArchResult] = field(default=None)  # 20x20

    def __post_init__(self):
        self._data = (self.small, self.medium, self.large)

    def __getitem__(self, idx):
        return self._data[idx]

    def __iter__(self) -> iter:
        self: dataclass  # 这是给若智PyCharm的静态分析看的
        for f in fields(self):
            yield f.name, getattr(self, f.name)

    def __len__(self) -> int:
        return len(self._data)


class YOLOX_L(torch.nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels

        self.backbone = CSPDarknet(
            in_channels=in_channels
        )

        self.neck = PAFPN()

        self.heads = tuple(
            YOLOXHead() for _ in range(3)
        )

    def forward(self, x: torch.Tensor) -> Iterable[BBoxArchResult]:
        if x.shape[1] != self.in_channels:
            raise ValueError("Input tensor channel miss match")

        backbone_features: CSPDarknetStageFeatures = self.backbone.forward(x)

        neck_features: PAFPNOutputFeatures = self.neck.forward(backbone_features)

        return YOLOXOutput(
            small=self.heads[0](neck_features[0]),
            medium=self.heads[1](neck_features[1]),
            large=self.heads[2](neck_features[2])
        )


def _test():
    import time
    t0 = time.time()
    module = YOLOX_L()
    print(f"Module created in {(time.time() - t0) * 1000:.2f} ms")
    t0 = time.time()
    features = module(torch.zeros((1, 3, 640, 640)))
    print(f"Forward in {(time.time() - t0) * 1000:.2f} ms")
    for name, res in features:
        print(f"{name}: ")
        for _name, feature in res:
            print(f"  {_name}: {tuple(feature.shape) if feature is not None else 'None'}")
        print()


if __name__ == "__main__":
    _test()
