# See: https://github.com/open-mmlab/mmyolo/tree/main/configs/yolox
from dataclasses import dataclass, field, fields
from typing import Optional

import torch

from yolox.modules import ConvModule


@dataclass
class BBoxArchResult:
    cls: Optional[torch.Tensor] = field(default=None)  # [N, num_classes, H, W]
    reg: Optional[torch.Tensor] = field(default=None)  # [N, 4, H, W]
    obj: Optional[torch.Tensor] = field(default=None)  # [N, 1, H, W]

    def __post_init__(self):
        self._data = (self.cls, self.reg, self.obj)

    def __getitem__(self, idx):
        return self._data[idx]

    def __iter__(self) -> iter:
        self: dataclass  # 这是给若智PyCharm的静态分析看的
        for f in fields(self):
            yield f.name, getattr(self, f.name)

    def __len__(self) -> int:
        return len(self._data)


class YOLOXHead(torch.nn.Module):
    def __init__(
            self,
            in_channels: int = 256,
            num_classes: int = 80,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.cls_conv = torch.nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            ConvModule(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        self.cls_out = torch.nn.Conv2d(
            in_channels=256,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.box_conv = torch.nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            ConvModule(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        self.reg_out = torch.nn.Conv2d(
            in_channels=256,
            out_channels=4,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.obj_out = torch.nn.Conv2d(
            in_channels=256,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x: torch.Tensor) -> BBoxArchResult:
        cls_feature = self.cls_conv(x)
        box_feature = self.box_conv(x)

        cls_res = self.cls_out(cls_feature)
        reg_res = self.reg_out(box_feature)
        obj_res = self.obj_out(box_feature)

        return BBoxArchResult(cls_res, reg_res, obj_res)


def _test():
    import time
    t0 = time.time()
    module = YOLOXHead()
    print(f"Module created in {(time.time() - t0) * 1000:.2f} ms")
    t0 = time.time()
    features = module(torch.zeros((1, 256, 80, 80)))
    print(f"Forward in {(time.time() - t0) * 1000:.2f} ms")
    for name, feature in features:
        print(f"{name}: {feature.shape if feature is not None else 'None'}")


if __name__ == "__main__":
    _test()
