import torch

from yolox.models import YOLOXOutput
from yolox.heads import BBoxArchResult


def parse_yolox_output(
        yolox_output: YOLOXOutput,
        img_height: int = 640,
        img_width: int = 640,
) -> torch.Tensor:
    output_list = []

    for _, bbox_arch_result in yolox_output:
        bbox_arch_result: BBoxArchResult
        if bbox_arch_result is None:
            raise ValueError("bbox_arch_result is None")

        cls = bbox_arch_result.cls  # [N, NC, H, W]
        reg = bbox_arch_result.reg  # [N, 4, H, W]
        obj = bbox_arch_result.obj  # [N, 1, H, W]

        if cls is None or reg is None or obj is None:
            raise ValueError("cls, reg or obj is None")

        batch_size, num_classes, height, width = cls.shape

        # 计算步长
        x_stride = img_height // height
        y_stride = img_width // width

        # 生成网格
        x_shift = torch.arange(width, device=cls.device)
        y_shift = torch.arange(height, device=cls.device)
        y_grid, x_grid = torch.meshgrid(y_shift, x_shift, indexing='ij')

        # 处理中心点坐标
        x = (x_grid.reshape(1, 1, height, width) + torch.sigmoid(reg[:, 0:1])) * x_stride
        y = (y_grid.reshape(1, 1, height, width) + torch.sigmoid(reg[:, 1:2])) * y_stride

        # 处理宽高
        w = torch.exp(reg[:, 2:3]) * x_stride
        h = torch.exp(reg[:, 3:4]) * y_stride

        # 计算边界框坐标
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x1 + w
        y2 = y1 + h

        # obj: [N, 1, H, W] -> [N, H*W, 1]
        obj = obj.permute(0, 2, 3, 1).reshape(batch_size, -1, 1)

        # cls: [N, NC, H, W] -> [N, H*W, NC]
        cls = cls.permute(0, 2, 3, 1).reshape(batch_size, -1, num_classes)

        # boxes: [N, 4, H, W] -> [N, H*W, 4]
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        boxes = boxes.reshape(batch_size, -1, 4)

        # 合并当前特征层的预测结果
        output = torch.cat([obj, cls, boxes], dim=2)
        output_list.append(output)

    # 合并所有特征层的预测结果
    return torch.cat(output_list, dim=1)  # [N, 8400, 1+NC+4]
