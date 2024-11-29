from dataclasses import dataclass, field
from typing import Any

import torch

from yolox.losses import bce_loss, iou_loss


# 扣了一下午头总算是把这玩意搞明白了，至少是跑通了看起来没什么大问题。就请不要使用此函数了，我晚点回来会用正常一点的方式来重新写一遍。
def _simota_assigner_legacy(
        pred: torch.Tensor,
        target: torch.Tensor,
        theta: float = 3,
        center_radius: float = 2.5
) -> Any:
    # pred: [1, num_anchors, 1 + 4 + num_classes]
    # pred: [1, 8400, 1 + 4 + 80]
    # target: [1, num_targets, 1 + 4]
    # target: [1, 10, 5]

    num_gt = target.shape[1]
    num_classes = pred.shape[2] - 5
    num_anchors = pred.shape[1]

    cls_loss = torch.zeros(num_gt, num_anchors)  # [10, 8400]
    pred_cls = pred[..., -num_classes:]  # [1, 8400, 80]
    for i in range(num_gt):
        gt_cls = torch.zeros(1, num_anchors, num_classes)  # [1, 8400, 80]
        target_idx = target[0, i, 0].long()
        gt_cls[0, :, target_idx] = 1
        current_cls_loss = bce_loss(pred_cls, gt_cls)  # [1, 8400, 80]
        cls_loss[i] = current_cls_loss.sum(dim=2).squeeze()  # [8400]

    reg_loss = torch.zeros(num_gt, num_anchors)  # [10, 8400]
    pred_reg = pred[..., 1:5]  # [1, 8400, 4]
    for i in range(num_gt):
        target_reg = target[0, i, 1:]  # [4]
        current_reg_loss = iou_loss(pred_reg, target_reg)  # [1, 8400]
        reg_loss[i] = current_reg_loss.squeeze()  # [8400]

    cost_matrix = cls_loss + reg_loss * theta

    valid_mask = torch.zeros(num_gt, num_anchors)  # [10, 8400]
    pred_reg = pred_reg  # [1, 8400, 4]  (x, y, w, h)
    for i in range(num_gt):
        target_reg = target[0, i, 1:]  # [4] (xmin, ymin, xmax, ymax)
        _w = target_reg[2] - target_reg[0]
        _h = target_reg[3] - target_reg[1]
        _x = target_reg[0] + _w / 2
        _y = target_reg[1] + _h / 2
        _w *= center_radius
        _h *= center_radius
        _l = _x - _w / 2
        _t = _y - _h / 2
        _r = _x + _w / 2
        _b = _y + _h / 2

        valid_mask[i] = ((pred_reg[..., 0] > _l) &
                         (pred_reg[..., 0] < _r) &
                         (pred_reg[..., 1] > _t) &
                         (pred_reg[..., 1] < _b)).squeeze()  # [8400]

        # print(f"SimOTA_assigner valid_mask for {i}: {valid_mask[i].sum().item()}")

    cost_matrix = cost_matrix  # [10, 8400]
    assignments = []
    for i in range(num_gt):
        current_valid_pred_cost_idx = torch.where(valid_mask[i] == 1)[0]
        num_valid_pred = len(current_valid_pred_cost_idx)
        current_valid_pred_cost = cost_matrix[i, current_valid_pred_cost_idx]  # [valid_mask[i].sum()]
        _cost_max = current_valid_pred_cost.max()
        _cost_min = current_valid_pred_cost.min()
        sim_score = ((_cost_max - current_valid_pred_cost) / (_cost_max - _cost_min + 1e-7)).mean().item()
        k = min(num_valid_pred, round(num_valid_pred * sim_score))
        print(f"SimOTA_assigner k for {i}: {num_valid_pred} * {sim_score} = {round(num_valid_pred * sim_score)}")

        low_k_idx = torch.topk(current_valid_pred_cost, k, largest=False)[1]  # [k]
        low_k_idx = current_valid_pred_cost_idx[low_k_idx]
        print(f"Top 3 SimOTA_assigner assignments for {i}: {low_k_idx[:3]}")

        assignments.append(low_k_idx)

    return assignments


@dataclass(frozen=True, slots=True)
class SimOTAAssignerResult:
    assignments: tuple[torch.Tensor, ...] = field()


# TODO: 写了一天然后发现好像和论文上的不太一样，真的无语，先用这个吧，跑起来再说。
# 主要是tok-k选取的部分不同，但是如果要改就最好获取到原始的网格输出，就是在`yolox/models.py`里定义的那个YOLOXOutput
# 然后再根据所归属的网格再次筛选
def simple_sim_ota_assigner(
        pred: torch.Tensor,
        target: torch.Tensor,
        theta: float = 3.,
        center_radius: float = 1.5
) -> SimOTAAssignerResult:
    assert len(pred.shape) in (2, 3), f"Expected 2D or 3D(with batch is 1) tensor for pred, got {pred.shape}."
    assert len(target.shape) in (2, 3), f"Expected 2D or 3D(with batch is 1) tensor for target, got {target.shape}."
    # 判断格式
    if len(pred.shape) == 3:
        assert pred.shape[0] == 1, "Only support batch size 1"
        pred = pred[0]  # [NumAnchors, 1 + 4 + NumClasses]
    if len(target.shape) == 3:
        assert target.shape[0] == 1, "Only support batch size 1"
        target = target[0]  # [NumTargets, 1 + 4]

    # 解析数据
    num_targets = target.shape[0]
    num_classes = pred.shape[1] - 5
    pred_cls = pred[..., 5:]  # [NumAnchors, NumClasses]
    pred_reg = pred[..., 1:5]  # [NumAnchors, 4]
    # obj在分配中不参与计算就跳过了

    # 逐一处理
    # ps: 选择逐一处理和向量化处理类似于在性能和内存上抉择，选择逐一处理只是因为一般真实框很少
    assignments = []
    for i in range(num_targets):
        # 准备真实框相关数据
        target_reg = target[i, 1:]  # [4] (xmin, ymin, xmax, ymax)
        target_cls = torch.zeros(num_classes)  # [NumClasses] 后续利用自动广播而无需为每一个锚点单独分配
        target_cls[target[i, 0].long()] = 1

        # 先筛选候选框,大幅减少cost计算
        target_wh = target_reg[2:] - target_reg[:2]  # [w, h]
        target_center = target_reg[:2] + target_wh / 2  # [cx, cy]
        center_box = torch.cat([
            target_center - center_radius * target_wh / 2,  # [l, t]
            target_center + center_radius * target_wh / 2  # [r, b]
        ])

        # 使用 & 运算符直接进行布尔运算
        valid_mask = (pred_reg[..., 0] > center_box[0]) & \
                     (pred_reg[..., 1] > center_box[1]) & \
                     (pred_reg[..., 2] < center_box[2]) & \
                     (pred_reg[..., 3] < center_box[3])

        candidate_idx = torch.where(valid_mask == 1)[0]
        candidate_pred_cls = pred_cls[candidate_idx]  # [NumCandidate, NumClasses]
        candidate_pred_reg = pred_reg[candidate_idx]  # [NumCandidate, 4]
        num_candidate = candidate_idx.shape[0]

        # 剪枝
        if num_candidate == 0:
            assignments.append(torch.zeros(0, dtype=torch.int64))
            continue

        # 计算二元交叉熵损失
        cost = bce_loss(pred=candidate_pred_cls, target=target_cls).sum(dim=-1)  # [NumCandidate]

        # 计算IoU损失
        cost += iou_loss(pred=candidate_pred_reg, target=target_reg) * theta  # [NumCandidate]

        # 使用简化的方式计算k值
        cost_max = cost.max()
        cost_min = cost.min()
        sim_score = ((cost_max - cost) / (cost_max - cost_min + 1e-9)).mean().item()
        k = max(min(num_candidate, round(num_candidate * sim_score)), 1)
        print(f"SimOTA_assigner k for {i}: {num_candidate} * {sim_score} = {round(num_candidate * sim_score)}")

        # 选择后k个锚框
        _, pos_idx = cost.topk(k=k, dim=0, largest=False)

        assignments.append(candidate_idx[pos_idx])

    return SimOTAAssignerResult(tuple(assignments))


def _dev():
    from yolox.utils import parse_yolox_output
    from yolox.models import YOLOX_L

    module = YOLOX_L(num_classes=80)
    features = module(torch.rand((1, 3, 640, 640)))

    pred = parse_yolox_output(features)

    print(pred.shape)

    target = torch.zeros(1, 10, 5)
    target[..., 0] = torch.randint(0, 80, (10,))
    target_reg_xy = torch.rand(10, 2) * (640 - 100)
    target_reg_wh = torch.rand(10, 2) * 45 + 5
    target[..., 1] = target_reg_xy[:, 0] - target_reg_wh[:, 0]
    target[..., 2] = target_reg_xy[:, 1] - target_reg_wh[:, 1]
    target[..., 3] = target_reg_xy[:, 0] + target_reg_wh[:, 0]
    target[..., 4] = target_reg_xy[:, 1] + target_reg_wh[:, 1]
    # target[..., 1:] *= 320
    # target[..., 3:5] += 320

    print(target.max())

    simple_sim_ota_assigner(pred, target)


if __name__ == "__main__":
    _dev()
