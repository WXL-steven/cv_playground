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
    # pred: [1, num_anchors, 1 + num_classes + 4]
    # pred: [1, 8400, 1 + 80 + 4]
    # target: [1, num_targets, 1 + 4]
    # target: [1, 10, 5]

    num_gt = target.shape[1]
    num_classes = pred.shape[2] - 5
    num_anchors = pred.shape[1]

    cls_loss = torch.zeros(num_gt, num_anchors)  # [10, 8400]
    pred_cls = pred[..., 1:1 + num_classes]  # [1, 8400, 80]
    for i in range(num_gt):
        gt_cls = torch.zeros(1, num_anchors, num_classes)  # [1, 8400, 80]
        target_idx = target[0, i, 0].long()
        gt_cls[0, :, target_idx] = 1
        current_cls_loss = bce_loss(pred_cls, gt_cls)  # [1, 8400, 80]
        cls_loss[i] = current_cls_loss.sum(dim=2).squeeze()  # [8400]

    reg_loss = torch.zeros(num_gt, num_anchors)  # [10, 8400]
    pred_reg = pred[..., -4:]  # [1, 8400, 4]
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
        sim_score = ((_cost_max - current_valid_pred_cost) / (_cost_max - _cost_min)).mean().item()
        k = min(num_valid_pred, round(num_valid_pred * sim_score))
        print(f"SimOTA_assigner k for {i}: {num_valid_pred} * {sim_score} = {round(num_valid_pred * sim_score)}")

        low_k_idx = torch.topk(current_valid_pred_cost, k, largest=False)[1]  # [k]
        low_k_idx = current_valid_pred_cost_idx[low_k_idx]
        print(f"Top 3 SimOTA_assigner assignments for {i}: {low_k_idx[:3]}")

        assignments.append(low_k_idx)

    return assignments


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

    _simota_assigner_legacy(pred, target)


if __name__ == "__main__":
    _dev()
