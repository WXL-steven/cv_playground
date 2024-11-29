import torch


def bce_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        epsilon: float = 1e-7
) -> torch.Tensor:
    """
    Usage1:
    pred = torch.rand(1, 8400, 80)  # 假设有80个类别
    target_idx = 35  # 假设第35个类别为目标类别
    target = torch.zeros(1, 8400, 80)  # 那么先生成全零矩阵(负标签)
    target[:, :, target_idx] = 1  # 然后将目标类别设为1(正标签)
    """
    # 确保target可以广播到pred的形状
    assert target.shape[-1] == pred.shape[-1], "Last dimension must match"
    assert all(s1 <= s2 for s1, s2 in zip(target.shape[::-1], pred.shape[::-1])), \
        "Target shape must be broadcastAble to pred shape"

    pred = torch.sigmoid(pred)

    pred = torch.clamp(pred, epsilon, 1 - epsilon)
    target = torch.clamp(target, epsilon, 1 - epsilon)

    return -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))


def iou_loss(
        pred: torch.Tensor,  # (x, y, w, h)
        target: torch.Tensor  # (xmin, ymin, xmax, ymax)
) -> torch.Tensor:
    """
    Usage1:
    pred = torch.rand(1, 8400, 4)  # 假设有8400个锚框
    pred[..., :2] *= 640
    pred[..., 2:] *= 640
    target = torch.tensor([100, 100, 200, 200])
    """
    # 单次只接收一个真实框
    assert target.shape == (4,), "Input target should be 1-dim"
    # 确保最后一维元素数为4
    assert pred.shape[-1] == 4, "Input pred last dim should be 4"

    pred_xyxy = torch.zeros_like(pred)
    pred_xyxy[..., :2] = pred[..., :2] - pred[..., 2:] / 2
    pred_xyxy[..., 2:] = pred[..., :2] + pred[..., 2:] / 2

    # eg:
    # pred: [NumAnchors, 4]
    # target: [4]

    pred_area = pred[..., 2] * pred[..., 3]  # [NumAnchors]
    target_area = (target[2] - target[0]) * (target[3] - target[1])

    lt = torch.max(pred_xyxy[..., :2], target[:2])  # [NumAnchors, 2]
    rb = torch.min(pred_xyxy[..., 2:], target[2:])  # [NumAnchors, 2]

    wh = (rb - lt).clamp(min=0)  # [NumAnchors, 2]
    inter = wh[..., 0] * wh[..., 1]  # [NumAnchors]

    union = pred_area + target_area - inter + 1e-7
    iou = inter / union
    return 1 - iou


def _test_bce():
    pred = torch.rand(1, 8400, 80)  # 假设有80个类别
    target_idx = 35  # 假设第35个类别为目标类别
    target = torch.zeros(1, 8400, 80)  # 那么先生成全零矩阵(负标签)
    target[:, :, target_idx] = 1  # 然后将目标类别设为1(正标签)

    print(bce_loss(pred, target).mean())  # 鉴于是随机生成的,那么期望的损失就是(1-1/80)


def _test_iou():
    img_size = 640  # 假设图像尺寸为640x640
    max_len = 50  # 假设最大边长为50

    pred = torch.rand(1, 8400, 4)  # 假设有8400个锚框
    pred[..., :2] *= img_size
    pred[..., 2:] *= max_len

    target = torch.tensor([100, 100, 200, 200])  # 假设目标框为[100, 100, 200, 200]

    print(iou_loss(pred, target).mean())


if __name__ == '__main__':
    _test_iou()
