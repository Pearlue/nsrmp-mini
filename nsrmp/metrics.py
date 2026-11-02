"""Evaluation metrics (paper Sec. IV-B): IOU, IOU-M, subject accuracy, placement accuracy."""
import torch


def iou(a, b):
    """Intersection-over-union of boxes (..., 4) given as (x1, y1, x2, y2). Differentiable."""
    inter_w = (torch.minimum(a[..., 2], b[..., 2]) - torch.maximum(a[..., 0], b[..., 0])).clamp(min=0)
    inter_h = (torch.minimum(a[..., 3], b[..., 3]) - torch.maximum(a[..., 1], b[..., 1])).clamp(min=0)
    inter = inter_w * inter_h
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    union = area_a + area_b - inter
    return torch.where(union > 0, inter / union.clamp(min=1e-9), torch.zeros_like(union))
