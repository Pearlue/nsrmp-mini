import pytest
import torch

from nsrmp.metrics import iou


def box(*v):
    return torch.tensor(v, dtype=torch.float32)


def test_identical_boxes_have_iou_one():
    assert iou(box(0.1, 0.1, 0.3, 0.3), box(0.1, 0.1, 0.3, 0.3)).item() == pytest.approx(1.0)


def test_disjoint_and_edge_touching_boxes_have_iou_zero():
    assert iou(box(0, 0, 0.2, 0.2), box(0.5, 0.5, 0.7, 0.7)).item() == 0.0
    assert iou(box(0, 0, 0.2, 0.2), box(0.2, 0, 0.4, 0.2)).item() == 0.0


def test_half_overlapping_boxes_have_iou_one_third():
    # two unit-height boxes overlapping on half their width: inter 1, union 3
    assert iou(box(0, 0, 2, 1), box(1, 0, 3, 1)).item() == pytest.approx(1 / 3)


def test_degenerate_zero_area_or_inverted_boxes_give_zero_not_nan():
    assert iou(box(0.3, 0.3, 0.3, 0.3), box(0.3, 0.3, 0.3, 0.3)).item() == 0.0
    assert iou(box(0.5, 0.5, 0.4, 0.4), box(0.4, 0.4, 0.5, 0.5)).item() == 0.0


def test_iou_works_on_batches_of_boxes():
    a = torch.stack([box(0, 0, 2, 1), box(0, 0, 1, 1)])
    b = torch.stack([box(1, 0, 3, 1), box(0, 0, 1, 1)])
    assert iou(a, b).tolist() == pytest.approx([1 / 3, 1.0])
