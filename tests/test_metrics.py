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


class StubModel(torch.nn.Module):
    """Returns canned predictions: either gold boxes, or the unmoved scene."""

    def __init__(self, move, subject_wrong=False):
        super().__init__()
        self.move, self.subject_wrong = move, subject_wrong

    def forward(self, batch):
        boxes = batch.final if self.move else batch.boxes
        B, S, N = batch.subjects.shape[0], batch.subjects.shape[1], batch.mask.shape[1]
        w = torch.zeros(B, S, N)
        for k in range(B):
            for s in range(batch.n_steps[k]):
                w[k, s, (batch.subjects[k, s] + self.subject_wrong) % batch.mask[k].sum().long()] = 1
        return boxes, w


def test_perfect_predictions_score_one_on_every_metric():
    from nsrmp.batch import collate
    from nsrmp.data import make_dataset
    from nsrmp.metrics import evaluate
    r = evaluate(StubModel(move=True), collate(make_dataset(20, seed=0)))
    assert r == {"iou": pytest.approx(1.0), "iou_m": pytest.approx(1.0),
                 "subject_acc": pytest.approx(1.0), "placement_acc": pytest.approx(1.0)}


def test_doing_nothing_keeps_iou_high_but_iou_m_low_and_misses_the_placement():
    from nsrmp.batch import collate
    from nsrmp.data import make_dataset
    from nsrmp.metrics import evaluate
    exs = make_dataset(30, seed=1)
    b = collate(exs)
    r = evaluate(StubModel(move=False), b)
    moved_iou = iou(b.boxes, b.final)[b.moved].mean().item()
    assert r["iou_m"] == pytest.approx(moved_iou)
    assert r["iou"] > 0.7 > r["iou_m"] and r["iou"] == pytest.approx(
        iou(b.boxes, b.final)[b.mask > 0].mean().item())
    assert r["placement_acc"] == 0.0


def test_wrong_subject_is_counted_per_step():
    from nsrmp.batch import collate
    from nsrmp.data import make_dataset
    from nsrmp.metrics import evaluate
    r = evaluate(StubModel(move=True, subject_wrong=True), collate(make_dataset(20, seed=2)))
    assert r["subject_acc"] == 0.0 and r["iou_m"] == pytest.approx(1.0)
