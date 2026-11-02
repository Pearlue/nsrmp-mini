import numpy as np
import pytest
import torch

from nsrmp.batch import collate
from nsrmp.data import ANY, ACTIONS, RELATIONS, Obj, Example, noun_phrase_program, make_dataset
from nsrmp.executor import Executor
from nsrmp.model import Concepts
from nsrmp.data import feature_matrix, COLOURS, SHAPES


class PerfectConcepts(Concepts):
    """Hand-set concepts: noise-free features decode back to one-hots, relations are exact."""

    def __init__(self):
        super().__init__(feat_dim=32, dim=9)
        with torch.no_grad():
            self.proj.weight.copy_(torch.tensor(np.linalg.pinv(feature_matrix()), dtype=torch.float32))
            self.proj.bias.zero_()
            self.emb.weight.copy_(torch.eye(9))

    def relation_probs(self, boxes, rel_ids):
        c = torch.stack([(boxes[..., 0] + boxes[..., 2]) / 2, (boxes[..., 1] + boxes[..., 3]) / 2], -1)
        d = c[:, :, None, :] - c[:, None, :, :]           # d[b,i,j] = centre_i - centre_j
        signed = torch.stack([-d[..., 0], d[..., 0], d[..., 1], -d[..., 1]], -1)  # left,right,front,behind
        return torch.sigmoid(1000 * signed.gather(-1, rel_ids[:, None, None, None].expand(-1, d.shape[1], d.shape[2], 1))[..., 0])


def scene(specs):
    """specs: list of (colour, shape, (cx, cy)); returns a one-example Batch with clean features."""
    objs = [Obj(c, s, (x - 0.05, y - 0.05, x + 0.05, y + 0.05)) for c, s, (x, y) in specs]
    onehots = np.zeros((len(objs), 9))
    for i, o in enumerate(objs):
        onehots[i, COLOURS.index(o.colour)] = 1
        onehots[i, 6 + SHAPES.index(o.shape)] = 1
    feats = onehots @ feature_matrix().T
    ex = Example(objs, feats, "put the red cube to the left of the blue dice",
                 (("Move", "MoveLeft", noun_phrase_program("red", "cube"), noun_phrase_program("blue", "dice")),),
                 [0], [1], np.array([o.box for o in objs]), False)
    return collate([ex])


def run(prog, b):
    ex = Executor(PerfectConcepts())
    return ex.run_noun_phrase(prog, b.feats, b.boxes, b.mask)[0]


def test_filter_and_unique_select_the_red_cube_among_distractors():
    b = scene([("red", "dice", (0.2, 0.2)), ("red", "cube", (0.5, 0.5)), ("blue", "cube", (0.8, 0.8))])
    w = run(noun_phrase_program("red", "cube"), b)
    assert w.argmax().item() == 1 and w[1].item() > 0.7


def test_wildcard_concept_keeps_every_object_matching_the_other_attribute():
    b = scene([("red", "dice", (0.2, 0.2)), ("blue", "cube", (0.5, 0.5)), ("green", "cube", (0.8, 0.8))])
    w = run(noun_phrase_program(ANY, "dice"), b)
    assert w.argmax().item() == 0 and w[0].item() > 0.7
    both = run(("Unique", ("Filter", ("Filter", ("Scene",), ANY), "cube")), b)
    assert both[1].item() == pytest.approx(0.45, abs=0.1) and both[2].item() == pytest.approx(0.45, abs=0.1)


def test_relate_picks_the_lego_that_is_behind_the_green_dice():
    # two legos; only the one with smaller y (behind) the green dice qualifies
    b = scene([("red", "lego", (0.5, 0.2)), ("green", "dice", (0.5, 0.5)), ("blue", "lego", (0.5, 0.8))])
    anchor = noun_phrase_program("green", "dice")
    behind = noun_phrase_program(ANY, "lego", source=("Relate", anchor, "behind"))
    front = noun_phrase_program(ANY, "lego", source=("Relate", anchor, "front"))
    assert run(behind, b).argmax().item() == 0 and run(behind, b)[0].item() > 0.7
    assert run(front, b).argmax().item() == 2 and run(front, b)[2].item() > 0.7


def test_relate_ignores_the_anchor_object_itself():
    b = scene([("green", "dice", (0.5, 0.5)), ("red", "lego", (0.1, 0.1))])
    related = ("Relate", noun_phrase_program("green", "dice"), "left")
    w = run(related, b)
    assert w[0].item() < 0.05 and w[1].item() > 0.7


def test_batched_steps_with_different_program_shapes_match_one_at_a_time_execution():
    exs = make_dataset(24, seed=11, n_objects=(4, 6))
    b = collate(exs)
    ex = Executor(PerfectConcepts())
    subj, ref, action = ex.run_step([p[0] for p in b.programs], b.feats, b.boxes, b.mask)
    for k in range(len(exs)):
        one = b.select([k])
        s1, r1, a1 = ex.run_step([one.programs[0][0]], one.feats, one.boxes, one.mask)
        n = len(exs[k].objs)
        assert torch.allclose(subj[k, :n], s1[0, :n], atol=1e-5)
        assert torch.allclose(ref[k, :n], r1[0, :n], atol=1e-5)
        assert action[k].item() == a1[0].item() == ACTIONS.index(exs[k].program[0][1])
        assert subj[k].argmax().item() == exs[k].subjects[0] and ref[k].argmax().item() == exs[k].refs[0]
