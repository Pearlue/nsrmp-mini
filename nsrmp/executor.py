"""Soft program executor (paper Sec. III-B Visual Reasoner, with NS-CL-style soft execution).

A program is a nested tuple of DSL operators (see data.py). Every operator maps to a tensor of
object weights p in [0, 1]^N, one per (padded) object:

    Scene            -> 1 for every real object
    Filter(p, c)     -> p * sigmoid((cos(W f_i, e_c) - gamma) / tau)
    Relate(w, r)     -> sum_j w_j * sigmoid(logit_r(box_i, box_j))   (self pairs masked)
    Unique(p)        -> p / sum(p)   (same as softmax(log p))

Examples in a batch have different programs, so `run_step` groups examples whose programs have
the same shape (they differ only in concept names) and runs each group as one tensor operation.
"""
from collections import defaultdict

import torch

from nsrmp.data import ACTIONS, CONCEPTS, RELATIONS

_LEAF_VOCAB = {"Filter": CONCEPTS, "Relate": RELATIONS, "Move": ACTIONS}


def _skeleton(prog):
    """The program with its concept/relation/action names erased."""
    return (prog[0],) + tuple(_skeleton(a) if isinstance(a, tuple) else "_" for a in prog[1:])


def _stack(progs):
    """Programs with identical skeletons -> one program whose leaves are id tensors (one per example)."""
    first, args = progs[0], []
    for k in range(1, len(first)):
        if isinstance(first[k], tuple):
            args.append(_stack([p[k] for p in progs]))
        else:
            vocab = _LEAF_VOCAB[first[0]]
            args.append(torch.tensor([vocab.index(p[k]) for p in progs]))
    return (first[0], *args)


class Executor:
    def __init__(self, concepts):
        self.concepts = concepts

    def _run(self, prog, feats, boxes, mask):
        op = prog[0]
        if op == "Scene":
            return mask
        if op == "Unique":
            p = self._run(prog[1], feats, boxes, mask)
            return p / (p.sum(-1, keepdim=True) + 1e-6)
        if op == "Filter":
            p = self._run(prog[1], feats, boxes, mask)
            return p * self.concepts.filter_scores(feats, prog[2])
        if op == "Relate":
            w = self._run(prog[1], feats, boxes, mask)
            probs = self.concepts.relation_probs(boxes, prog[2])  # probs[b, i, j]: i is r of j
            eye = torch.eye(boxes.shape[1], device=boxes.device)
            return torch.einsum("bj,bij->bi", w, probs * (1 - eye)) * mask
        raise ValueError(op)

    def run_noun_phrase(self, prog, feats, boxes, mask):
        """Run one program (with string leaves) on every example of the batch."""
        return self._run(_stack([prog] * feats.shape[0]), feats, boxes, mask)

    def run_step(self, moves, feats, boxes, mask):
        """Run one Move per example (None = no such step). Returns subject weights (B, N),
        reference weights (B, N) and action ids (B,). Missing steps get all-zero weights."""
        B, N = mask.shape
        subj, ref = torch.zeros(B, N), torch.zeros(B, N)
        action = torch.zeros(B, dtype=torch.long)
        groups = defaultdict(list)
        for k, move in enumerate(moves):
            if move is not None:
                groups[_skeleton(move)].append(k)
        for idx in groups.values():
            _, a, s_prog, r_prog = _stack([moves[k] for k in idx])
            f, b, m = feats[idx], boxes[idx], mask[idx]
            subj = subj.index_put((torch.tensor(idx),), self._run(s_prog, f, b, m))
            ref = ref.index_put((torch.tensor(idx),), self._run(r_prog, f, b, m))
            action[idx] = a
        return subj, ref, action
