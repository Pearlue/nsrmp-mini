"""Turn a list of Examples into padded tensors the models can consume.

Programs come from the *parser* (not from the gold annotation), as in the paper's setup.
"""
from dataclasses import dataclass

import numpy as np
import torch

from nsrmp.data import COLOURS, SHAPES
from nsrmp.parser import parse, parse_roles, split_steps

VOCAB = (["<pad>", "put", "the", "which", "is", "to", "left", "right", "of", "in", "front",
          "behind", "object"] + COLOURS + SHAPES)
_WORD_ID = {w: i for i, w in enumerate(VOCAB)}
# Word groups the parser gives the parse-matched baseline, one row each in Batch.role_tokens.
ROLE_GROUPS = ("subj_head", "subj_rel", "subj_anchor", "ref_head", "ref_rel", "ref_anchor", "action")


@dataclass
class Batch:
    feats: torch.Tensor     # (B, N, 32)
    boxes: torch.Tensor     # (B, N, 4) initial boxes
    final: torch.Tensor     # (B, N, 4) gold final boxes
    mask: torch.Tensor      # (B, N) 1 for real objects, 0 for padding
    moved: torch.Tensor     # (B, N) bool, gold final box differs from the initial one
    subjects: torch.Tensor  # (B, S) gold subject per step, -1 for missing steps
    n_steps: torch.Tensor   # (B,)
    is_complex: torch.Tensor  # (B,) bool
    tokens: torch.Tensor    # (B, S, L) word ids per step (for the baseline)
    lengths: torch.Tensor   # (B, S)
    role_tokens: torch.Tensor  # (B, S, len(ROLE_GROUPS), Lr) word ids per parsed role, 0 = padding
    programs: list          # parsed program (tuple of Moves) per example

    def __len__(self):
        return len(self.programs)

    def select(self, idx):
        idx = torch.as_tensor(idx, dtype=torch.long)
        fields = {name: (value[idx] if torch.is_tensor(value) else [value[i] for i in idx.tolist()])
                  for name, value in self.__dict__.items()}
        return Batch(**fields)


def _role_groups(clause):
    r = parse_roles(clause)
    return [*r["subject"], *r["reference"], r["action"]]


def collate(examples):
    B = len(examples)
    N = max(len(e.objs) for e in examples)
    S = max(e.n_steps for e in examples)
    clauses = [split_steps(e.text) for e in examples]
    L = max(len(c.replace(",", " ").split()) for cs in clauses for c in cs)
    feats, boxes, final = np.zeros((B, N, 32)), np.zeros((B, N, 4)), np.zeros((B, N, 4))
    mask, subjects = np.zeros((B, N)), -np.ones((B, S), dtype=np.int64)
    tokens, lengths = np.zeros((B, S, L), dtype=np.int64), np.zeros((B, S), dtype=np.int64)
    roles = [[_role_groups(c) for c in cs] for cs in clauses]
    Lr = max(len(g) for rs in roles for r in rs for g in r)
    role_tokens = np.zeros((B, S, len(ROLE_GROUPS), Lr), dtype=np.int64)
    for k, e in enumerate(examples):
        n = len(e.objs)
        feats[k, :n], mask[k, :n] = e.feats, 1
        boxes[k, :n] = [o.box for o in e.objs]
        final[k, :n] = e.final_boxes
        subjects[k, :e.n_steps] = e.subjects
        for s, clause in enumerate(clauses[k]):
            words = clause.replace(",", " ").split()
            tokens[k, s, :len(words)] = [_WORD_ID[w] for w in words]
            lengths[k, s] = len(words)
            for g, group in enumerate(roles[k][s]):
                role_tokens[k, s, g, :len(group)] = [_WORD_ID[w] for w in group]
    boxes_t, final_t = torch.tensor(boxes, dtype=torch.float32), torch.tensor(final, dtype=torch.float32)
    return Batch(
        feats=torch.tensor(feats, dtype=torch.float32), boxes=boxes_t, final=final_t,
        mask=torch.tensor(mask, dtype=torch.float32),
        moved=((boxes_t - final_t).abs().sum(-1) > 1e-6) & (torch.tensor(mask) > 0),
        subjects=torch.tensor(subjects), n_steps=torch.tensor([e.n_steps for e in examples]),
        is_complex=torch.tensor([e.is_complex for e in examples]),
        tokens=torch.tensor(tokens), lengths=torch.tensor(lengths),
        role_tokens=torch.tensor(role_tokens),
        programs=[parse(e.text) for e in examples])
