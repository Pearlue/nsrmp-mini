"""Purely neural baseline, NMN+-like (paper Sec. IV-B "NMN+" comparison).

It sees the same object features/boxes and uses the same Action Simulator and loss as the
neuro-symbolic model, but reads the instruction with a GRU and never builds a program.
Each object is scored on its own (feature, box, sentence), so relations between objects have to
be squeezed into the sentence vector.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from nsrmp.batch import VOCAB
from nsrmp.data import ACTIONS, FEATURE_DIM
from nsrmp.model import ActionSimulator, apply_move


def _scorer(hidden):
    return nn.Sequential(nn.Linear(FEATURE_DIM + 4 + hidden, 64), nn.ReLU(), nn.Linear(64, 1))


class NeuralBaseline(nn.Module):
    def __init__(self, word_dim=32, hidden=64):
        super().__init__()
        self.words = nn.Embedding(len(VOCAB), word_dim, padding_idx=0)
        self.gru = nn.GRU(word_dim, hidden, batch_first=True)
        self.subject_head, self.reference_head = _scorer(hidden), _scorer(hidden)
        self.action_head = nn.Linear(hidden, len(ACTIONS))
        self.sim = ActionSimulator()

    def encode(self, tokens, lengths):
        """tokens (M, L), lengths (M,) -> sentence vector (M, hidden) = GRU state at the last word."""
        out, _ = self.gru(self.words(tokens))
        last = (lengths - 1).clamp(min=0)
        return out[torch.arange(len(tokens)), last]

    def _attend(self, head, feats, boxes, sent, mask):
        sent = sent[:, None, :].expand(-1, feats.shape[1], -1)
        logits = head(torch.cat([feats, boxes, sent], dim=-1))[..., 0]
        return F.softmax(logits.masked_fill(mask == 0, float("-inf")), dim=-1)

    def forward(self, batch):
        B, S, L = batch.tokens.shape
        sentences = self.encode(batch.tokens.view(B * S, L), batch.lengths.view(B * S)).view(B, S, -1)
        boxes, subj_ws = batch.boxes, []
        for k in range(S):
            active = (batch.n_steps > k).float()[:, None]
            subj = self._attend(self.subject_head, batch.feats, boxes, sentences[:, k], batch.mask) * active
            ref = self._attend(self.reference_head, batch.feats, boxes, sentences[:, k], batch.mask) * active
            action_probs = F.softmax(self.action_head(sentences[:, k]), dim=-1)
            boxes = apply_move(self.sim, boxes, subj, ref, action_probs @ self.sim.action_emb.weight)
            subj_ws.append(subj)
        return boxes, torch.stack(subj_ws, dim=1)
