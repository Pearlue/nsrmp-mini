"""Neuro-symbolic model (paper Sec. III): concepts + Action Simulator + program execution."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from nsrmp.data import ANY, CONCEPTS, FEATURE_DIM

GAMMA, TAU = 0.2, 0.1  # Filter threshold and temperature (NS-CL style)
N_REAL_CONCEPTS = len(CONCEPTS) - 1  # every concept except the ANY wildcard


class Concepts(nn.Module):
    """Learned concept embeddings (colours, shapes) and relation classifiers."""

    def __init__(self, feat_dim=FEATURE_DIM, dim=32):
        super().__init__()
        self.proj = nn.Linear(feat_dim, dim)              # W: object feature -> concept space
        self.emb = nn.Embedding(N_REAL_CONCEPTS, dim)     # e_c
        # pair input: both boxes and their (scaled-up) difference, so tiny offsets are easy to see
        self.rel_mlp = nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 4))

    def filter_scores(self, feats, concept_ids):
        """feats (B, N, F), concept_ids (B,) -> sigma((cos(W f_i, e_c) - gamma) / tau), shape (B, N)."""
        v = F.normalize(self.proj(feats), dim=-1)
        e = F.normalize(self.emb(concept_ids.clamp(max=N_REAL_CONCEPTS - 1)), dim=-1)
        score = torch.sigmoid((torch.einsum("bnd,bd->bn", v, e) - GAMMA) / TAU)
        is_any = (concept_ids == CONCEPTS.index(ANY))[:, None]
        return torch.where(is_any, torch.ones_like(score), score)

    def relation_probs(self, boxes, rel_ids):
        """boxes (B, N, 4), rel_ids (B,) -> probs[b, i, j] = P(object i is rel of object j)."""
        N = boxes.shape[1]
        bi, bj = boxes[:, :, None, :].expand(-1, -1, N, -1), boxes[:, None, :, :].expand(-1, N, -1, -1)
        logits = self.rel_mlp(torch.cat([bi, bj, 10 * (bi - bj)], dim=-1))  # (B, N, N, 4)
        idx = rel_ids[:, None, None, None].expand(-1, N, N, 1)
        return torch.sigmoid(logits.gather(-1, idx)[..., 0])
