"""Neuro-symbolic model (paper Sec. III): concepts + Action Simulator + program execution."""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nsrmp.data import ANY, CONCEPTS, FEATURE_DIM
from nsrmp.execution import apply_move, execute_steps
from nsrmp.executor import Executor
from nsrmp.learned_lr import RolePolicy

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


class ActionSimulator(nn.Module):
    """Paper Sec. III-C: (action, subject box, reference box) -> new subject box.

    The MLP predicts an offset added to the subject box, which makes "stay the same size" the
    easy default to learn.
    """

    def __init__(self, n_actions=4, emb_dim=16, hidden=128):
        super().__init__()
        self.action_emb = nn.Embedding(n_actions, emb_dim)
        self.mlp = nn.Sequential(nn.Linear(emb_dim + 8, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4))

    def forward(self, action_emb, subj_box, ref_box):
        return subj_box + self.mlp(torch.cat([action_emb, subj_box, ref_box], dim=-1))


class NSModel(nn.Module):
    """Parser-given programs executed softly; trained end to end from final boxes only."""

    def __init__(self, *, sim=None):
        super().__init__()
        self.concepts = Concepts()
        self.sim = ActionSimulator() if sim is None else sim
        self.executor = Executor(self.concepts)

    def _ground_step(self, moves, feats, boxes, mask):
        subject, reference, action = self.executor.run_step(moves, feats, boxes, mask)
        return subject, reference, self.sim.action_emb(action)

    def execute(self, scene, program, *, move_mode):
        return execute_steps(scene, program, self.sim, self._ground_step, move_mode=move_mode)

    def forward(self, batch):
        """Returns predicted final boxes (B, N, 4) and subject weights per step (B, S, N)."""
        boxes, subj_ws = batch.boxes, []
        for k in range(batch.subjects.shape[1]):
            moves = [prog[k] if len(prog) > k else None for prog in batch.programs]
            subj, ref, action = self.executor.run_step(moves, batch.feats, boxes, batch.mask)
            boxes = apply_move(self.sim, boxes, subj, ref, self.sim.action_emb(action))
            subj_ws.append(subj)
        return boxes, torch.stack(subj_ws, dim=1)


@dataclass
class LearnedOutput:
    pred_boxes: torch.Tensor
    subject_weights: torch.Tensor
    programs: list
    role_traces: list
    log_prob: torch.Tensor
    local_entropy: torch.Tensor


class ParseNSModel(NSModel):
    def __init__(self):
        nn.Module.__init__(self)
        self.parser = RolePolicy()
        self.concepts = Concepts()
        self.sim = ActionSimulator()
        self.executor = Executor(self.concepts)

    # Unlike NSModel, flatten K parser samples across B inputs into K*B paths,
    # so each step uses one batched Executor call instead of K separate calls.
    def _execute(self, inputs, samples):
        # K is the number of program samples drawn from the neural parser for each instruction.
        K, B = len(samples), len(inputs)
        S, N = inputs.step_mask.shape[1], inputs.boxes.shape[1]
        boxes = inputs.boxes.unsqueeze(0).expand(K, -1, -1, -1).reshape(K * B, N, 4)
        feats = inputs.feats.unsqueeze(0).expand(K, -1, -1, -1).reshape(K * B, N, -1)
        mask = inputs.object_mask.unsqueeze(0).expand(K, -1, -1).reshape(K * B, N)
        weights = []
        for s in range(S):
            moves = [samples[k].programs[b][s] if inputs.step_mask[b, s] else None
                     for k in range(K) for b in range(B)]
            subj, ref, action = self.executor.run_step(moves, feats, boxes, mask)
            boxes = apply_move(self.sim, boxes, subj, ref, self.sim.action_emb(action))
            weights.append(subj)
        return boxes.reshape(K, B, N, 4), torch.stack(weights, dim=1).reshape(K, B, S, N)

    # Issue #1 (Execute the program): run sampled programs through the existing
    # Executor and Action Simulator, chaining predicted scenes across steps.
    def forward(self, inputs, *, num_samples=1, greedy=True):
        """inputs: ParserInputs, num_samples: int, greedy: bool
        -> LearnedOutput(pred_boxes: Tensor[K, B, N, 4], subject_weights: Tensor[K, B, S, N],
                         programs, role_traces, log_prob: Tensor[K, B], local_entropy: Tensor[K, B])
        """
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if greedy and num_samples != 1:
            raise ValueError("greedy inference requires num_samples=1")
        # Sample K candidate program sets.
        samples = [self.parser.sample(inputs, greedy=greedy) for _ in range(num_samples)]

        # Execute the sampled programs.
        boxes, weights = self._execute(inputs, samples)

        # Collect predictions and sampling statistics.
        decisions = (inputs.keyword_mask & inputs.step_mask[..., None]).sum(dim=(1, 2)).clamp(min=1)
        log_prob = torch.stack([sample.log_prob for sample in samples])
        entropy = torch.stack([sample.entropy for sample in samples]) / decisions
        return LearnedOutput(boxes, weights, [sample.programs for sample in samples],
                             [sample.role_traces for sample in samples], log_prob, entropy)
