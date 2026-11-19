from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nsrmp.learned_parser import ROLES, VOCAB, Grammar, role_to_program


@dataclass
class ParserSample:
    programs: list
    role_traces: list
    log_prob: torch.Tensor
    entropy: torch.Tensor


class RolePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB), 32, padding_idx=0)
        self.gru = nn.GRU(32, 64, batch_first=True, bidirectional=True)
        self.head = nn.Linear(128, len(ROLES))

    # Issue #1 (Policy): encode text and predict seven roles for each keyword.
    # surface_context retains context; keywords_only follows the issue's input.
    def forward(self, inputs):
        """inputs: ParserInputs (context IDs/lengths, keyword positions/mask, step mask)
        -> role_logits: Tensor[B, S, W, 7], ordered as ROLES
        """
        B, S, T = inputs.context_ids.shape
        W = inputs.keyword_positions.shape[-1]
        active = inputs.step_mask & (inputs.context_lengths > 0)
        logits = self.head.weight.new_zeros(B, S, W, len(ROLES))
        if not active.any():
            return logits
        embedded = self.emb(inputs.context_ids[active])
        packed = pack_padded_sequence(embedded, inputs.context_lengths[active].cpu(),
                                      batch_first=True, enforce_sorted=False)
        hidden, _ = self.gru(packed)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True, total_length=T)
        mask = inputs.keyword_mask[active]
        positions = inputs.keyword_positions[active].masked_fill(~mask, 0)
        keywords = hidden.gather(1, positions[..., None].expand(-1, -1, 128))
        logits[active] = self.head(keywords).masked_fill(~mask[..., None], 0)
        return logits

    @staticmethod
    def _decode_roles(keywords, scores, greedy, b, s):
        grammar, trace = Grammar(keywords), []
        log_prob = scores.sum() * 0
        entropy = scores.sum() * 0
        for t, (keyword, role_scores) in enumerate(zip(keywords, scores)):
            valid = torch.tensor(grammar.valid_roles(trace),
                                 dtype=torch.bool, device=role_scores.device)
            if not valid.any():
                raise ValueError(f"expected a completable role assignment in "
                                 f"example {b}, step {s}, keyword {t}: "
                                 f"{keywords!r}")
            masked = role_scores.masked_fill(~valid, -float("inf"))
            dist = Categorical(logits=masked)
            selected = masked.argmax() if greedy else dist.sample()
            log_prob = log_prob + dist.log_prob(selected)
            entropy = entropy + dist.entropy()
            role = ROLES[int(selected)]
            trace.append(role)
        return role_to_program(keywords, trace), tuple(trace), log_prob, entropy

    # Issue #1 (Build the program): mask invalid roles before sampling and reuse
    # role_to_program; return the sampled trace's differentiable log probability.
    def sample(self, inputs, greedy=False):
        """inputs: ParserInputs, greedy: bool
        -> ParserSample(programs: list[tuple], role_traces: list[tuple],
                        log_prob: Tensor[B], entropy: Tensor[B])
        log_prob sums masked trace log-probabilities; entropy sums visited local entropies.
        """
        logits = self(inputs)
        programs, role_traces, log_probs, entropies = [], [], [], []
        for b in range(inputs.context_ids.shape[0]):
            moves, traces = [], []
            log_prob = logits[b].sum() * 0
            entropy = logits[b].sum() * 0
            for s in range(inputs.context_ids.shape[1]):
                if not inputs.step_mask[b, s]:
                    continue
                keywords = inputs.step_keywords(b, s)
                scores = logits[b, s, inputs.keyword_mask[b, s]]
                move, trace, step_logp, step_entropy = self._decode_roles(
                    keywords, scores, greedy, b, s)
                moves.append(move)
                traces.append(trace)
                log_prob = log_prob + step_logp
                entropy = entropy + step_entropy
            programs.append(tuple(moves))
            role_traces.append(tuple(traces))
            log_probs.append(log_prob)
            entropies.append(entropy)
        return ParserSample(programs, role_traces, torch.stack(log_probs),
                            torch.stack(entropies))
