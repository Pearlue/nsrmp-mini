# nsrmp-mini — Spec

A small, from-scratch re-implementation of the core ideas of
*Learning Neuro-symbolic Programs for Language Guided Robot Manipulation* (Namasivayam K. et al., ICRA 2023, arXiv:2211.06652),
in a simplified **2D top-down blocks world** with no images and no robot.
It is **inspired by** the paper. It is not a reproduction of their numbers, and no code is copied from the official repo.

## Goal (what the experiments must show)
1. **Table II analogue:** our neuro-symbolic model vs a purely neural baseline (NMN+-like) on IOU, IOU-M and subject-grounding accuracy, split into *simple* vs *complex* instructions.
2. **Fig. 4a analogue:** train on scenes with 3–5 objects, test on scenes with 3…10 objects, and plot IOU-M vs object count for both models.
3. **Fig. 4b analogue (phase 2):** train on 1–2 step instructions, test on 1…5 steps, and plot IOU-M vs number of steps.
4. Every number is reported as **mean ± std over 3 seeds**. This addresses a weakness we criticise in the paper.

## World (`data.py`)
- Each object has:
  - a colour from `{red, green, blue, yellow, cyan, white}`;
  - a shape from `{cube, dice, lego}`;
  - an axis-aligned box `(x1, y1, x2, y2)` in [0, 1]², side ~0.08–0.12, with no overlaps at generation.
- **"Visual features"** stand in for the paper's YOLO+ResNet features. They are computed as `f = P · [onehot(colour); onehot(shape)] + ε`:
  - `P` is a fixed random 32×9 matrix, seeded once and shared by train and test.
  - `ε ~ N(0, σ²)` with σ = 0.1.
  - Boxes are given separately (the paper's "gold BB" setting).
  - Point this out in the README: we have no detector, so we only test the gold-box setting.
- **Actions** (placed adjacent to the reference object with a fixed gap, clipped to [0, 1]): `MoveLeft`, `MoveRight`, `MoveFront` (+y), `MoveBehind` (−y).
- **Relations** for complex references: `left`, `right`, `front`, `behind`, defined by comparing box centres.
- **Instructions**, generated from templates:
  - Simple: "put the red cube to the right of the blue dice".
  - Complex: the subject (or the reference) is described with a relation, e.g. "put the lego which is behind the green dice to the left of the red cube".
  - The generator must only emit instructions whose referring expressions pick out **exactly one** object.
- **Gold programs:** each example also stores its gold program, as nested tuples using the paper's DSL operators (`Scene`, `Filter`, `Relate`, `Unique`, `Move`).
- **Multi-step (phase 2):** "…, then put …". The steps are applied sequentially and the gold final state is after the last step.
- **Ground-truth final state:** all boxes after executing the steps. Only the subject moves.

## Neuro-symbolic model (`model.py`, `executor.py`)
Follows the paper's Visual Reasoner and Action Simulator, plus NS-CL-style soft execution.
- **Concept embeddings:** a learned vector per colour and per shape. Object features are projected by a learned linear map, then:
  `Filter(set p, c) = p ⊙ σ((cos(Wf_i, e_c) − γ)/τ)`.
- **Relation concepts:**
  - A small MLP on the pair `(box_i, box_j)` gives a logit per relation.
  - `Relate(obj weights w, r)_i = Σ_j w_j σ(logit_r(i, j))`, with self-pairs masked.
- **Unique:** normalise the set into weights over objects, `w = p / Σp`. A softmax over `log p` is also fine.
- **Action Simulator:** an MLP that takes `(action embedding, subject box, reference box)` and outputs the new box.
  - Subject box and reference box are soft: `Σ_i w_i · box_i`.
- **Predicted final world:** `box̂_i = (1 − w_subj_i) · box_i + w_subj_i · AS(…)`.
- **Program for training/testing:** comes from a rule-based parser over the known keywords. This is the paper's "lexer is given" assumption, and it corresponds to the official repo's ablation that trains with given programs.
  - The parser is deterministic and is tested.
- **Loss:** the paper's loss, `Σ_i ||box̂_i − box_i||₁ + β(1 − IoU(box̂_i, box_i))` with β = 1.
- **Training:** Adam and mini-batches.
  - Curriculum: simple instructions first, then all instructions, as in the paper. A flag must switch it off for an ablation.
- **Multi-step (phase 2):** run the steps sequentially on the predicted world, with the loss on the final state only.

## Neural baseline (`baseline.py`), NMN+-like
- Same object features, same Action Simulator architecture and same loss.
- Instruction encoder: token embeddings and a GRU/LSTM over the whole sentence, with **no program**.
- Two attention heads score the objects, each `MLP([obj_feat; box; sentence_vec])`, followed by a softmax. One gives the subject weights, the other the reference weights.
- An action head classifies the sentence vector into the 4 actions. Its soft action embedding feeds the Action Simulator.
- Multi-step for the baseline: split the sentence on "then", using the same given splitter for both models.

## Metrics (`metrics.py`)
- **IOU:** mean over all objects. **IOU-M:** mean over moved objects only.
- **Subject accuracy:** `argmax w_subj == gold subject`.
- **Placement accuracy:** the predicted subject box centre lies inside the gold final box.

## Experiments (`run_experiments.py`)
- **Data:** 4000 train / 1000 test, 3–5 objects, 50/50 simple vs complex. Generalisation test sets have 3…10 objects (phase 1) and 1…5 steps (phase 2).
- **Runs:** both models × 3 seeds.
- **Outputs:** written to `results/` as JSON plus PNG plots (matplotlib), plus a markdown results table.
- **Runtime:** a full run must finish in under ~20 minutes on an Apple M2 CPU. Shrink epochs or data if needed, and report what was used.

## Engineering
- Python 3.12 venv via `uv` inside this folder. Dependencies: torch, numpy, matplotlib, pytest. Pin versions in `requirements.txt`.
- **Tests (pytest), written test-first:**
  - The generator always produces unique referents and a correct gold final state.
  - The parser returns the gold program for every template.
  - With hand-set "perfect" concept embeddings, executor `Filter`/`Relate`/`Unique` pick the right object.
  - Applying the AS to a gold subject reproduces the loss formula.
  - IoU edge cases.
  - An end-to-end smoke test: a tiny training run lowers the loss.
- **README.md:**
  - Paper citation and link.
  - What this project is and is not.
  - The mapping from paper module to our file.
  - Simplifications, and why each one was made.
  - Setup and run commands.
  - Results tables and plots, clearly labelled **"our results"**, next to the **paper's numbers**, labelled "from the paper (Table II / Fig. 4)".
  - Discussion of differences.
  - An "Our contributions vs reused" section: everything is written by us; ideas come from the NSRMP and NS-CL papers.
- `git init` locally. Do **not** create a GitHub repo or push: the group decides that.
