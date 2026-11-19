# Learned Parser Development

Commit `3a62055` introduces a grammar-constrained learned parser and refactors mini to support multiple parsing, grounding, and training configurations.

## 1. Learned Parser Modules

| File | Responsibility |
|---|---|
| `learned_parser.py` | `lex_step` extracts keywords and context; `Grammar` constrains role assignments to valid DSL programs; `role_to_program` constructs a `Move` AST. |
| `batch.py` | Adds `ParserInputs` and `LearnedBatch`, separating parser inputs from final-scene targets. |
| `learned_lr.py` | `RolePolicy` uses Embedding, BiGRU, and Linear layers to predict seven keyword roles; Grammar masks invalid roles before sampling or greedy decoding. |
| `train.py` | Adds `train_parser()` with scene loss, REINFORCE policy loss, and entropy regularization. Task loss trains scene modules; policy gradients train the parser. |
| `learned_metrics.py` | Reports scene and exact-program metrics (including IoU-M), split into overall, simple, and complex results. |

### Training flow

```text
Instruction + Scene
  → batch.py / ParserInputs
  → BiGRU RolePolicy
  → Grammar-constrained roles
  → role_to_program / Move AST
  → Executor + Action Simulator
  → Predicted final scene
  → Scene loss + REINFORCE
  → Parameter updates
```

The original rule-based `train()` remains available; dedicated learned-parser experiments call `train_parser()`.

## 2. Shared Execution and Model Abstraction

### `execution.py`: Shared execution

**`apply_move(sim, boxes, subj_w, ref_w, action_emb, *, move_mode="soft")`**

- `sim`: Action Simulator; `boxes`: current object locations; `subj_w` and `ref_w`: object-selection weights.
- `move_mode`: `"soft"` uses weighted positions; `"hard"` uses argmax with straight-through gradients.
- Returns the updated object boxes.

**`execute_steps(scene, program, sim, ground_step, *, move_mode)`**

- `scene`: initial scene; `program`: batch of Move sequences; `ground_step`: backend-specific grounding callback.
- Executes steps in order, updating object locations, and returns `ExecutionResult(final_boxes, grounded_program)`.

`NSModel` and `ParsedNeuralBaseline` each implement `_ground_step()` and `execute()`. They share the execution loop but preserve different object-grounding backends.

### `curriculum.py`: Model selection and training

**`build_model(*, parser_source, backend)`**

| Parameter | Options | Effect |
|---|---|---|
| `parser_source` | `"rule"` / `"learned"` | No attached parser / `RolePolicy` |
| `backend` | `"symbolic"` / `"neural"` | `NSModel` / `ParsedNeuralBaseline` |

Returns the selected backend with an optional learned parser and an Action Simulator.

**`train_group(group, seed, config)`**

- `group`: experiment specification in `config["groups"]`; `seed`: initialization and data generation; `config`: loss, schedule, mode, and other hyperparameters.
- Trains the selected model through its phases and saves checkpoints.
- **Schedule A:** 20 epochs of joint simple-instruction training, then 40 joint mixed-instruction epochs.
- **Schedule B:** 20 joint warm-up epochs, 30 Action Simulator-only epochs, then 20 relation-grounding epochs (also training the learned parser, if present).

**`_draw(parser, scene, keywords, patterns, count, greedy)`** samples `count` grammar-valid programs per instruction; `greedy=True` selects highest-scoring valid roles. Returns programs, log probabilities, and entropy.

**`_fit(..., phase, mode)`** trains one phase. `phase` selects the trainable modules and instruction pool; `mode` chooses soft/hard execution. Learned-parser phases use REINFORCE; rule-parser phases use fixed programs.

## 3. Changes to Existing Mini Modules

The commit refactors existing mini modules while retaining the legacy path.

| File | Changes |
|---|---|
| `batch.py` | Extracts `_scene_tensors()` and splits the original collation into `_collate_rule()` and `_collate_learned()` behind the backward-compatible `collate(mode="rule")` wrapper. |
| `baseline.py` | Extracts `_ground_roles()` from `ParsedNeuralBaseline.forward()`. Adds `_phrase_tokens()` and `_move_tokens()` for AST-to-token conversion, `_ground_step()` and `execute()` for shared execution, and optional `sim` injection. |
| `model.py` | Moves `apply_move()` to `execution.py`; adds `_ground_step()`, `execute()`, and optional `sim` injection to `NSModel`; introduces `ParseNSModel` and `LearnedOutput`. |
| `data.py` | Adds shared `keyword_kind()` for colour, noun, and relation classification. |
| `train.py` | Adds `scene_loss()`, `policy_objective()`, and `train_parser()` without replacing `train()`. |
| `run_experiments.py` | Adds CLI routes for learned-parser experiments, ten-group curriculum comparisons, and checkpoint evaluation while retaining its original default path. |

`parser.py` and `executor.py` are unchanged.

## 4. Two Experiment Paths

**Dedicated learned-parser experiments:**

```text
collate(mode="learned") → ParseNSModel → train_parser() → evaluate_learned()
```

`ParseNSModel` batches K sampled programs across K × B execution paths, reporting scene and exact-program accuracy.

**Configurable curriculum experiments:**

```text
run_experiments.py --experiment → train_group() → build_model(parser_source, backend)
→ model.execute() → execute_steps() → experiment_eval.py
```

This path compares ten groups across parser source, backend, loss, schedule, and move mode. Its `_fit()` reuses `scene_loss()` and `policy_objective()` rather than calling `train_parser()`.

The abstraction preserves rule parsing and existing grounding backends while allowing independent parser/backend selection. `ParseNSModel` retains its own K-sample execution method, separate from the shared `execute_steps()` path.
