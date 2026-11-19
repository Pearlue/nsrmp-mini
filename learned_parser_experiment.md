# Reproduce the learned-parser experiments

Run the commands from the repository root. The test suite can be run independently.

## Set up the environment

```sh
uv venv --python 3.12.14 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Run the tests

The suite includes 59 upstream regression cases and 103 parser cases covering program construction, masking, policy gradients and final-scene supervision.

```sh
# Upstream regressions, parser tests, and full suite
.venv/bin/python -m pytest -q --ignore=tests/test_learned_parser.py  # Expected: 59 passed
.venv/bin/python -m pytest -q tests/test_learned_parser.py  # Expected: 103 passed
.venv/bin/python -m pytest -q  # Expected: 162 passed
```

## Training datasets

The runner generates the training scenes from the configuration and seed at launch. All ten groups share the same training pool for a given seed. Each training scene contains 3–5 objects and one instruction step.

- [Baseline](experiments.json) and [Round 1](results/ten-group-followup/round1.json). For each seed from 0 through 9, generate 2,000 simple and 2,000 complex scenes.
- [Round 2](results/ten-group-followup/round2.json) and [Round 3](results/ten-group-followup/round3.json). Keep the base pool and generate 4,000 additional complex scenes with `20000 + seed`. A mixed epoch uses all 2,000 simple scenes and samples 2,000 of the 6,000 complex scenes without replacement.

[Training pool generation](nsrmp/curriculum.py#L78) · [Scene generator](nsrmp/data.py#L226) · [Per-epoch sampling](nsrmp/curriculum.py#L86)

## Reproduce the third ten-group matrix

Start each run in a fresh directory. Run the small-data pilot first, verify the checkpoints, then launch the formal matrix.

```sh
out=$(mktemp -d)
cfg=results/ten-group-followup
py=.venv/bin/python
"$py" run_experiments.py --experiment 1 2 3 4 5 6 7 8 9 10 --experiment-config "$cfg/pilot3.json" --out "$out/pilot" &&
cp "$cfg/pilot3/verify.py" "$out/pilot/verify.py" &&
PYTHONPATH=. "$py" "$out/pilot/verify.py" &&
"$py" run_experiments.py --experiment 1 2 3 4 5 6 7 8 9 10 --experiment-config "$cfg/round3.json" --out "$out/formal"
```

Each of the ten groups trains independently on seeds 0–9 with up to ten single-threaded workers. Each seed uses 1,000 test scenes per instruction length. The runner writes moved-object IoU for single- and two-step instructions to `curriculum/group-<g>/seed-<s>/metrics.json`. The aggregate `curriculum/summary.json` contains seed means, sample standard deviations and paired differences. [Conditions and results](learned_parser_result.md)
