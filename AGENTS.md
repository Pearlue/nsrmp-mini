# nsrmp-mini — contribution guidance

## Start here

- Upstream: `kwesi07/nsrmp-mini`; contribution target: upstream Issue #1.
- Read [the planning index](docs/issue-1/README.md), then the design, implementation plan and validation checklist.
- This branch currently contains a **development plan, not a learned-parser implementation**. See [STATUS.md](docs/issue-1/STATUS.md) for verified work and pending work.
- Preserve the upstream miniature 2D experiment. Do not expand this issue into a detector, robot simulator, learned sentence splitter or full-paper reproduction.

## Repository and collaboration

- Work on a feature/documentation branch; push to the contributor fork, not upstream `main`.
- Before editing, inspect branch, status and any concurrent changes. Never overwrite unrelated work.
- Do not use `git stash`, `git reset --hard`, or `git checkout --` to dispose of another person's edits; use a WIP commit or isolate the work instead.
- Stage explicit paths. Do not commit `.env`, credentials, virtual environments, caches, or model checkpoints.
- Preserve existing tracked experimental results. New experiments must write to a distinct output directory.
- Commit/push/open a PR only within the current user's requested scope. Publishing a documentation branch is not permission to claim the issue is implemented or to close it.

## Development contract

- Python 3.12 through `uv`; install the versions in `requirements.txt`; use pytest.
- Start implementation with tests in `tests/test_learned_parser.py`; keep all existing tests and default APIs working.
- Learned-parser training may use instruction text, provided lexical vocabulary/splitter, initial scene features/boxes and final-scene targets. It must not use gold programs, rule-parsed roles, gold subject/reference identities, or gold intermediate states as learning signals.
- Gold programs are allowed only in offline diagnostics, deterministic builder fixtures, and evaluation. Keep those paths separate from the training data path.
- `Example.n_steps` currently reads `Example.program`; the new training path must derive step counts from `split_steps(text)` instead.
- Do not hide parser errors by falling back to the rule-based parser. Do not use scene targets or gold structure to select/mask candidates.
- Keep baseline code paths, RNG behavior and published numbers unchanged. Add the learned model through an explicit opt-in experiment flag.
- Report program accuracy and scene metrics separately. A low scene loss is not proof of a semantically correct parser.
- Distinguish proposed interfaces/commands from commands that already exist. Do not report planned tests or experiments as completed.

## Documentation

- Update the planning/status documents when implementation decisions change.
- The lexer ambiguity correction and leave-one-out baseline are explicit refinements of the issue suggestion; do not silently attribute them to the paper or to the maintainer.
- Label experimental reporting `Our results`; do not claim that the original paper's results were reproduced.
- If training fails after the documented attempts, retain the failure logs and accuracy curve rather than inventing results or changing the acceptance claim.
