# Agent instructions

Any coding agent (Codex, Claude Code, others) working in this repository follows the same rules.

1. Read `CLAUDE.md` first: it holds the project rules (English names and code, Vietnamese prose, no invented numbers, anti-leakage, never assign names in `task-assignment.xlsx`).
2. Then read `README.md`, `docs/interfaces.md` (file formats and APIs shared between work packages) and `plan.md`.
3. Paths come from `configs/default.yaml` through `src/utils/config.py`; do not hard-code them.
4. Every number in a report goes through `src/eval/metrics.py`. Every run keeps its script (`scripts/`), its result file (`experiments/`) and a short report; never delete run artifacts.
5. Design decisions use train and validation only; the test split is touched once, for the final numbers.
6. Code changes go through a branch and a pull request. Conventional commit messages, no AI attribution.
7. Large data is not committed; see `data/README.md` and `data/MANIFEST.md`.
