# Source Layout

Drifting Preference Optimization keeps the open-source surface deliberately shallow.

```text
src/drpo/
  data.py, rewards.py, drift.py, features.py
  methods/               # baseline shims plus maintained SDXL trainers
  methods/sdxl_common.py # shared SDXL training infrastructure
  utils/                 # tensor and selection helpers

src/inference/           # sampling and metric code
baselines/               # compact comparison implementations
scripts/train/           # maintained launch recipes
scripts/inference/       # sampling/evaluation recipes
tests/                   # behavior and layout checks
```

## Core Boundary

Keep reusable training logic in `src/drpo`. Keep shell files as recipes only:
they should set paths, pass hyperparameters, and call a Python entrypoint.
Machine-specific waits, paper table builders, sweep launchers, generated
figures, and copied PDFs are local artifacts and should stay out of git.

## Maintained Entrypoints

SD-Turbo baselines:

```bash
bash scripts/train/draft.sh
bash scripts/train/dpo.sh
bash scripts/train/grpo.sh
bash scripts/train/spo.sh
bash scripts/train/vggflow.sh
```

SDXL-Turbo:

```bash
bash scripts/train/sdxl_turbo_drpo_mae.sh
bash scripts/train/sdxl_turbo_drpo_teacher.sh
bash scripts/train/sdxl_turbo_draft.sh
bash scripts/train/sdxl_turbo_grpo.sh
```

## Third-Party Code

Do not vendor package source when a normal dependency is available. In
particular, OpenCLIP is supplied by the `open-clip-torch` dependency rather than
by a copied `src/open_clip` package.
