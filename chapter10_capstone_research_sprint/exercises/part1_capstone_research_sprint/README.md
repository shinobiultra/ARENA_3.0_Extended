# [10.1] Capstone Research Sprint

Start with `10.1_Capstone_Research_Sprint_exercises.ipynb`. The notebook first
builds the planning scaffold for a paper-style capstone, then inspects a
committed CUDA run of a mini activation-oracle sprint.

The committed experiment is intentionally scoped. It trains a question-conditioned
MLP oracle on generated latent-state activations, compares it against text-only
and linear-probe baselines, and validates the result with held-out templates,
ablation, counterfactual patching, random-patch, random-activation, and
label-shuffle controls. This is a model-organism capstone, not a released-model
mechanistic discovery.

Key artifacts:

- `scripts/run_capstone.py` runs the live experiment and regenerates the result
  files.
- `results/metrics.json` contains aggregate three-seed metrics.
- `results/metrics_by_seed.json` contains per-seed metrics.
- `results/failure_cases.jsonl` records held-out-template failures, if any.
- `reports/capstone.md` is the generated writeup.
- `artifacts.lock.yml` pins the claim scope, baselines, controls, thresholds,
  seeds, and VRAM budget.
- `verification_report.json` records the CUDA verification report used by the
  release gate.

Use `research_projects/00_project_template/` when turning this model-organism
capstone into a new project on a real model. The same standard applies there:
declare the claim narrowly, pin artifacts, run baselines and controls, and keep
the writeup tied to reproducible scripts.
