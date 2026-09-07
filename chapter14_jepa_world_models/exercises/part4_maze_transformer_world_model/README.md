# [14.4] Maze Transformers: Policies, Probes, and State Patching

This section trains a decoder-only transformer from scratch on deterministic
mazes with exact breadth-first-search supervision. Students build the maze and
BFS oracle, serialize causal maze states, implement the transformer and cache,
train with immutable checkpoints, probe distance / local legal degree / action,
and causally patch counterfactual next-action activations.

The result is accepted only when action accuracy, path legality, goal reach,
and exact shortest-path optimality are reported separately. The notebook shows
both raw autoregressive decoding and a legal-action-constrained planner. It does
not hide the raw model's illegal moves behind constrained decoding.

Files:

- `14.4_Maze_Transformers_and_Latent_Planning_exercises.ipynb`: learner stubs,
  visible tests, controls, play cells, and anomaly hunting.
- `14.4_Maze_Transformers_and_Latent_Planning_solutions.ipynb`: completely
  executed CPU reference notebook.
- `solutions.py`: exact BFS oracle, model, training, probes, patching, and strict
  CPU/CUDA experiment contracts.
- `tests.py`: hand-computed fixtures, causality and patching invariants, real
  checkpoint training, and an end-to-end smoke test.
- `artifacts.lock.yml`, `verification_report.schema.json`, and
  `expected_outputs/`: pinned evidence and reporting contract.

The checked CPU policy reaches the goal on 41/60 mazes with legal-action
constraints, versus 16/60 with raw argmax and 12/60 for the wall-blind template.
The stronger internal-search interpretation fails the new controls: Manhattan
distance bins score 0.853 versus a best distance probe of 0.647; same-action donors
from other mazes give median patch effect 0.253 versus 0.189 for targeted donors
on the same 23 eligible pairs. The model contains useful action information;
these results do not establish route-specific state or a search algorithm.

The student implements the core method in 12 exercise groups. Both notebooks and
the instruction page are generated from `lesson.md` and canonical definitions
by `build_notebooks.py`. All dropdown answers are checked against the actual
learner cell, including class methods and immediate tests.

`score_rollouts` independently replays actions; it does not trust success flags.
Detour labels exclude different tie breaks between equally short actions. Local
legal degree counts adjacent open cells, not a BFS search frontier. The full maze
is supplied at every step; this is repeated policy evaluation, not latent rollout.

`run_gpu_test` requires CUDA and never falls back to CPU. CPU and CUDA experiments
use different documented model and dataset sizes, so their results are reported
separately. The candidate remains unreleased pending independent review and the
broader roadmap's maze-topology and planning requirements.

For CUDA, start a fresh process with `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
Training requires that setting, enables deterministic algorithms, and uses
highest float32 matrix precision. The reference generator sets the environment
before importing PyTorch and can verify identical weights and metrics across two
same-process repetitions with `--verify-cuda-repeatability`.
