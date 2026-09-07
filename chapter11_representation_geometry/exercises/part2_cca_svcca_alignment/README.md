# [11.2] CCA, SVCCA, and Representation Alignment

Open `11.2_CCA_SVCCA_and_Representation_Alignment_exercises.ipynb`. Implement the
functions before opening their solution dropdowns. The paired solution notebook
contains the same lesson and actual executed outputs. The instruction page has
all nine figures beside the code that produces them.

The lesson derives whitening and CCA, compares SVCCA with Procrustes and ridge,
then asks whether a fitted cross-layer map can reproduce an actual GPT-2 donor
intervention. Five implementation groups expose eight core functions, and one
interpretation exercise separates correlation, donor reproduction and name
correctness. Three play cells vary sample dimension, causal-feature variance,
and an actual GPT-2 intervention rank. Nine direct mathematical checks include
independent oracles and counterexamples.

The real CPU experiment uses pinned GPT-2, 72 training / 48 development / 96
final prompts, three layer pairs, all 17 rank/method candidates and 16 matched
random draws per pair. It records 12,816 outcomes, including failed donor and
transported answers. The main model work plus the play uses 12,893 forward rows.
A fresh model/fit check compares 5,184 condition outcomes across the first and
last 24 final prompts, including every method and random draw. It runs on CPU;
no CUDA experiment is claimed for this section.

```bash
uv run python -m pytest chapter11_representation_geometry/exercises/part2_cca_svcca_alignment/alignment_methods_tests.py -q
uv run python chapter11_representation_geometry/exercises/part2_cca_svcca_alignment/execute_learner.py --scope full --output-directory "$HOME/.cache/arena_extended/alignment/my-own-run"
uv run python chapter11_representation_geometry/exercises/part2_cca_svcca_alignment/alignment_reproduce.py --learner-directory "$HOME/.cache/arena_extended/alignment/my-own-run" --output-directory "$HOME/.cache/arena_extended/alignment/my-fresh-check"
```

Use a new output directory for each run. `--scope foundations` executes just the
mathematical part. `lesson.md` is the readable canonical source; the builder
completes only each exercise's own answer. Large actual tensors stay in the
named persistent run directories, with their hashes and small records archived
under `expected_outputs`. The first model run needs network access unless the
pinned checkpoint is already cached.

This is a bounded cross-layer alignment study, not a complete IOI circuit
reproduction or a claim that correlated directions are causally interchangeable.
Pythia development's generic-token failure is retained as a negative comparison.
