# [16.3] Shapley Interactions with shapiq

This section asks whether second-order Shapley interaction indices can recover
a planted feature pair that individual Shapley-value rankings miss while
correctly exposing contributions from a known three-way interaction.

The learner notebooks begin from all 16 coalitions of a four-feature polynomial
game containing both pairwise and three-way dividends. Students implement
coalition enumeration, discrete second differences,
exact pairwise SII, exact individual Shapley values, merged-player permutation
sampling, a within-size value permutation control, `shapiq` parity, and recovery
metrics. The signature result is generated from those implementations rather
than loaded from `verification_report.json`.

Files:

- `16.3_Shapley_Interactions_with_shapiq_exercises.ipynb`: seven exercises,
  immediate semantic tests, controls, play cell, and anomaly hunt.
- `16.3_Shapley_Interactions_with_shapiq_solutions.ipynb`: inline solved code
  with executed CPU outputs.
- `tests.py`: exact toy oracles and control-sensitive tests.
- `solutions.py`: section-local references plus the existing serialized CUDA
  neural-game path.
- `verification_report.json`: supporting release evidence only.

The preserved CUDA path trains the finite four-feature MLP and compares its
ablation table with the same exact SII and pinned `shapiq` conventions. It is
not required for the CPU lesson.
