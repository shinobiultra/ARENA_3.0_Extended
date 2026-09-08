# [16.7] Data Shapley in One Training Run

This section is a CPU-sized learner lab built around one falsifiable claim: in
an eight-example logistic-regression game, a checkpoint-gradient estimator
accumulated during one 40-step full-data run identifies the planted bad label
and correlates `r=0.995472` with exact Data Shapley from all 256 subset-training
runs.

The lesson opens with an independently inspectable four-example game. Averaging
all 24 orderings gives exact values
`[0.641204, 0.641204, 0.641204, -1.173611]`, verifies efficiency, and shows that
deleting the planted error improves utility by `0.25` before any approximation
is introduced.

The exercise notebook keeps the method visible. Students implement coalition
training and held-out utility, exact Shapley, sampled permutation Shapley,
checkpoint-gradient scores, leave-one-out, damped influence, and label
shuffling. Immediate semantic tests check pinned training utilities, exact
values, efficiency, duplicate symmetry, ordering bias, and deletion behavior.

The solution notebook generates the signature visualization from those
implementations. Its controls include random-order budget curves, opposite
fixed orders, influence and leave-one-out baselines, matched one-row deletions,
and relocation of the bad label to its duplicate twin. No metric is loaded from
`verification_report.json`, and the learner result uses no white-noise control.

Run the focused CPU checks from the repository root:

```bash
.venv/bin/python -m pytest -q \
  chapter16_shapley_attribution_baselines/exercises/part7_data_shapley_in_one_training_run/tests.py
```

Execute the solution notebook in place from this directory:

```bash
../../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=180 \
  16.7_Data_Shapley_in_One_Training_Run_solutions.ipynb
```

The existing CUDA entry point and verification report remain supplementary
infrastructure. They are not imported by either learner notebook and are not
needed for this section's result.
