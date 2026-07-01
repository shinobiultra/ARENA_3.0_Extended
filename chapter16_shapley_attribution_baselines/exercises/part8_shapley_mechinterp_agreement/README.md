# [16.8] Do SHAPley and Mechanistic Interpretability Agree? Verification Assets

Generated support files for the roadmap verification contract and the real CUDA
mechanistic-agreement preflight.

- `artifacts.lock.yml` pins the current smoke-test artifact contract plus a
  CUDA-trained nonlinear feature game.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.
- `artifacts/agreement_matrix.csv` records agreement/disagreement metrics using
  the roadmap schema.
- `artifacts/deletion_curves.png`, `artifacts/insertion_curves.png`, and
  `artifacts/topk_overlap_heatmap.png` visualize consequence tests from the
  same trained model run.
- `artifacts/method_disagreement_examples.md` records the tested additive, XOR,
  and shuffled-control disagreement cases.

The graded GPU path in `solutions.py` trains the finite neural coalition game,
compares Shapley values against known mechanistic scores, checks deletion and
pair-interaction consequences, and rejects a shuffled-label agreement control.

Notebook entrypoints:

- `16.8_Do_SHAPley_and_Mechanistic_Interpretability_Agree_exercises.ipynb`
- `16.8_Do_SHAPley_and_Mechanistic_Interpretability_Agree_solutions.ipynb`
