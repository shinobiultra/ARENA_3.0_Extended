# [8.5] Sparse Feature Circuits

Course-ready support files for the Sparse Feature Circuits section.

- `8.5_Sparse_Feature_Circuits_exercises.ipynb` teaches the GT-0 learner
  theorem: students implement SAE encode/decode checks, exact node patching,
  exact edge patching, attribution patching, EAP-IG, graph thresholding,
  faithfulness/minimality/completeness, same-size random controls, and a
  generated SHIFT-style edit.
- `8.5_Sparse_Feature_Circuits_solutions.ipynb` executes the filled CPU
  reference path and saves the visible signature-result figures.
- `assets/sparse_feature_circuits_planted_graph.png` is the planted sparse
  feature graph.
- `assets/sparse_feature_circuits_metric_curves.png` shows threshold curves,
  random-control failure, and EAP vs EAP-IG error.
- `tests.py` contains immediate learner tests plus notebook-surface checks.
- `artifacts.lock.yml` pins the broader released-artifact contract.
- `verification_report.schema.json` defines the serialized CUDA/external report.
- `expected_outputs/` records the historical section contract slots.

Scope: the notebook's main scientific claim is the exact GT-0 planted graph.
The committed CUDA report is supporting GT-2 evidence: Pythia-70M-deduped
residual preflight, released SAE state-dict and feature-attribution checks, a
100-example official-code graph artifact, held-out faithfulness on 40
`simple_test` examples, and generated SHIFT-style editing. Those checks remain
serialized to the parent verification pass and are not rerun by this CPU task.
