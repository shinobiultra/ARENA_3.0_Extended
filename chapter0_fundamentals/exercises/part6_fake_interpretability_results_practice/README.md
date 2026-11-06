# [0.6] How to Know When an Interpretability Result Is Fake Verification Assets

Generated support files for the roadmap verification contract.

- `artifacts.lock.yml` pins the current diagnostic contract and CUDA diagnostic
  preflight.
- `0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_exercises.ipynb`
  is the local learner notebook with stubs and visible tests.
- `0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_solutions.ipynb`
  runs the reference implementation, visible tests, and committed CUDA report
  checks.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the historical `run_smoke_test` contract hook.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

This section is a GT-0 diagnostic lab: the bogus results are synthetic and
known in advance, so every detector should flag its assigned failure mode.
It is not evidence that a full real-model mechanism has been validated.
The graded GPU path records label-leakage accuracy, cherry-pick inflation,
probe train-vs-heldout gap, and random-direction control rejection on CUDA.


## Active Practice pair and intact references

Open [0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_Practice_exercises.ipynb](0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_Practice_exercises.ipynb) for the active Practice copy,
and [0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_Practice_solutions.ipynb](0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_Practice_solutions.ipynb) for historical solutions.
The exercise copy clears code outputs and execution counts. The duplicated foundation historical
appendix is read-only source, while all active teaching stubs, immediate tests and reveals stay inline.
Section 16.3 keeps all original teaching code cells executable and blank in the exercise copy.
This is exposed public guided practice, not an unseen assessment or a fresh scientific run.

The complete prior combined reference pair remains alongside under `0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_*.ipynb`,
with its original historical code outputs and counts. Every direct original source book is linked
in the lesson introduction; original directories, modules, tests, utilities and arena_ext remain intact.
Copied verification reports retain their historical locators and hashes and are not current execution
or acceptance receipts for this Practice directory. No scientific report was recomputed or rehashed.

For explicit module use, import `chapter0_fundamentals.exercises.part6_fake_interpretability_results_practice.solutions` and call `initialize_runtime()` before using runtime callbacks.
The module is inert on import. Original books use their unchanged original helper modules and setup,
not this new module API. The original helper paths are statically preserved; no replay was executed.
