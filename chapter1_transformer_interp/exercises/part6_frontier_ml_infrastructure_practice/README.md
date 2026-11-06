# [1.6] Local Frontier ML Infrastructure Verification Assets

Generated support files for the roadmap verification contract.

- `artifacts.lock.yml` pins the current runtime artifact contract.
- `1.6_Local_Frontier_ML_Infrastructure_exercises.ipynb` is the local learner
  notebook with stubs and visible tests.
- `1.6_Local_Frontier_ML_Infrastructure_solutions.ipynb` runs the reference
  implementation, visible tests, and committed CUDA report checks.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The graded report path requires the uv environment to expose Python 3.14,
PyTorch `2.12.1+cu132`, torchvision `0.27.1+cu132`, CUDA 13.2, BF16 support,
and a deterministic BF16 CUDA matmul. The Gemma-sized memory estimate is
reported separately from measured peak VRAM.


## Active Practice pair and intact references

Open [1.6_Local_Frontier_ML_Infrastructure_Practice_exercises.ipynb](1.6_Local_Frontier_ML_Infrastructure_Practice_exercises.ipynb) for the active Practice copy,
and [1.6_Local_Frontier_ML_Infrastructure_Practice_solutions.ipynb](1.6_Local_Frontier_ML_Infrastructure_Practice_solutions.ipynb) for historical solutions.
The exercise copy clears code outputs and execution counts. The duplicated foundation historical
appendix is read-only source, while all active teaching stubs, immediate tests and reveals stay inline.
Section 16.3 keeps all original teaching code cells executable and blank in the exercise copy.
This is exposed public guided practice, not an unseen assessment or a fresh scientific run.

The complete prior combined reference pair remains alongside under `1.6_Local_Frontier_ML_Infrastructure_*.ipynb`,
with its original historical code outputs and counts. Every direct original source book is linked
in the lesson introduction; original directories, modules, tests, utilities and arena_ext remain intact.
Copied verification reports retain their historical locators and hashes and are not current execution
or acceptance receipts for this Practice directory. No scientific report was recomputed or rehashed.

For explicit module use, import `chapter1_transformer_interp.exercises.part6_frontier_ml_infrastructure_practice.solutions` and call `initialize_runtime()` before using runtime callbacks.
The module is inert on import. Original books use their unchanged original helper modules and setup,
not this new module API. The original helper paths are statically preserved; no replay was executed.
