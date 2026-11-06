# 9.3 — execute a supplied completed CPU notebook

The exercise notebook remains the lesson. Its ten methods, immediate tests, toy heatmap, causal
patch controls, historical Pythia adjudication and editable patch play remain inline. This tool
records supplied-source execution; it does not implement the methods, infer student authorship or
certify mastery. The existing CUDA API retains its separate strict native behavior.

Complete the exercises in a private notebook, or explicitly prepare exposed reference practice from
the notebook's own public Solution blocks. Different algorithm bodies and reviewed mathematical
imports are supported. Actual submitted answers are bound rather than required to match published
answers. Execution accepts an already completed copy, never fills stubs and never overwrites input.

## Source-only commands

Use the repository's existing environment:

```bash
uv run --no-sync --offline python chapter9_alignment_interpretability/exercises/part3_cot_faithfulness/execute_learner.py --check
```

CHECK is the default. Import and CHECK use standard-library AST, JSON, hashing and compilation
without execution. They do not import Torch, Matplotlib, the generated/native module, nbformat or
nbclient; run tests or lesson methods; load a model; or start a kernel.

```bash
uv run --no-sync --offline python chapter9_alignment_interpretability/exercises/part3_cot_faithfulness/execute_learner.py --prepare-reference --output-dir /tmp/cot-reference-practice
```

This explicit operation changes only the ten declared definition spans in a new practice copy.
Nonanswer statements/cells, helpers, immediate tests, order, metadata and saved output objects stay
exact. Its receipt says exposed public-reference practice, `executed:false` and
`authorship:not_inferred`. Existing output directories are refused. Relative output directories
resolve against caller cwd before any file or runtime path is created. Saved figures remain
historical; preparation produces no numerical result.

```bash
uv run --no-sync --offline python chapter9_alignment_interpretability/exercises/part3_cot_faithfulness/execute_learner.py --validate-completed /absolute/path/to/my_completed_copy.ipynb
```

Validation binds actual submitted source/signatures without rewriting input. Fixed tests, interfaces,
defaults, order, cohorts and dependencies remain bound. Private runtime outputs/counters,
execution/trust metadata, newly assigned cell IDs and Python version metadata may differ. Missing or
ambiguous methods, remaining stubs, changed fixed code/tests/defaults, unreviewed imports, explicit
historical/native delegation and namespace manipulation fail. Nested helpers, local mathematical
aliases and different reasonable algorithms are supported. Unsupported layouts need review.
These structural checks are not a sandbox, complete dataflow analysis, semantic proof or authorship
assessment.

## Explicit future execution boundary

`--execute NOTEBOOK --output-dir NEW_DIR [--reference-practice]` validates and snapshots an already
completed copy before the only lazy `NotebookClient.execute` call. No default/source operation
enters this branch. Source engineering has not run a real kernel; actual CPU execution needs
separate authority.

Its scope is `canonical_9_3_cpu_toy_and_historical_adjudication`. All 15 unchanged code cells run.
`RUN_GPU_SIGNATURE=False` and the fixed native branch remain exact. There is no native/publish
switch. The existing `run_full_experiment -> run_gpu_test` wrappers still call the historical part2
module's own globals. This tool does not claim supplied-method CUDA closure or fresh Pythia results.

The new run uses an absolute isolated cwd and exact source files at original relative paths.
Canonical and historical root discovery both find the copied chapter9 directory. Fully qualified
part2 tests/utils/solutions remain available through their original namespace-package topology.
Some old tests use independent historical toy fixtures while invoking supplied student callables;
those tests are preserved. The historical 9.2 notebook is an explicit read dependency of its optional
shape helper; the whole historical test suite is not run automatically.

The fixed CPU figure writes to `chapter9_alignment_interpretability/instructions/assets`.
The tool creates this directory empty. Historical reports/provenance inputs get separate custody
copies outside the runtime workspace; no historical PNG/report is copied as a fresh figure. Known
root/import paths have AST/path/spec checks only. Third-party/kernel startup and actual runtime
import compatibility remain open. Isolation protects fixed output paths, not arbitrary Python.

## Results and truthful receipts

The existing signature is the three-panel toy plot: held-out hidden-answer localization, donor-margin
patch effects and donor flip rates. All shuffled-label, random-donor, irrelevant-position, text-only
and other controls, cohorts, train/evaluation rows, layers, thresholds and play defaults stay exact.

All ten supplied methods have roles across the notebook. `answer_logit_diff` and
`replace_position_in_layer_output` have immediate tests and do not feed the CPU signature.
`adjudicate_pythia_result` evaluates the locked historical negative dictionary with its original
assertions. That is new adjudication of recorded measurements, not fresh Pythia inference. Existing
toy `accepted`, `tests_passed` and `contract_passed` flags retain their toy-contract meaning.

Only after all nonempty cells finish without error outputs, source still validates and a newly
created PNG exists at the fixed destination may receipts say `execution_completed:true` and
`completed_from_supplied_answers:true`. The new figure is hashed separately. Canonical saved outputs
are labelled historical; supplied prior outputs are not verified fresh. Receipts retain
`authorship:not_inferred`, `fresh_pythia_measurements:false` and `scientific_acceptance:false`.
Failures preserve input, FAILED receipts and available partial notebooks; reruns require a new dir.

The contract pins 13 lesson/dependency/provenance inputs. Two shared audit/test sources are custody
preimages, not runtime imports. The direct course audit also requires a native/publication relation
and legacy claim markers. This tool does not change that detector or claim it becomes green.
No fresh scientific result, unseen assessment, student authorship or course readiness is inferred.
