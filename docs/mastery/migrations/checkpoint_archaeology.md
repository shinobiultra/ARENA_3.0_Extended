# 17.1 checkpoint archaeology master migration

## Plan before edits

- **Current learner failure:** the published finite-domain lesson bypasses the canonical ARENA master pipeline. Its actual checkpoint study, six exercises,
  visible methods, interpretation and controls must survive migration.
- **Target section claim:** a tiny MLP trained on the complete mod-13 table develops legal Fourier structure; reloaded checkpoint trajectories and keep/ablate
  interventions distinguish that structure from random-label and same-size random-mode controls. This is finite-domain mechanism emergence, not a grokking
  result.
- **Signature result:** retain the exact addition-table heatmap and three-panel trajectory/intervention/control figure, with the metrics table generated from
  the visible study result.
- **Student methods:** finite-domain enumeration, exact logits, legal and conjugate-symmetric random Fourier masks, centered Fourier power, projection/ablation,
  accuracy, checkpoint save/reload, signature assembly and plotting. All 27 historically stubbed functions remain visible, with the original immediate tests or
  acceptance assertions and their own solution dropdowns.
- **Controls/baselines:** exact finite-table oracle, shuffled training labels evaluated against true addition, same-size nonlegal random Fourier modes,
  original/keep-legal/ablate-legal/keep-random logits, early/mid/late checkpoints and first versus stable threshold crossings.
- **Play cells:** preserve seed-1 replication; explain the two unused historical knobs and how to perform explicit mask or threshold investigations. Forecast a
  width change separately without claiming its unrun outcome.
- **Files:** master 17.1, canonical generated notebooks/module/page/media, section-local report plumbing, byte-identical old module copy, this note, archive and
  parity evidence, focused migration tests. Shared routing/config/syllabus/checkpoint/workbook files belong to the lead.
- **Checks:** archive and historical input hashes; all fourteen code cells' AST, nonblank-source and output parity with explicit adaptations; all original
  explanatory lines; own-answer completion; notebook schema; import safety outside the checkout; independent tiny CPU mathematical fixtures. Build without
  `--execute`.

No model is loaded or trained, no checkpoint is downloaded or reloaded, and no GPU or cluster work is performed by this migration. Original ARENA chapters 0–4
and their masters, notebooks, pages and assets remain untouched. No files are cleaned up, staged or committed. No scientific or course-readiness status is
promoted.

## Canonical source and scope

The source is `infrastructure/chapters/chapter17_training_dynamics/master_17_1.py`. Existing shared configuration generates the **Checkpoint Archaeology and
Grokking** filenames.
The current introduction explicitly explains that grokking coverage remains missing. Training and repeated evaluation both use all 169 mod-13 pairs: there is no
held-out split, OOD
arithmetic evaluation, delayed generalization measurement or evidence of generalization after memorization. The Fourier intervention operates on the complete
logit table; it does
not locate a hidden-unit or transformer circuit.

The canonical exercises notebook contains 62 cells (the solutions notebook has 56), fourteen code cells, six exercises and approximately 5,366 words of prose
including the visible
solution dropdowns. All original explanatory lines survive, except the canonical title/link routing, the Learning Objectives heading renamed to avoid converter
ambiguity, standard
emoji difficulty/importance formatting and the signature image's canonical placement. The tests explicitly bound these presentation adaptations. The actual
explanatory content,
exact expectations, helpers, complete study functions, anomaly suggestions, readings and limitations remain. The original statement that a signature result was
“accepted” is
retained historical source/output, not a new learner-acceptance claim.

Six additive forecast/reflection pairs cover the exact table, exact Fourier fraction, exact keep/ablate intervention, small reload fixture, learned checkpoint
study and seed play.
They ask for ranges or rankings, confidence, causal reasoning, observed controls, discrepancies and changed-condition forecasts. The study forecast occurs
before training and
before the reference figure. Answers remain blank. Both reference plots and expected-output dropdowns expose results, so this is guided practice, never an
unseen challenge.

`PLAY_SEED` works in the preserved activity. `PLAY_RANDOM_MASK_SEED` and `PLAY_THRESHOLD` were defined but never consumed by the historical code. The new prose
says so and
specifies explicit `random_fourier_mask`/`fourier_intervention_report` or `mechanism_emergence_report` calls for proposed investigations; it does not silently
rewire the historical
study. Seed changes are replication. A proposed hidden-width change from 96 to 32 at fixed seed, complete domain and 180-step budget changes the scientific
condition; its outcomes
are unmeasured.

## Pre-edit archive and historical evidence

[`before.json`](../evidence/checkpoint_master_migration/before.json) records the SHA-256 of
[`published_before_master.tar.gz`](../evidence/checkpoint_master_migration/published_before_master.tar.gz), all twenty archived files, all original code/prose
cell source hashes,
AST/output hashes and execution counts. It was captured before any canonical source edit. Every member is reopened and checked against its hash. All sixteen
`report_inputs.files`
hashes match the published verification report dated **2026-09-04T23:53:46+00:00**. The original report and all its artifacts remain unchanged.

The old exercises notebook, solutions notebook and matching page are the authority; there is no builder to reconstruct them. The old `solutions.py` is preserved
byte for byte as
`legacy_solutions.py` and in the archive. Its `run_modular_addition_checkpoint_archaeology_preflight` differs from the notebook wrapper. The canonical wrapper
is taken from the
executed notebook. All 27 top-level notebook functions, the nested study runner and two model methods match the canonical module AST, including that wrapper;
this is tested against
the deliberately different old module.

[`output_parity.json`](../evidence/checkpoint_master_migration/output_parity.json) records every original and generated code-cell hash, normalized AST hash,
original output hash and execution count. All fourteen cells have exact scientific AST and nonblank-line parity after two named adaptations:

1. The setup imports the identical dependency closure from section-local `checkpoint_reports.py`, rather than `arena_ext.training_dynamics`. All eight imported
   public functions are used by visible signature or compatibility smoke functions; their four dataclasses and private validator are dependencies. The entire
   9,097-byte original module is therefore the minimal closure, with no unused definitions.
   [`helper_provenance.json`](../evidence/checkpoint_master_migration/helper_provenance.json) records source and copy hashes and every definition's AST hash.
   This is **new plumbing**, not an old historical filename.
2. The plot destination changes from `instructions/assets/checkpoint_archaeology_signature_result.png` to section
   `media/checkpoint_archaeology_signature_result.png`. No figure data or plotting calculation changes. The historical printed destination remains verbatim in
   the historical output; it describes the original run.

The generated Python module additionally uses `__file__` for root discovery, moves the future import ahead of the Python-only `MAIN` assignment and guards
random seeding, tests,
training, file reads and plotting with `MAIN`. These Python-only changes leave notebook code intact. Import safety checks forbid model construction,
`torch.load`, `torch.save`,
plotting, display and seeding; import from an unrelated directory preserves RNG and CUDA initialization state and creates no files.

On that evidence, fourteen historical execution counts and thirteen output-bearing cells are copied verbatim. Both PNGs are decoded from the original notebook
outputs and saved
byte-identically in `media/`; their hashes are in [`figures.json`](../evidence/checkpoint_master_migration/figures.json). A second ordinary build preserves
those outputs using the
canonical same-code check. The exact table and three-panel result were visually inspected: the visible controls, labels and early/mid/late interventions
survive.

**Retained historical reference results are not a fresh execution of the canonical notebook.** The master, notebooks, page and durable parity sidecar say so.
The historical verification report's acceptance fields and runtime/device values are not rerun evidence or a course-readiness promotion. Engineering checks
below do not replace learner acceptance.

## Retained structural exceptions

All existing artifacts remain physically present. Within `part1_checkpoint_archaeology`, retain:

- `17.1_Checkpoint_Archaeology_and_Mechanism_Emergence_exercises.ipynb`
- `17.1_Checkpoint_Archaeology_and_Mechanism_Emergence_solutions.ipynb`
- `README.md`, `artifacts.lock.yml`, `verification_report.json`, `verification_report.schema.json`
- `expected_outputs/README.md`, `expected_outputs/reference_metrics.json`, `expected_outputs/smoke_test.json`
- unchanged `tests.py` and `utils.py`
- `legacy_solutions.py`, the byte-identical historical module copy.

Also retain `instructions/pages/01_[17.1]_Checkpoint_Archaeology_and_Mechanism_Emergence.md` and the three original instruction assets:
`checkpoint_archaeology_signature_result.png`, `checkpoint_archaeology_signature_result.svg`, `checkpoint_archaeology_validation_loop.svg`. The new
`checkpoint_reports.py` is
section-local report plumbing; it should be listed separately from the historical artifacts. The archive and evidence sidecars are durable migration provenance.
AUTHORING's cleanup
suggestion is superseded by this task's explicit preservation instruction.

## Validation and remaining gates

Commands use the existing environment without synchronization or network:

```bash
UV_CACHE_DIR=/tmp/arena-mastery-uv uv run --no-sync --offline python infrastructure/extension/build.py 17.1
UV_CACHE_DIR=/tmp/arena-mastery-uv uv run --no-sync --offline python -m pytest tests/mastery/test_checkpoint_master_migration.py -q
```

The focused suite checks eight migration invariants/fixture groups: physical archive/input preservation; all-cell scientific/output parity and notebook-wrapper
identity; PNG/helper
dependency closure; all six own-answer completions; original prose/forecasts/control/play retention; safe import; tiny mod-5 Fourier/enumeration/projection
invariants; and an
independent transient-versus-stable crossing fixture. No training-oriented legacy test is run. A pure source compile check covers the generated module, report
helper and generated
instruction page.

Initial check attempts found and corrected three assertion bookkeeping issues (timestamp precision, counting nested model methods and acceptance assertions
versus named tests) and
one real converter integration issue: inserting the HTML expected-figure cell caused an automatic solution dropdown while the old manual placeholder survived.
The plotting exercise
now has exactly one usable solution dropdown; the completion test catches regressions. These were migration-check failures, not failed scientific runs.

Remaining gates: independent review, live learner acceptance, any fresh training/checkpoint reproduction and the unimplemented grokking/held-out study. No
missing study is inferred from the route title or historical validation report.
