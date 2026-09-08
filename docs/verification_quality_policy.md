# Verification Quality Policy

The extension should not become a pile of fake tests. The roadmap requires
verification blocks, but those blocks need a clear scope.

## Test Categories

Use shared `tests/test_*.py` files for reusable behavior in `arena_ext`:

- exact math identities
- tensor shape and dtype contracts
- deterministic toy ground truth
- parity against a reference implementation
- negative controls that should fail

Use notebook-local `tests.py` files only for learner-facing exercise checks.
These files should verify that the exercise solution returns the expected
fields and passes the section's minimal deterministic contract. They are not a
substitute for shared library tests or real-model validation.

Hard exercises have an additional decomposition requirement. Difficulty 3/5+
exercise blocks should use the step-by-step ladder in
[Hard Exercise Verification Ladders](hard_exercise_verification_ladders.md):
visible sub-function tests, toy oracle/reference checks, property tests, pinned
expected outputs, explicit tolerances, and debug caches for complex functions.
Run `uv run python scripts/audit_hard_exercise_ladders.py` before release; it
checks the generated hard-exercise registry against visible tests, fixture
provenance, and CUDA-backed report evidence.
Run `uv run python scripts/audit_report_evidence_contracts.py` alongside it;
that gate rejects accepted reports with placeholder evidence, empty baselines,
empty negative controls, empty OOD declarations, lingering known failures, or
missing safety notes.
Run `uv run python scripts/audit_course_surface.py` as a learner-surface gate:
it checks that extension chapters have Streamlit home pages, theme config,
standalone requirements, config-map consistency, instruction pages, and
exercise directories.
Run `uv run python scripts/audit_extension_artifact_hygiene.py` as a repo
hygiene gate: it rejects committed raw weights/checkpoints, oversized extension
artifacts, unredacted sensitive prompt records, ignored generated prompt
fixtures used by tests, and missing cache/data ignore rules.

Keep the roadmap's `run_smoke_test` hook for compatibility, but treat it as a
notebook-contract hook. Verification reports should record it under
`notebook_contract` and include `evidence_level` and `claim_scope`.

## Synthetic Controls

Synthetic data is acceptable when it is the point of the exercise:

- `GT-0` exact games, known planted features, known circuits, or known bogus
  interpretability results
- deterministic labels and seeds
- explicit negative controls
- assertions tied to known ground truth

Do not present synthetic controls as evidence that a full real-model mechanism
has been reproduced. A report can make that stronger claim only after it names
exact model revisions, datasets, seeds, baselines, negative controls, measured
GPU artifacts, and known failures.

## Evidence Levels

Use `evidence_level: notebook_contract` for generated starter reports.

Upgrade the evidence level only when the section has the corresponding proof:

- `implementation_contract`: reusable implementation tests pass against exact
  or reference behavior.
- `real_model_local`: a named local model/dataset revision ran successfully
  under the GPU budget.
- `research_claim`: the notebook includes baselines, held-out checks, negative
  controls, causal interventions where applicable, and failure cases.

The default claim scope is deliberately narrow: the notebook contract ran. That
is useful for course hygiene, but it is not a research result.
