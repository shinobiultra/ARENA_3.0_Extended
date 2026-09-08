# Publication interface for 16.18 and 16.19

This tooling prepares an exact, reviewable candidate in a new external directory.
Staging needs completed scientific and learner execution, but accepts **pending
independent review**. Installation is a separate operation requiring approval of
the exact candidate manifest. No actual native result or publication approval is
claimed by the tooling's synthetic tests.

The publisher owns copying, receipt verification, packaging and installation. The
section authors own notebook execution, plots, exported Markdown, the native-query
fixture, and its substantive loader/replay checks. It never regenerates those
artifacts, imports a native model, runs a fit, or changes frozen experiment inputs.

## Canonical outputs

| Section | Notebook stem | Instruction page | Figure prefix |
| --- | --- | --- | --- |
| 16.18 | `16.18_Find_the_Interactions_Without_Trying_Every_Subset` | `18_[16.18]_Find_the_Interactions_Without_Trying_Every_Subset.md` | `1618_` |
| 16.19 | `16.19_Explain_the_Proxy_Check_the_Model` | `19_[16.19]_Explain_the_Proxy_Check_the_Model.md` | `1619_` |

Each pair is installed under its existing exercise section, with `_exercises.ipynb`
and `_solutions.ipynb` suffixes. Pages and figures use the chapter's existing
`instructions/pages` and `instructions/assets` directories. The generated Markdown
must already reference the final prefixed figure names. No source `lesson.md`
builder markers are published.

Each section also receives its exact supplied `verification_report.json`,
`verification_report.schema.json`, `artifacts.lock.yml`, and the generated report
tool's `expected_outputs/native_summary.json`. It receives these
unchanged supporting receipts under `expected_outputs`:

- `learner_execution.json`: actual own-answer notebook execution.
- `native_execution.json`: the completed shared native run.
- `fresh_native_execution.json`: the separate cold source-only run.
- `fresh_reproduction.json`: the completed reproduction comparison.
- `native_comparison_summary.json`: bounded comparison summary; all differences
  remain in the physically hashed external full comparison report.

There is one shared `part18_spex/expected_outputs/execution_review.tar.gz`, with a
mandatory `.review-bundle.json` contract. Both sections' content and receipts are
inside it. The archive and its own contract are separately bound installation
artifacts; an archive does not include itself.

## Portable learner data

Both sections use the single canonical file:

```text
chapter16_shapley_attribution_baselines/exercises/part18_spex/expected_outputs/native_query_fixture.json.gz
```

The fixture is plain gzip JSON, bounded at 16 MiB encoded and 64 MiB decoded. Its
adjacent `.data-contract.json` binds exact encoded/decoded hashes and the physical
native execution receipt. It is included in the canonical tree and review bundle.
The section exporter/loader must additionally validate its semantic schema,
prompt/query coverage, actual values, full-study lineage and documented omissions.
Packaging checks do not establish those scientific properties.

After export and before portable own-answer execution, seal the fixture:

```bash
.venv/bin/python chapter16_shapley_attribution_baselines/exercises/part18_spex/publish_study.py seal-fixture /absolute/export/native_query_fixture.json.gz --native-execution /absolute/completed-native-run/execution.json
```

The data instructions supplied with each section must explain how its default
cells find this checked-in fixture, which selected fits/plots can be recomputed,
and which full-study outcomes are recorded comparison values. They must give exact
source-regeneration inputs/commands and model/query-plan acquisition or restoration
instructions for the optional full study. An absolute local cache locator is not
a public download link. Do not claim the review archive contains the full study's
NPZ arrays, SQLite oracle or fitted states.

## Section input manifest

Each author supplies one JSON manifest with schema
`arena.sparse-interaction.section-publication.v1`. All hashes are SHA256. A physical
record contains `absolute_path`, `sha256`, and integer `size_bytes`.

```json
{
  "schema": "arena.sparse-interaction.section-publication.v1",
  "section": "16.18",
  "native_execution_sha256": "<completed shared execution receipt SHA256>",
  "fresh_reproduction_sha256": "<fresh reproduction receipt SHA256>",
  "learner_execution": {
    "absolute_path": "/absolute/learner/own_answer_execution.json",
    "sha256": "<receipt SHA256>",
    "size_bytes": 1234
  },
  "artifacts": [
    {
      "role": "exercises",
      "absolute_path": "/absolute/learner/exercises.ipynb",
      "sha256": "<artifact SHA256>",
      "size_bytes": 1234
    }
  ],
  "data_access": {
    "fixture_target": "chapter16_shapley_attribution_baselines/exercises/part18_spex/expected_outputs/native_query_fixture.json.gz",
    "fixture_sha256": "<shared fixture SHA256>",
    "default_requires_full_study": false,
    "full_study_access": "verified_external_cache_or_source_regeneration",
    "regeneration_instructions_target": "chapter16_shapley_attribution_baselines/exercises/part18_spex/expected_outputs/DATA_ACCESS.md"
  }
}
```

The abbreviated `artifacts` example must be completed with these roles:

| Role | Multiplicity and name |
| --- | --- |
| `exercises`, `solutions`, `page`, `verification_report` | One each; destination names are fixed. |
| `native_summary`, `artifacts_lock`, `verification_schema` | One each, supplied by `native_report.py`; destinations above are fixed. |
| `figure` | One or more; `name` is the final prefixed PNG filename. |
| `data_instructions` | One or more; `name`, such as `DATA_ACCESS.md`, goes under section `expected_outputs`. |
| `learner_data` | Exactly once in 16.18; shared fixture destination is fixed. |
| `supporting_evidence` | Optional named review evidence, historical failures, runtime-amendment lineage, source provenance or READMEs. |

Do not supply publisher-generated roles or conflicting receipt filenames. All
supplied files must pass the existing bounded evidence format validator. Raw
NPZ/NPY, tensors, models, SQLite, pickle, joblib or nested archives are not allowed
as lesson/review artifacts.

The `solutions` input must be the actual own-answer executed notebook, including
its retained outputs, not an unexecuted solution drafting template. Its bytes must
match the execution receipt's `notebook_sha256`. Every published figure must match
the receipt's complete `figures` filename-to-SHA256 inventory; prefix renaming is
allowed because exact figure bytes and multiplicities are checked.

The actual learner execution receipt must include:

```json
{
  "status": "passed",
  "sources_unchanged": true,
  "executed_code_cells": 28,
  "total_code_cells": 28,
  "executed_notebook": "/absolute/learner/own_answer_executed.ipynb",
  "notebook_sha256": "<actual executed notebook SHA256>",
  "figures": {"actual_figure.png": "<actual figure SHA256>"},
  "native_execution_sha256": "<same execution SHA256 as section manifest>",
  "fresh_reproduction_sha256": "<same reproduction SHA256 as section manifest>",
  "portable_replay": {
    "status": "PASS",
    "fixture_sha256": "<same shared fixture SHA256>",
    "no_full_study_cache_required": true
  }
}
```

`FULL_LEARNER_PASS` is also accepted as an existing receipt status. Cell counts
above are illustrative: record the actual equal nonzero totals. Keep additional
source inventories, before/after hashes, timings, controls, fit counts, plots and
replay evidence in the original receipt. Do not create these values merely to
satisfy packaging. The independent review must inspect actual execution evidence.

## Stage, inspect and approve

```bash
.venv/bin/python chapter16_shapley_attribution_baselines/exercises/part18_spex/publish_study.py stage --native-run /absolute/completed-native-run --fresh-run /absolute/cold-fresh-run --section-manifest /absolute/1618/publication_inputs.json --section-manifest /absolute/1619/publication_inputs.json --output /absolute/new-publication-candidate
.venv/bin/python chapter16_shapley_attribution_baselines/exercises/part18_spex/publish_study.py validate /absolute/new-publication-candidate
```

Staging verifies cold completed receipts, unchanged physical source inputs, every
summary-linked native record, the prediction/audit boundary, SQLite bytes, and the
supervised reproduction detail hash. It preserves explicit failed reference
outcomes; a completed study does not imply every author comparator succeeded.
It accepts a prospectively approved resource amendment already validated by the
frozen runner. Earlier interrupted runs are retained as failures and supplied as
supporting evidence; they are not relabelled completed runs.

The candidate contains the exact canonical `tree`, `candidate_manifest.json`,
bounded review archive/contract and full external file inventory. Every retained
file in both completed runs is hashed, including all numerical arrays and all
differences, regardless of inclusion in the review archive. Nothing is deleted.
Bundle/member/contract bounds remain those in `scripts/evidence_bundles.py`.
Exceeding a bound is an explicit packaging failure, not permission to omit evidence.

A separate review JSON must have status `APPROVE_SPARSE_INTERACTION_PUBLICATION`,
`independent_review: true`, the exact `candidate_manifest_sha256`, ordered
`sections: ["16.18", "16.19"]`, and physical `review_evidence` records. Its `checks`
must explicitly pass `scientific_outcomes`, `visible_figures`,
`controls_and_failures`, `portable_student_replay`, and `source_and_reproduction`.
This requirement applies only to installation, not creation of the review packet.

Reports and artifact locks bind upstream native, cold reproduction and learner
evidence. They cannot bind the hash of a downstream archive that includes those
same reports. The separate candidate manifest binds the final review archive and
its strict contract, avoiding a circular report/archive hash requirement.

```bash
.venv/bin/python chapter16_shapley_attribution_baselines/exercises/part18_spex/publish_study.py install /absolute/new-publication-candidate --approval /absolute/independent-publication-review.json
```

Installation verifies the complete candidate, archive, external evidence and review
bytes again, preserves any prior target under candidate `previous_canonical`, and
writes a retained `installation.json`. It copies only these two sections' declared
canonical paths. Root separately owns shared registries, navigation and HANDOFF.
Notebook metadata and scientific reports are not rewritten during installation.
