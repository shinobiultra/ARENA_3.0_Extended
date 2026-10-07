# Data for the SPEX and ProxySPEX lessons

## The normal notebook route

Both lessons use the same portable file:

```text
chapter16_shapley_attribution_baselines/exercises/part18_spex/expected_outputs/native_query_fixture.json.gz
```

The adjacent `.data-contract.json` binds its compressed and decoded bytes. The
loader checks that contract, the prompt/query inventory and the numeric arrays.
Keep the two files together. You can point the notebook at another copy of the
same pair with `ARENA_NATIVE_QUERY_FIXTURE`; that changes the file location, not
the model task or the expected scores.

Run the notebook from this repository using its project Python environment.
The default path needs no Hugging Face login, model download, GPU, historical
SQLite database or fitted tree objects. You implement the method and fit it to
the supplied model responses. The main and native play cells recompute selected
real/shuffled conditions and compare their predictions point by point with the
recorded experiment. Changing a prompt, seed or budget is an exploratory refit.

The portable file contains all 32 prompt definitions and their actual training
and held-out scores, the frozen ordered mask designs, all 1,152 condition metrics,
the largest signed terms for each condition, and all nine reference outcomes.
Full prediction vectors are included for the main/play cases chosen before the
model run. It does not contain fitted trees or the full raw experiment archive.
The aggregate curves across all prompts are recorded comparisons: executing one
notebook does not repeat all of those fits or query GPT-2 again.

The constant and additive predictors, shuffled-response fits, subject-only
counterfactuals and unsuccessful reference outcomes are part of the data. They
are not filtered to make a preferred explanation look better. Fourier terms
describe the defined fixed-position masking game; they are not automatically
Shapley interactions or causal statements about natural language.

## Where the scores come from

The experiment uses `openai-community/gpt2` at revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, CPU float32 eager inference, and the
32 fixed prompts constructed by `native_data.prompt_records()`. Each has 28 word
features. Removing a word masks its token keys and replaces its token IDs while
preserving absolute positions. The final suffix remains present. The score is
the difference between the two declared candidate-name logits.

`native_query_plan.py` fixes the masks, ordering, folds, permutations and seeds.
Every prompt uses 14,549 distinct training masks and 512 common disjoint audit
masks. The complete experiment declares 481,999 physical model rows, including
numerical probes and subject counterfactuals. Per-condition requested masks and
actual new model rows differ because the physical oracle caches shared masks.
The final native receipts, rather than these planned counts alone, establish
which evaluations completed.

## Full model reproduction

The optional full run is separate from ordinary notebook study. It requires the
pinned model/tokenizer snapshot, query-plan files, reference environment, exact
Python environment, approved freeze and the freeze's linked historical evidence.
The historical execution receipts contain absolute locations; those locations
are provenance, not public download URLs.

The checked-in bounded review archive is not a backup of all raw NPZ arrays,
SQLite values, model weights or fitted experiment records. To reproduce the
historical full run exactly, restore the retained experiment cache and its
hash-bound inputs at their recorded locations, then validate them before launch.
The repository alone supports portable notebook replay; it does not supply every
historical full-run dependency. A new location or changed Python environment
requires its own prospectively recorded provenance rather than edits to the
historical receipts.

On the recorded maintainer checkout, the strict environment uses
`.venv/bin/python3`. The `python` alias can name the same binary but still differ
in `sys.executable`, which the frozen validator checks. Validate explicitly:

```python
from pathlib import Path
from chapter16_shapley_attribution_baselines.exercises.part18_spex.native_study import validate_freeze

validate_freeze(Path(
    "chapter16_shapley_attribution_baselines/exercises/part18_spex/native_freeze_runtime_v2.json"
))
```

The declared inputs are in `native_freeze_runtime_v2.json`; the complete original
scientific design and its disclosed resource amendment are in
`native_protocol.md`. Use a new output directory for each cold run. Never resume
by reusing partial values or fits. Each run retains its failures and resource
measurements. The full source-only comparison checks all saved scientific data
under the frozen numerical tolerances and distinguishes bitwise matches from
nonzero differences within tolerance.

## Rebuilding a publication candidate

After complete native execution and independent cold reproduction, use
`native_learner.export_native_fixture` with their actual hash bindings, then
`publish_study.py seal-fixture`. Execute each builder's full own-answer route
against that fixture and inspect the actual figures and play results.

`native_report.py` assembles the supporting reports from those completed inputs.
Its CLI requires the native and fresh directories, actual learner receipt,
portable fixture, canonical-to-physical artifact mapping and a new external
output directory. The legacy `scripts/run_extension_verification_reports.py`
smoke/GPU writer is not this section's report entrypoint.

See `NATIVE_LEARNER_API.md`, `NATIVE_REPORT_API.md` and `PUBLICATION_API.md` for
exact interfaces. Staging creates a reviewable candidate. Installation follows
an independent review of its exact bytes; synthetic tests alone do not establish
a completed native result or a published lesson.
