# Priority pilot method repairs (2026-10-16)

## Repair plan

- Current learner failure: 7.1 weights short final batches like full batches, and nine method exercises across 7.1/8.4 lack immediate independent semantic checks. 8.4 attention-offset slicing also skips valid queries when the offset exceeds `start_pos`.
- Target section claim: preserve the lessons' lens/Patchscope comparisons and circuit interventions while making token denominators and knockout semantics checkable before expensive calls.
- Signature result: retain the existing real-model layer curves, Patchscope controls, circuit heatmaps and completeness/minimality plots as historical reference evidence with explicit provenance; no new model result is authorized.
- Student-implemented functions: `train_tuned_lens`, `evaluate_lenses`, `shifted_translator_agreement`, `patchscope_controls`, `single_edge_effects`, `attention_by_offset`, `top_k_circuit`, `completeness_scores`, `edge_drops` stay visible in master/notebook exercises.
- Controls/baselines: tiny CPU fixtures independently check unequal batch/token counts, exclusion of position zero, shifted indices, real/text/random/BOS controls, valid edge masks, identical knockout sets, negative edge effects and no mutation.
- Play cells: preserve existing learner play cells; repaired metrics must support changed batch sizes correctly.
- Files to modify: the 7.1/8.4 masters, current generated exercises/solutions/pages, section `tests.py`, focused mastery regression tests and this evidence record. Original ARENA and shared build/config files remain untouched.
- Tests to run: bounded CPU semantic and mutation checks, generated artifact consistency and notebook validity; builds without `--execute` only.

## Authority and review boundary

The assigned repair permits fake-model/CPU checks and offline generation. It excludes model loading, downloads, model studies, GPU/cluster work, commits, staging and publication.
Scientific execution and learner acceptance remain separate. The existing 7.1 published configuration has 200 test sequences of length 64, batch size 20; 8.4 uses start position 8
and maximum offset 6. Equal batch weighting and nonnegative diagonal slicing respectively make these published configurations mathematically unaffected. Numerical checks and exact
artifacts will be recorded below; affected new configurations require fresh execution.

## Decision queue

- PMR-1 (settled by assignment): correct weighting to evaluated tokens, including a short final batch.
- PMR-2 (settled by source inspection): correct negative diagonal slicing with a clamped start; retain existing `start_pos=8,max_offset=6` semantics.
- PMR-3 (pending human/scientific acceptance): accept a future full fresh execution of revised notebooks. This lane cannot run it.

## Implemented repairs

`evaluate_lenses` now accumulates tokenwise KL sums and correct-prediction counts for all four fields in float64 CPU totals, then divides by the total number of non-BOS tokens.
`shifted_translator_agreement` counts correct non-BOS predictions with the same denominator and returns an empty mapping when no shifted translator is valid. Both reject
empty/no-non-BOS data and nonpositive batch sizes rather than returning undefined metrics. Three sequences containing two correct predictions now give agreement 2/3 regardless of
whether the batch size is 1, 2, 3, or larger than the dataset.

`attention_by_offset` clamps the diagonal start to zero, so an offset larger than `start_pos` includes every valid query. Its independent test indexes `(query, query-offset)`
explicitly across two prompts and two heads, rather than duplicating the diagonal expression. KL token reductions stay float32 on the model device before transfer to CPU totals,
avoiding a new unsupported float64 reduction on MPS. No actual MPS execution was performed. No other method definitions changed, as checked against the archived ASTs.

Nine immediate test cells now follow the corresponding method-definition cells. They use small distributions, fake cache outputs and hand-worked nonlinear graphs. They check actual
absolute single-edge metric values, fresh singleton cuts, valid top-k masks with negative scores, shared completeness knockout sets, retained negative minimality effects, row
order, unchanged inputs, seeded training and distinct real/text/random/BOS control paths. The training check uses three pre-update losses and a tight final-weight comparison
against a plain-Python scalar Adam oracle with bias-corrected moments. A focused AST mutation test removes `optimizer.zero_grad()` and confirms the learner check rejects
accumulated gradients; the earlier two-step broad-tolerance test missed that error. Independent re-review passed all 26 tests with no outstanding P1/P2 in this bounded scope. The
tests are supporting scaffolding; the taught methods remain student-implemented in the notebook. Existing real-model signature results, controls and play cells remain visible.

## Historical reference provenance

Before editing, `evidence/pilot_method_repairs_20261016/before.tar.gz` captured 28 current master/section-generated/test/utils/media files with per-file SHA256s in
`before_manifest.json`. The initial page glob incorrectly treated square brackets as metacharacters, omitting the four instruction `.md`/`.py` files. This was detected during final
manifest review. The parent had independently verified all 238 hashes in `second_continuation_source_manifest.json` against current bytes before this lane. Old pages were
reconstructed **only in TEMP** from the exact archived masters and unchanged converter; all four reconstructed page hashes exactly match that pre-lane registry.
`before_pages_reconstructed_verified.tar.gz` retains these verified bytes separately, and `before_pages_reconstruction.json` labels them **reconstructed and verified against
pre-lane hash**, rather than original byte capture. The original 28-file archive was not altered. The saved 7.1 data-cell output explicitly records `torch.Size([1600, 64])
torch.Size([200, 64])`. Its evaluation and shifted-control call sites use the default batch size 20. There are ten batches of 1,260 evaluated tokens each, so the old expression
`(sum_b S_b/1260)/10` equals the repaired `sum_b S_b/12600`. This applies to both KL fields and both agreement fields. CPU regression tests compare the old archived and repaired
functions at the published 200-by-64/batch-20 shape within 2e-7 tolerance; float64 accumulation can change rounding. This is an algebraic equivalence argument with a CPU numerical
check, **not** fresh model execution.

For 8.4, the published call has `start_pos=8,max_offset=6`. Every offset d is at most 6, hence `max(0,8-d)=8-d` and the slice is exactly unchanged. Existing scientific call sites and all other method ASTs remain unchanged.

Both sections were generated with `build.py` **without `--execute`**. The normal builder correctly cleared solutions outputs after source changes. The local, auditable
`preserve_reference_outputs.py` then copied only outputs whose code-cell sources matched the archived source exactly, retaining their prior execution counts and attaching
`historical_reference` metadata with snapshot hash/source hash/old cell index and `fresh_execution=false`. The changed metric dependencies are covered only by the specific
published-configuration argument above. All media bytes remain identical to the archive. A visible note near each lesson's Introduction and notebook-level metadata identify the
saved evidence as historical.

The receipt `reference_preservation.json` records 26 and 29 historical solutions output cells and 6 and 10 exercise reference-output cells. All four notebooks retain these existing
references; **all nine newly added test cells in each notebook variant remain unexecuted with empty outputs and null execution counts**. The separate CPU test receipt is not
written into notebook outputs. After any offline rebuild of these two sections, rerun the local preservation script if retaining these explicitly historical references is intended;
ordinary build generation alone does not carry the provenance metadata forward.

## Validation and failures retained

- `before_tests.txt`: 3 expected failures (uneven-batch lens KL/agreement, shifted agreement, negative-start attention slicing), 6 passing checks before method edits.
- `after_cpu_tests.txt`: **26 passed**, 16 dependency warnings, including nine independent method fixtures, generated-method parity, denominator guards, archived/current equal-batch comparison, immediate unexecuted-cell placement, exact historical output/media preservation published configuration evidence, reconstructed page provenance, and rejection of a missing-`zero_grad` mutation.
- `build_7_1.txt`, `build_8_4.txt`: successful offline generation. No model loading, data download, GPU work, cluster submission or model-study execution occurred.
- An initial offline build rejected the provenance paragraph because the converter requires `# Introduction` to occupy its own special header cell. The paragraph was moved to a separate markdown cell; both subsequent offline builds passed. No model execution was attempted by that failed build.
- The first draft of the CPU harness patched a decorator wrapper's globals rather than its underlying function. It was corrected using `inspect.unwrap` before recording the meaningful red regression baseline; no scientific result came from that harness error.

Reproduction from the repository root:

```bash
UV_CACHE_DIR=/tmp/arena-mastery-uv uv run --no-sync --offline python -m pytest -q tests/mastery/test_pilot_method_repairs.py --disable-warnings
```

## Changed files and preservation boundary

The changes are limited to the two extension masters; each current section's `tests.py`, generated `solutions.py`, exercises/solutions notebooks and instruction page files;
`tests/mastery/test_pilot_method_repairs.py`; this migration note; and `docs/mastery/evidence/pilot_method_repairs_20261016/`. The post-repair file manifest names exact current
artifacts and hashes. The lane did not edit original ARENA source, notebooks or assets, shared build/config files, or other workers' files. The workspace already contained
unrelated edits; this statement describes this lane's ownership, not a claim that the entire checkout matches a clean upstream tree.

## Remaining execution and acceptance needs

PMR-3 remains pending: a fresh full solutions-notebook execution, then human review of student-facing outputs, is still required before presenting the repaired notebooks as freshly
executed or course accepted. New uneven-batch and `start_pos < offset` configurations have only small CPU-fixture evidence here. The historical real-model references show the prior
configurations; their retention does not establish fresh runtime compatibility, real-model performance on changed settings, or learner acceptance. No commits, staging, push or
publication were performed.
