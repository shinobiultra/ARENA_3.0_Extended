# 5.2 full Mixture of Experts master migration

## Implementation plan

- **Current learner failure:** the rich 5.8 synthetic MoE candidate bypasses canonical 5.2 master generation, and its rejected specialization hypothesis must
  survive rather than becoming a positive released-model claim.
- **Question:** can high task accuracy coexist with collapsed routing, and which controls distinguish load balance, router scale, route preference and causal
  expert necessity?
- **Signature result:** preserve the complete six-panel three-seed paired aux×z study, routing/capacity/intervention tables, negative preferred-ablation result,
  compute accounting and exploratory capacity/regularizer play.
- **Student methods:** deterministic probability-gated top-k routing; per-sequence left-to-right capacity; differentiable dispatch/combine; independent experts;
  load-balance/z/factorial effects; sparse layer/block/cached decoder; mixed task; delexicalized probe; expert permutation; exhaustive fixed-model
  interventions.
- **Controls:** top-1 gradient versus unit gate, ties, prefix/batch/cache causality, brute-force dense dispatch, paired aux-only/z-only/neither, equal-width
  dense and explicit unmatched padded work, shuffled probe labels, coherent expert permutation, all-expert ablation/forcing and original failed specialization
  decision rule.
- **Play:** preserve the original 60-step capacity/regularizer play with historical outputs; do not retrain it during migration.
- **Files:** new `master_5_2.py`, canonical `part2_mixture_of_experts` notebooks/module/semantic tests/media, matching pages, this note, scoped
  archive/parity/current-manifest/proposed-case evidence and focused migration tests. Root owns shared config/cases/navigation.
- **Validation:** snapshot all original files first; compare all 49 visible definitions against current original module AST; own-answer completion; strict
  scientific code/output parity with bounded setup/publication-output-route adaptations; negative result/plot/prose retention; safe import; tiny existing
  independent CPU semantic fixtures only; no model study/download/GPU run. New executable notebook cells, if necessary, require separate bounded CPU
  receipts/counts rather than borrowing full-notebook execution evidence.

Original ARENA chapters 0–4 and every old MoE notebook/module/test/page/asset/receipt remain physically unchanged. Existing counts prove only retained
historical publication
outputs: no separate full-own-answer archive/source-hashed execution receipt exists for this source. The new master will state that limit. The released-model
routing extension in
the current brief remains unfinished and outside scope. No scientific result, threshold, data generator, cohort, optimizer or release predicate is silently
changed.

## Implemented learner arc

The canonical master now generates the complete long lesson: 94 cells, 22 code cells, 12 exercise groups covering 21 student functions/classes, and 49 visible
top-level definitions
including supplied experiment helpers. Ten method-specific forecast/reflection pairs precede and follow routing, capacity, dispatch, losses, model execution,
mixed-task evaluation,
probe/permutation, exhaustive interventions, the paired negative study and the original play. Core implementations, immediate independent tests, help and
solution dropdowns remain
visible. The original 60-step capacity/regularizer play remains intact, with its retained historical tables.

The signature is the complete original six-panel figure, visually inspected during migration: final and longer-sequence accuracy, dev-selected final routing,
paired aux×z effects,
exhaustive expert ablations, padded FFN/router work, and router-scale controls. Both original PNG output objects survive (one unique figure rendered twice), as
do all 17
output-bearing cells and all 22 historical execution counts. Aux+z balancing improves load entropy, but the context-only probe stays near shuffled-label chance
and the
preferred-ablation hypothesis remains rejected. The original positive-specialization release predicate is unchanged and false. Equal-width dense
compute/parameter mismatches,
current-token conditioning, finite-capacity/cache limitations and the distinction between forcing and ablation remain explicit.

## Preservation and adaptations

`published_before_master.tar.gz` contains all 13 original MoE notebooks, source, builder, tests, page, asset, lock, schema and historical output files, captured
before canonical
publication. `before.json` records each exact hash; all 13 remain physically unchanged. No original ARENA chapters 0–4 files were edited. The current legacy
builder reconstructs
all code exactly and only reflows markdown cells 0, 27 and 30 with word-token parity; it was checked with publication paths redirected to a temporary directory.

All 49 definition ASTs match both the archived notebook and original module. All 22 canonical solution cells match archived scientific code by nonblank lines
and AST after only the
explicit substitutions in `route_adaptations.json`: setup imports/section become `part2_mixture_of_experts`, and the future signature figure write goes to the
new canonical media
directory. This prevents canonical execution overwriting the preserved old asset. Python-only MAIN guards make imports safe and do not alter notebook scientific
source. Supplied
immutable configs, attention and positional plumbing still come from the preserved legacy module; every taught method and all scientific helpers remain visible
in the notebook.

The canonical semantic-test module is the exact original prefix containing 29 independent CPU method oracles. Original training/publication-contract tests
remain physically at the
old path; they were not repurposed into a claim of fresh scientific execution. All 21 own-answer implementations, including decorators and four class stubs,
reconstruct the
solution AST. Difficulty/importance values and time estimates are retained, with converter-compatible rating formatting. The old numeric identity
`5_8_mixture_of_experts_from_scratch` and release-wrapper namespace remain archival identities; current canonical 5.8 is a different topic and no old checkpoint
is adopted by
number.

There is no independent full-own-answer source-hashed execution receipt for the original MoE study. The retained published notebook and reference JSON provide
historical evidence
with exact current source/AST parity, not independently recovered execution provenance. `output_parity.json` explicitly records that limit. The migration does
not establish a fresh
full notebook run, CUDA parity or a positive release. The broader current 5.2 released-small-MoE routing brief remains unfinished.

## Fresh bounded CPU evidence

After root authorization, `cpu_subset_protocol.json` froze source identities and runtime protocol before one Jupyter execution of exact canonical code ordinals
0–19: setup, core
methods, immediate tests and pure helper definitions. Two separately labeled administrative cells set/check one CPU thread and deterministic fixture seeds.
Signature study ordinal
20 and original training-play ordinal 21 were excluded. CUDA was hidden/uninitialized, offline environment flags were set, and an external 60-second wall bound
plus conservative
process CPU limits applied. This single run passed in 2.864 seconds wall and 2.632 aggregate CPU seconds. `cpu_methods_executed.ipynb` and
`cpu_subset_execution.json` retain actual
counts, outputs, source/AST hashes and CPU runtime. These actual counts are separate from the unchanged canonical historical counts; no full-notebook provenance
promotion occurred.
No training study, play retraining, model download, GPU or cluster run occurred.

The focused migration suite passes 41 tests, including the 29 existing semantic oracles, safe import, original/archive recovery, all code/output/count parity,
completed student
answers, negative findings, proposal validation and the actual bounded CPU receipt. Rebuilding through the canonical converter preserves the retained outputs.
`current_publication.json` describes the current publication with per-code source/AST/output/count identities and informational whole-file hashes. Shared
navigation integration
must run `python docs/mastery/evidence/moe_master_migration/refresh_current_manifest.py` afterwards; the immutable archive and execution protocol must not be
rewritten. This avoids
binding preservation solely to a pre-navigation whole-notebook hash.

## Guided case proposal and integration

`proposed_case.json` is a strict 20-key proposal only; root owns adoption into shared registries. Its binary primary asks whether three-seed mean dev-preferred
expert ablation
damage exceeds mean nonpreferred damage. Observation uses exact retained aggregate and per-seed reference JSON values with a `historical_reference_json` source
receipt and original
hash. The figure's representative seed-7 panel must not substitute for the three-seed target. Its transfer changes only ablation to forcing within the same
trained models, data,
preferences and equal-seed averaging; this intervention was already published and is explicitly exposed guided practice. Neither case is learner-unseen evidence
or a newly measured
result.

Root integration still owns registry/checkpoint adoption and navigation generation, followed by a current-publication manifest refresh and focused tests.
Independent review is separate from learner acceptance and from scientific release acceptance.

Independent review passed with no remaining P1/P2 findings; separate evidence is in `docs/mastery/evidence/moe_independent_review_20261016/`. Reviewer ran
structural/parity/import/CPU-receipt checks and the corrected proposal identity check without rerunning the scientific study. Root shared integration remains.
