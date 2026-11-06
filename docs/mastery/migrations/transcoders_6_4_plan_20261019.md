# Draft preservation plan: canonical 6.4 from complete legacy 6.3

Status: planning only; root chooses next work after failure diagnosis. Snapshot 2026-10-19T03:22:24.221083+00:00. No adoption, fresh execution, new method, or
result is authorized by this note.

## Learner and claim boundary

**Current learner failure:** canonical `part4_transcoders` has no complete canonical master/route while a substantial legacy lesson remains at
`part3_transcoders_attribution_graphs`. A numeric rename must not discard its exercises, inline implementation, stored figures, controls, interpretation or
play.

**Target preserved claim:** an ordinary sparse ReLU transcoder exactly replaces a hand-specified colored-shape MLP, exposes signed input-to-feature/readout
paths, and supports exact feature intervention and matched-control checks. Its separately recorded GELU-1L preflight is supporting historical evidence.

**Expanded canonical brief is not yet covered:** syllabus 6.4 requests training an MLP replacement, comparison against an output SAE, adding a skip connection,
and input-invariant inter-feature virtual weights. The primary legacy learner arc is a hand-specified one-layer ReLU organism. A trained tiny transcoder exists
in optional historical GELU-1L plumbing, but this is not an inline learner training/SAE/skip-transcoder comparison. Do not silently advertise full
`sparse.skip_transcoders` coverage or import new lessons. Preserve the full ordinary-transcoder source first; root must explicitly choose a scoped claim versus
separate future expanded-brief work.

## Concrete preservation mapping

| Existing source | Proposed canonical destination / treatment |
|---|---|
| `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs` (entire directory) | Leave intact, including legacy IDs, tests, report, fixtures and optional CUDA wrappers. |
| `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs/6.3_Transcoders_and_Attribution_Graphs_exercises.ipynb` and `_solutions.ipynb` | Append canonical pair under `chapter6_sparse_feature_methods/exercises/part4_transcoders/`; exact final display-title basename is root-owned. Preserve every cell and stored solution output, adjusting only reviewed route/identity metadata. |
| Complete 29-cell learner/solution arc | New `infrastructure/chapters/chapter6_sparse_feature_methods/master_6_4.py`; keep meaningful method definitions and solution dropdowns inline. |
| `chapter6_sparse_feature_methods/instructions/pages/03_[6.3]_Transcoders_and_Attribution_Graphs.md` | Retain legacy Markdown; add canonical 6.4 generated instruction route only after route collision checks. |
| `instructions/assets/transcoder_attribution_toy_ground_truth.png` and `transcoder_attribution_faithfulness.png` | Keep existing assets byte-identical and point canonical page to them; no redraw or result regeneration required. |
| `verification_report.json`, `artifacts.lock.yml`, `expected_outputs/*` | Preserve historical 6.3 identity and timestamp; canonical narrative may reference provenance without rewriting it as new 6.4 execution. |
| `solutions.py`, `tests.py`, `utils.py`, `arena_ext/transcoders.py` | Preserve reference/protocol helpers. Canonical bridge must be explicit; tests accept provided student functions and must not silently exercise only reference methods. |

The legacy README names `_build_salvage_notebooks.py`, but no such file was found in the repository file inventory or source directory. Treat that builder claim
as stale; do not invoke an assumed publisher. Build a canonical master directly from the complete notebook pair only when implementation is assigned.

## Cell preservation and inline teaching map

Cell numbers below are zero-based solution-notebook indices. Counts: 29 cells, 13 code cells, 12 output-bearing code cells, 3 embedded PNG outputs; the page
references two named standalone figures. Preserve the third embedded figure associated with the play rather than dropping it because it lacks a named standalone
asset.

| Cells | Preserve teaching or behavior |
|---|---|
| 0–2 | Complete question, objectives, setup, dataclasses, organism/input constructors, exact replacement equation. |
| 3–4 | `encode`, `decode`, `transcoder_forward`; immediate exact-MLP replacement tests and expected-output/help/solution guidance. |
| 5–6 | `reconstruction_decomposition`; feature components and bias reconstruction test. |
| 7–8 | `feature_edge_attributions`; signed bias paths and edge conservation. |
| 9–10 | `extract_attribution_graph`; signed incoming paths and feature ranking. |
| 11–12 | `plot_attribution_graph`; visible attribution heatmap plus graph. |
| 13–14 | `edge_conservation_score`, `shuffle_input_edge_targets`; shuffled-edge negative control. |
| 15–16 | `intervene_features`, `causal_validate_graph`, `faithfulness_curve`, `reconstruction_only_ranking`; retention/ablation and matched rankings. |
| 17–19 | `plot_signature_result`, primary controls and result interpretation, with stored outputs retained. |
| 20–21 | Try It Yourself: color, shape and graph feature-count knobs; inspect signs and intervention behavior. |
| 22–23 | Missing-edge anomaly hunt and feature-level conservation localization. |
| 24–25 | Clearly separate recorded real-model supporting metrics read from legacy report. |
| 26–27 | Optional `run_gpu_test` / `run_full_experiment` wrappers remain opt-in, with no execution implied by migration. |
| 28 | Full limitations and paper links. |

Core method must remain editable inline. Shared utilities may locate files and load historical metadata; they must not replace the teaching cells. Preserve
fixed same-size random graph, shuffled feature identities, shuffled edge targets, and reconstruction-only decoder-norm ranking as separate controls. Avoid
collapsing them to an average control.

## Prediction opportunities, without new answers

- Before opening stored signature figures, predict which ranking best preserves the target behavior at a fixed graph size and explain what reconstruction
  quality alone can miss.
- Before opening the missing-edge anomaly output, predict which conservation check should reveal an omitted signed bias edge.
- For the existing play knobs, commit a directional prediction about changing color/shape or retained feature count before viewing the existing plot. A changed
  knob is a new exploratory condition, not a new accepted benchmark.

All existing answers and figures are public in notebooks/pages/fixtures. Label these opportunities **historical exposed guided practice**; record prior
exposure, confidence and reasoning. Do not call them blind, held-out assessment, or fresh experiments. Root should choose one small entrypoint only after
confirming whether the existing checkpoint registry has changed. No answer/result was computed for this plan.

## Duplication and protected boundaries

No canonical chapter-6 master directory exists at this snapshot. No adopted extension checkpoint claims section 6.4. A repository-wide static text scan for
transcoder mentions found the following masters (mentions indicate overlap, not duplicate implementation):

- `infrastructure/chapters/chapter1_transformer_interp/section_3_probing_and_representations/master_1_3_3.py`
- `infrastructure/chapters/chapter1_transformer_interp/section_4_circuits/master_1_4_2.py`

Original ARENA `master_1_4_2.py` already contains extensive SAE-circuit/transcoder teaching, pretrained-transcoder loading/replacement, graph construction, and
`W_skip` handling (including its bonus manual propagation). Original `master_1_3_3.py` introduces transcoders. These are canonical protected originals: link to
them as context, never modify them or silently copy their richer model-specific curriculum into a toy 6.4 migration. The extension organism complements original
material by providing exact interpretable signed-edge ground truth; it does not supersede the original course.

Potential conceptual overlap with adopted circuit sections is a reason to cross-link, not replace their ACDC/AtP definitions or evidence. Numeric collision is
especially dangerous: legacy 6.4 is Crosscoders; canonical 6.4 is Transcoders. Preserve the legacy `04_[6.4]_Crosscoders_and_Model_Diffing.md` and source
directory. Route registration and page names must use subject and canonical ID together, never overwrite by number alone.

**Exact ownership boundary:** this lane writes only this plan. Future worker may own the new canonical 6.4 master, destination directory and new page after
assignment. Root owns shared `syllabus.yaml`, generated config, checkpoint registry, navigation and any status/adoption changes. Do not touch concurrent 16.5,
other migration plans, any original ARENA material, legacy crosscoders/transcoders, shared helpers, or runtime environments.

## Dependencies and future verification gates

Static imports identify PyTorch, torch.nn.functional, Matplotlib and standard-library JSON/path/dataclasses, plus legacy semantic tests. Optional GELU-1L helper
uses TransformerLens and existing verification machinery; no model/cache/network lookup was performed. Notebook setup and optional wrappers must resolve old
versus new section directories intentionally. The exact toy teaching and reporting read must not auto-launch optional GPU work.

Before future publication: preserve source/output hashes; verify every teaching cell and solution-dropdown implementation survives; compare normalized code and
stored output objects across migration; verify both named assets and embedded play output survive; inspect resolved imports/media routes; compare
student-completed method AST to solutions where practical; run only separately authorized bounded tests. A page build or green static check cannot accept
skip-transcoder coverage or establish a fresh scientific result. No tests or methods were run in this planning task.

## Historical protocol identity and current hash comparison

Historical report date: `2026-09-04T23:53:08+00:00`; report commit: `8eab192b`. Its `accepted=true` is a historical field, not acceptance of this migration.
Model pin: `NeelNanda/GELU_1L512W_C4_Code` revision `bddc0e332f0ae84279e6a6a45d91b314899e1603`; optional preflight seed, prompt counts, hooks and bounds remain
in `artifacts.lock.yml`. No new protocol or learner-execution archive was located in the source directory.

Current source hash comparisons against report_inputs (read/hash only; no report recomputation):

| Report input | Current comparison |
|---|---|
| `arena_ext/verification.py` | match |
| `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs/6.3_Transcoders_and_Attribution_Graphs_exercises.ipynb` | match |
| `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs/6.3_Transcoders_and_Attribution_Graphs_solutions.ipynb` | match |
| `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs/README.md` | match |
| `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs/artifacts.lock.yml` | match |
| `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs/expected_outputs/README.md` | match |
| `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs/expected_outputs/reference_metrics.json` | match |
| `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs/expected_outputs/smoke_test.json` | match |
| `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs/solutions.py` | match |
| `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs/tests.py` | match |
| `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs/utils.py` | match |
| `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs/verification_report.schema.json` | match |
| `chapter6_sparse_feature_methods/instructions/pages/03_[6.3]_Transcoders_and_Attribution_Graphs.md` | match |
| `scripts/run_extension_verification_reports.py` | match |

All 14 report-input hashes match at this snapshot. Any future mismatches in shared infrastructure or presentation inputs must remain disclosed; do not refresh
provenance to hide them. Matching input hashes demonstrate identity only, not correctness of stored numerical outputs.

## Source and provenance SHA-256 snapshot

| Physical path relative to repository | SHA-256 |
|---|---|
| `arena_ext/transcoders.py` | `ae029daaa8dbd04d46faf315bd8e587c30c62871cb242237818cc65ff8303fa7` |
| `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs/6.3_Transcoders_and_Attribution_Graphs_exercises.ipynb` | `e6f87a7cf003a81af3ec628136c296ff1894954f32c31071def6473e58cbd771` |
| `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs/6.3_Transcoders_and_Attribution_Graphs_solutions.ipynb` | `af01d2a4aeed111dca3b57727bc0289636cae3bc47d1d3882df19a933cf7e80e` |
| `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs/README.md` | `756ab097226ee625163ae07db4eb26c57439f09334dbfde8bc62044a5d46a8e4` |
| `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs/artifacts.lock.yml` | `31a48d51f48edd655bfbd7b59614cc51c5ef22fad17a685f1ace82100ad721da` |
| `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs/expected_outputs/README.md` | `2c64b74fc6f5d1f556ddfbe3eaff9fd83f512023fdcb97c2082bae49d1dfadab` |
| `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs/expected_outputs/reference_metrics.json` | `9cdac02c288f40146da32a44e6382517e4cf53e4c89736d7bac0d36dae757e0e` |
| `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs/expected_outputs/smoke_test.json` | `a718812f2458597e54385390636555e3839afa218581e8d506129667995faca9` |
| `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs/solutions.py` | `196566ed25b8faaae8e77cba8ea1cbe6c6844017664fe7590a57cda4fdd60c4d` |
| `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs/tests.py` | `810e27ee14eab2d8055001a16615ef2e6f96a022a5c55613ae0eff0c1824a76c` |
| `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs/utils.py` | `8f2e9130c5821a18ede8b2424118c35cb1216e2ae0b8d4b385976b7a20f4c9a9` |
| `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs/verification_report.json` | `9c008356a73bf32a6ebc736dea66171f4d4fc05d82a3d4da946250c37ca9d9eb` |
| `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs/verification_report.schema.json` | `ecd833883885077ad5d7a0e239d984d01462255e94266212cf195ab4d4e27b3d` |
| `chapter6_sparse_feature_methods/instructions/assets/transcoder_attribution_faithfulness.png` | `2603ab5052ba562bed307948e0d0299a88ab93960f67504bc564b976e5971d0e` |
| `chapter6_sparse_feature_methods/instructions/assets/transcoder_attribution_toy_ground_truth.png` | `cd64549e18af108a5171fc0c1fe1168f1160ea9c17737371815ffbe97f137bfc` |
| `chapter6_sparse_feature_methods/instructions/pages/03_[6.3]_Transcoders_and_Attribution_Graphs.md` | `0b4d97800ba794950d269debfe69d3be0218d273398b831b33d30237a6801732` |
| `infrastructure/extension/syllabus.yaml` | `f416f0f339ec44aa5b7a0b97ed7393f3a79aa5b3ce286cf9a8a323e82cd13945` |
| `mastery_hub/extension_checkpoints.json` | `9ebca291ca2cd7272fc10087c3004f3cc48e22915cca036be5bc78e1c68accbe` |
| `infrastructure/chapters/chapter1_transformer_interp/section_4_circuits/master_1_4_2.py` | `6af8659c20ad18cfcc8cedff00c5d47e05dbe1b2464259a24d22cfe000f6029e` |

## Unresolved decisions for root

- **TC64-01:** Adopt only a preservation-complete ordinary-transcoder migration with explicit skip/SAE/virtual-weight gaps, or defer publication until expanded
  coverage is separately authorized? Recommendation: preserve the complete legacy curriculum and keep gaps explicit; do not invent new topics/results.
- **TC64-02:** Canonical notebook/page display title and routing coexistence with legacy crosscoders 6.4. Recommendation: root resolves canonical filenames
  while preserving both legacy routes.
- **TC64-03:** Timing of bounded CPU validation versus static-only migration. No execution authority is created here; root decides after the current failure
  diagnosis.

Original ARENA preservation: unchanged. Student-facing improvements/signature results/controls/play described above are proposed preservation work, not
completed publication. Only this plan was written.
