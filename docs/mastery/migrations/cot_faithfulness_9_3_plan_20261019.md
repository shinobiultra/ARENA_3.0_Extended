# 9.3 Chain-of-Thought Faithfulness: read-only migration preflight

Status: **plan only**. This note and the uniquely named source inventory
`docs/mastery/evidence/cot_faithfulness_9_3_preflight_20261019_source_only/inventory.json` are the only writes. No production source, notebook,
registry/config/workbook, existing migration evidence or review was changed. No taught module was imported; no tests, data/cache access, model, GPU, generation
or scientific computation ran. All observations below come from repository source, historical JSON/notebook outputs and source-byte hashes. Implementation
remains a root decision after its broader failure review.

## Scope decision comes before implementation

Canonical syllabus entry `9.3`, `part3_cot_faithfulness`, cites the historical `part2_cot_faithfulness` and `arena_ext/cot_faithfulness.py`. Its expanded brief
asks for small-instruct-model arithmetic/word problems; CoT truncation, mistakes and paraphrases; biasing features; answer probing before CoT; and causal CoT
patching.

The existing curriculum does **not** establish that expanded brief. It is an exact deterministic A/B toy plus an explicitly negative Pythia-70M
private-answer/prompt experiment. The real path uses answer-token logits, hidden states and forward hooks; it generates no reasoning traces or completions. Its
prompts already contain the answer in plain text. Renumbering it cannot supply truncation, paraphrase, mistake-injection, natural reasoning, instruction
following or arithmetic evidence. Those require a separately proposed scientific design and execution authority, not a migration default.

Recommendation **U9.3-01**: migrate the complete legacy learner experience as a clearly labeled historical controlled model organism and falsification lesson;
retain the expanded-brief gaps visibly as unresolved. Do not claim full 9.3 brief coverage or add studies while repairing missing canonical artifacts.

## Salvage plan

- **Current learner failure:** the canonical route is absent; the existing introduction/README/report wording is partly verification-oriented; real hook
  orchestration is behind a reference-module wrapper; and the optional real-model output is absent from the executed solution notebook. Nevertheless, the legacy
  notebook already has a substantial eight-exercise learner arc and a visible controlled signature.
- **Target question:** when a rationale conflicts with an explicitly given answer, what can decodability, a causal state intervention and the strongest control
  establish—and what would falsify the claim?
- **Signature:** preserve the toy layer-position heatmap, toy patch-effect/flip panels and all qualitative examples; preserve the negative Pythia report and its
  complete source rather than substituting a success narrative. A report is supporting evidence, not the main teaching visual.
- **Student methods:** retain all existing visible methods and each exercise's own answer, tests/expected output, help and interpretation. Keep probe
  construction, hook-output replacement and donor-margin/control aggregation visible. Do not relabel a copied hidden helper or report utility as a student
  implementation.
- **Controls:** retain text-only, rolled-label, random-direction, named random-donor and irrelevant-position conditions, exact toy ground truth and all
  qualitative/tokenization checks; explain their actual constructions rather than relying on names.
- **Play:** retain the exact CPU toy layer/position/target/donor play, anomaly hunt and optional explicitly disabled-by-default CUDA path. Preserve every
  original default and its negative boundary.
- **Files after a separate GO:** new canonical `master_9_3.py`, derivative master notebook, `part3_cot_faithfulness` notebook pair/module/README/media,
  generated 9.3 pages and dedicated migration tests/evidence. Root owns shared registry/config/navigation/workbook decisions. Original ARENA and all historical
  9.2 files remain untouched.
- **Verification after GO:** strict source archive/closure, own-answer completion, scientific-code inverse proof, exact historical outputs/counts/metadata,
  expected-only exercise displays, import/default/decorator/class-base audit, transitive eager-import audit, guarded import if authorized, local figure links,
  strict case schema, candidate freeze and independent review. A source migration does not authorize any toy/model/study execution.

Original ARENA
`chapter1_transformer_interp/exercises/part32_function_vectors_and_model_steering/1.3.2_Function_Vectors_&_Model_Steering_{exercises,solutions}.ipynb` was
inspected as the closest intervention-teaching pattern. It connects concrete tasks, visible intervention code, immediate exercises and interpretations. Use that
structure without importing its results or changing original ARENA.

## Exact source surface

Historical directory: `chapter9_alignment_interpretability/exercises/part2_cot_faithfulness`.

| Artifact | Current content |
| --- | --- |
| `9.2_Chain_of_Thought_Faithfulness_exercises.ipynb` | 47 cells; 15 code cells; 68,245 bytes; zero output records |
| `9.2_Chain_of_Thought_Faithfulness_solutions.ipynb` | 47 cells; 15 code cells; 217,904 bytes; 14 historical output records: twelve streams and two PNG displays |
| `solutions.py` | 1,029 lines / 42,210 bytes; exact toy methods plus complete optional pinned CUDA hook experiment |
| `tests.py` | 339 lines / 15,280 bytes; immediate semantic checks, contract/negative-result checks and notebook-pedagogy check |
| `utils.py` | Twelve-line report-printing helper; boring presentation plumbing |
| Three expected-output files | README, small smoke contract and toy-reference metrics; not a fresh native execution archive |
| Lock/report/schema/README | Historical model/prompt/claim boundaries and report inputs |

There are **twelve source-tree files totaling 377,422 bytes**, excluding `__pycache__`. No source archive (`.tar.gz`, raw tensor/model/data archive) is present
inside that tree. Repository filename search found no dedicated CoT execution archive or source-only reproduction packet. This does not establish that none
exists outside the inspected repository: external caches/weights/activations are **unprobed**.

The historical page is `chapter9_alignment_interpretability/instructions/pages/02_[9.2]_Chain_of_Thought_Faithfulness.md` (51,999 bytes). Its only local figure
path is `instructions/assets/cot_faithfulness_toy_ground_truth.png`.

Preserve **three byte-distinct visual sources**, not an assumed duplicate:

- Solution cell 21 output 0: standalone heatmap PNG, 494×340, 30,235 bytes, SHA256 `3b3c8266bb0975126da4ed141e82d5faa9ea7a3391e2a54c13cfd7001206a56f`.
- Solution cell 33 output 1: toy multi-panel PNG, 1299×397, 66,732 bytes, SHA256 `9c54bc203d65a4ea512933e0981d95fb6d7fd0bd78eaf64dbd6c897a3b259123`.
- Published expected asset: 2336×710, 140,620 bytes, SHA256 `e949d3ef6a8ced6288a03ec352b73c287f76264dacbfef5f101b8c56495c07a2`.

These likely represent different saved/display renderings of related teaching plots, but only byte identity/dimensions were checked; do not silently deduplicate
or claim pixel/visual parity. A future visual review should determine the canonical expected display while preserving every original byte. No Pythia signature
asset is committed: `plot_gpu_signature_result` would write `cot_faithfulness_signature_result.png` only on an optional live rerun.

The solution notebook's actual optional-run stream states that CUDA was skipped (`RUN_GPU_SIGNATURE=False`); its qualitative-table cell says to run CUDA first.
Therefore its two PNGs are **toy outputs**, not Pythia figures. Preserve these skipped-path records truthfully. Do not construct a new GPU plot from report
numbers and present it as executed notebook evidence; any historical-report visualization requires an explicit separately labeled adoption decision.

## Complete existing learner arc

Eight exercise groups teach:

1. `make_toy_cot_batch`: 24 examples, four layers, four positions, width eight; explicit answer/rationale axes and readout.
2. `prediction_accuracy`, `answer_logit_diff`: binary `[A,B]` scoring and B-minus-A sign.
3. `fit_mean_difference_probe`, `probe_logits_from_direction`: normalized mean difference and midpoint threshold.
4. `layer_position_probe_heatmap`: fit on toy rows 0–11, evaluate rows 12–23 at every layer/position; zero-signal cells handled explicitly.
5. `replace_position_in_layer_output`: tensor/tuple hook output, cloned edit, preserved auxiliary tuple fields and negative-index validation.
6. `toy_forward_patch_answer_logits`, with provided `toy_answer_logits`: exact stipulated causal state transfer.
7. `patch_control_summary`, with visible `signed_margin_toward_donor`: per-condition means/flips, strongest individual control, signed target-control contrast.
8. `adjudicate_pythia_result`: combine probe, intervention and text-only evidence into a qualified claim boundary.

Keep the provided dataclasses, validation functions and `toy_ground_truth_signature` visible. The first exercise has immediate shape/label assertions rather
than a named helper test; preserve it. Later groups have immediate tests and expected-output blocks. The final adjudicator has explicit historical boundary
assertions; do not replace these with a generic report-accepted assertion.

The CPU play defaults are layer 2, `final_prompt`, target index 12 and donor index 13, with an opposite-answer guard. Retain comparisons to early final-prompt
and rationale positions, explicit signs/labels, source example text, and the anomaly hunt. The toy intervention is a stipulated causal rule: at the designated
position/layers it overwrites the final readout's answer coordinate with the selected donor state. It is not a trained transformer or a simulation of all later
layers.

## What the Pythia history actually says

Pinned model: `EleutherAI/pythia-70m-deduped`, revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`, float32, CUDA. Answer strings are `' A'` and `' B'`; the
historical report records tokenizer IDs 329 and 378. Prompt generation is deterministic: five training nouns and five disjoint held-out nouns, each crossed with
eight fixed A/B condition rows, producing forty prompts in each split. There are four conditions, ten held-out prompts per condition, not forty independent
reasoning problems drawn from an open population.

Private/rationale/final semantic positions are token-offset mapped. With no rationale, its semantic position aliases the private-answer position. The recorded
held-out grid has shape `[40,7,3,512]`; hidden-state index 6 corresponds to block 5 at the final prompt, not a layer selected by maximizing the held-out
heatmap. The final-token answer uses A/B logits only; no generation occurs. The optional hook uses `try/finally` to remove a forward hook and then reads the
continued forward pass's final logits, preserving the difference from merely applying an LM head to an edited vector.

The historical report, dated `2026-09-04T08:49:36+00:00`, records the negative result: hidden-answer accuracy about .775, shuffled-label accuracy about .225,
model answer accuracy .5; text-only recall 1.0 versus detector recall .55; equal target and named random-donor mean effects about .6445; target-control gap 0.0;
`preflight_passed=false`, `real_model_claim_supported=false`, `negative_result_detected=true`. These are **reported historical values**, not calculations or
confirmations from a current run. Overall report `accepted=true` concerns the exact-toy/negative-result contract and does not turn the failed Pythia claim
positive.

Retain the actual construction details and limitations:

- The label-shuffled condition rolls training labels by one; it is not a newly sampled independent label permutation.
- The text-only detector checks literal prompt markers (`biased note` / `after seeing option`). Its success exploits the constructed template labels; it is not
  a general semantic CoT detector or a hidden-answer discovery.
- Opposite-answer donors match noun/condition. The named random-donor index is deterministically `(i*17+5) % len(examples)`; preserve it as such rather than
  claiming iid donor sampling.
- The random-direction condition uses a local CUDA generator with seed 1234 and norm matching. Text-only intervention logits are unchanged clean logits, a
  separate meaning from the text-only recall detector.
- The irrelevant-position intervention uses target token position zero with the donor's semantic-grid position-zero vector. Do not silently substitute a newly
  chosen “background” token.
- In the toy signature, the named random-donor control uses target/self indices and can differ from the clean final-layer readout because the stipulated source
  layer has a different amplitude. Preserve this negative-control construction and explain it; do not quietly improve it or call it an iid random donor.

Thresholds are not completely harmonized. Notebook adjudication uses probe accuracy ≥.70 and a strict target-control gap >.05 plus detector advantage >0. The
CUDA preflight uses probe accuracy ≥.80 and additional conditions; `patch_control_summary` uses gap ≥its supplied threshold. The lock records still another
contract description alongside a required negative preflight. They agree that the retained run is negative, but they are distinct rules at their boundaries.
**U9.3-02:** preserve these exact original protocols and state the difference; root must explicitly adopt any future harmonization.

## Prototype/helper omissions and hidden implementation boundary

`arena_ext/cot_faithfulness.py` is a separate 7,061-byte report utility layer. Its functions summarize already-supplied logits/detection scores and compare
condition accuracies. It delegates a generic accuracy function to `arena_ext.activation_language`. Its `hidden_answer_patching_report` compares two supplied
logit vectors; it does not install a hook or run a transformer. Its tests use synthetic inputs. Do not import these report wrappers as replacements for the
notebook's visible probe, hook edit, patching or control method.

The legacy 1,029-line `solutions.py` contains nineteen top-level definitions absent from the notebook, including `CoTPromptExample`, prompt builders,
token/semantic-position mapping, hidden-grid capture, real `run_with_residual_patch`, matched donors, real patch-control orchestration and the full Pythia
preflight. Preserve that full source and explicit source links. Classify it as **provided optional execution orchestration**, not student-implemented core
simply because its text was copied into a master. If root wants new real-hook wiring or CoT editing exercises, that is a separate pedagogical/method-scope
choice (**U9.3-03**), not accomplished by migration formatting.

Three shared definitions intentionally differ between notebook and module: `_require_binary_ids`, `run_smoke_test`, `run_gpu_test`. Notebook-only definitions
include `RealModelVerdict`, `adjudicate_pythia_result` and plotting. In particular, the notebook's GPU wrapper calls the historical
`reference_solutions.run_gpu_test`; the module's wrapper calls its full local preflight. Do not complete or replace notebook functions from the module by
matching names alone. Use each exercise's own answer and preserve its actual code AST.

## Preservation, import safety and proposed route

The new source inventory hashes **21 files**: all twelve legacy-tree files, historical page/asset, both external report-input files,
`arena_ext/cot_faithfulness.py`, its `activation_language.py` dependency, the helper's root test, the asset-generation script and the syllabus source. Current
historical report closure is **14/14 matching files**. This is source agreement, not fresh CUDA evidence or independently reproduced activations. No dedicated
full own-answer/fresh-process native receipts or numerical execution archives were found in the inspected source tree; retain that distinction from the much
stronger 16.11/16.5 archives.

After GO, first recheck all 21 identities and create a strict flat source/publication review bundle using `scripts/evidence_bundles.py`, with no validator
changes, file omissions or native cache reads. All current members have accepted source/JSON/notebook/PNG formats and no nested archive is present. Preserve
native revision/dependency declarations and complete report bytes without inventing raw activations or weight availability. Any later external archive discovery
must be an explicit hash-bound additional identity, never a guessed path claim. Keep canonical teaching artifacts below 5MiB; strict sealed bundles use the
established separate validator bounds.

Then construct the canonical master from the complete 47-cell learner/solution source, preserving all fifteen code cells and fourteen actual historical output
records. Preserve every notebook/cell metadata field and original ID as namespaced provenance; currently there is no widget graph. Keep real-model source
separately visible/linked without falsely calling its nineteen absent definitions student exercises. Retain all original reading links, live-code paths and
disabled GPU default. Qualification of historical helper routes, heading normalization and equivalent difficulty/importance metadata must be separately
declared/invertible; prose/solution placement must not hide methods or drop caveats.

Import-safety lessons from current migrated sections matter here:

- Audit every default, decorator, class base and annotation. This notebook has no future directive, and `AnswerLabel`, `CoTCondition`, dataclasses and
  Torch/type names must exist when definitions execute. Do not put required annotation/type-alias bindings under `MAIN` without postponed annotations or exact
  named adaptation. Numeric/default arguments are literals in the current code; record the source-only audit rather than assuming defaults are safe.
- The notebook assigns `MAIN` itself; preserve that notebook assignment and use a Python-only bootstrap for safe module import. Guard
  calculations/plots/assertions/printing; do not evaluate CUDA availability, toy tensor construction or report work on import.
- Qualify unqualified legacy `part2_cot_faithfulness` routes explicitly; canonical 9.2 now names a different curriculum. Preserve historical model/noun/source
  IDs rather than globally replacing section numbers.
- Semantic tests use lazy `_solutions()` pointing to the old module. Keep tests one-way; redirecting it to canonical `solutions.py` can create a reference cycle
  or default to the implementation under test.
- Crucially, the notebook's public `run_gpu_test` references `reference_solutions`. Unlike 16.5's test-only aliases, this alias cannot simply be put under
  `MAIN` and left unbound after import. Keep a safe direct historical module alias if audited, or use a Python-only explicit lazy import/proxy inside that
  wrapper while preserving notebook imports/source/defaults. Its root-discovery/current-working-directory assumptions also require review. No wrapper should
  silently call itself or lose access to the original runner.
- Preserve `RUN_GPU_SIGNATURE=False`, its CUDA short-circuit, and the `globals()` gate for qualitative output. A build must carry the recorded skip messages,
  not flip the flag to obtain a more impressive figure.

Use the 16.11/16.5 pattern of track/code ordinal source mapping, exact inverse scientific AST proof, own-answer reconstruction and actual-output checks. Also
carry the lessons of the 16.5 import failure: static compilation is insufficient; a separately authorized inert-import guard must catch eager test fixtures,
definitions needing guarded constants and transitive side effects. Never run that import or fixtures under this preflight authorization.

## History-only prediction opportunities

Add meaningful commitments before probe/patch figures, the negative adjudication and the toy play. Ask about one specific layer-position or donor contrast, give
confidence and a falsifying control, then compare observations against that committed forecast. Count twelve fit/toy evaluation rows, six selected A-target
donor pairs, forty historical real prompts, ten prompts per condition and repeated layer/position/control views separately. New plots do not create new
examples; the original noun-template split remains historical exposed data.

A future strict standard case can use the **already reported** Pythia `patch_target_control_gap` as a numeric target or its predeclared claim-supported verdict
as a binary target. Freeze the exact report/source hashes, model revision/token strings, all forty held-out prompts/template split, layer/block/position
conventions, donor rules, control set and verdict policy in string-valued configuration fields. Do not place the reported target/gap, verdict or learner
observation into the case question/config as an answer. The observation must be an explicitly labeled historical-record reading after forecast commitment, not a
recomputed GPU result or a fabricated exact array receipt.

One binary transfer opportunity is another already retained control contrast on those same forty prompts, declared before revealing its report entry. Keep
method/control identity and denominator fixed and explicitly label the reuse exposed guided practice. Do not calculate the contrast or prefill its answer while
preparing the proposal. Another non-numeric reflection asks which missing control would be necessary for a broader reasoning claim; this is design reasoning,
not empirical transfer evidence.

Use `validate_case`'s exact existing schema if root later adopts a proposal: `recorded_real_run`, `data_role=historical`, `status=guided_practice` for the
CUDA-history case; or `analytic_toy` for a purely stipulated toy contrast. Choose one unambiguous target and do not conflate the two evidence classes. The full
queue can retain both toy and negative-history commitments, but source hashes, `accepted=true`, copied report helpers or an exposed expected figure do not
establish unseen mastery.

## Root decisions left explicit

- **U9.3-01:** adopt historical complete migration scope with expanded-brief gaps disclosed; separately authorize any instruct/arithmetic/CoT-editing study
  later.
- **U9.3-02:** preserve divergent original threshold/control constructions; harmonization or better null designs require an explicit new protocol, not silent
  repair.
- **U9.3-03:** preserve optional orchestration as provided source, with honest visibility; separately choose whether new real-hook student exercises belong in
  scope.
- **U9.3-04:** visually reconcile the byte-distinct inline and saved toy panels; preserve all originals regardless of chosen canonical expected view.
- **U9.3-05:** retain history-only negative Pythia evidence, skipped notebook path and unknown cache availability; neither regeneration nor a new claim is
  authorized here.

No question or pending decision blocks this read-only plan. No stage, commit or push occurred. Root can decide implementation after its full failure review; the
plan itself accepts neither full brief coverage nor scientific/learner readiness.
