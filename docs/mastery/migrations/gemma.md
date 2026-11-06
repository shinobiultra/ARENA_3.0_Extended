# 5.1 Gemma master migration

## Plan before edits

- **Current learner failure:** the existing 34-cell Gemma lesson already has
  progressive inline implementations, immediate checks, a visible parity plot,
  three architecture controls and a token-ID play. It bypasses the canonical
  master pipeline and lacks explicit forecast/reflection and provenance labels.
- **Target section claim:** preserve the existing deterministic, random-weight
  Gemma-style decoder parity lesson. Do not imply pretrained Gemma 3 or Llama
  validation, language ability, or coverage of later-generation architectural
  differences named in the broader syllabus.
- **Signature result:** retain the full historical layer-by-layer exact parity
  and final-logit error plots, including all three failing architecture controls,
  the changed-input play plot, and the public Hugging Face tiny-reference output.
- **Student functions:** retain every RMSNorm, RoPE, grouped-query replication,
  causal/sliding-window mask, SwiGLU, attention/cache, decoder, causal LM, trace,
  comparison and first-divergent-stage exercise and its worked answer.
- **Controls/baselines:** exact local reference; disabled embedding scaling;
  disabled RoPE; incorrect GQA head order; historical Hugging Face random-weight
  parity. A new exact CPU mask play contrasts absolute cache positions with the
  wrong zero-origin mask and a global-attention baseline.
- **Play cells:** retain the original token-ID investigation and anomaly hunt;
  add a small cache-position/window mask play with forecast, confidence,
  exposure, observation, causal reflection and changed-condition prediction.
- **Files:** master 5.1; its generated notebooks, solution module and page;
  section-local media; byte-identical old solution module; focused migration
  tests; this note; archive, parity, CPU-subset and proposal evidence. Shared
  syllabus/config/case/checkpoint files remain with root.
- **Tests:** exact archive bytes/hashes; historical source AST/nonblank-line and
  all output/count parity with named setup/release routing adaptations; prose
  preservation; exercise answer completion; import safety; bounded CPU primitive
  checks and mask play. No model downloads, GPU runs, training or full fresh
  notebook execution. New cells remain unexecuted unless separately measured.

The current canonical directory is the full legacy source, not a second
truncated copy. Before overwriting generated files, preserve the entire source
directory, matching page/assets, and declared Gemma/reference dependencies in
a validated review archive. Preserve existing report and fixture bytes even
when migration makes their old source hashes stale. Archive packaging is not
scientific or learner acceptance. Original ARENA is unchanged.

## Scope left open

The syllabus requests loading `google/gemma-3-270m` and a SmolLM2 checkpoint,
local/global layer interleaving, additional activation/soft-capping conventions,
and attention-distance/sink analysis. Those are not demonstrated by the current
Gemma-style tiny random-weight lesson. This migration neither adds those model
families nor promotes these missing claims to covered. The original historical
reference evidence and any new exact CPU primitive checks are distinct layers.

## Additive convention exercise authorized after static review

Independent review found that the preserved learner uses SiLU and interleaved
RoPE, while the saved HF diagnostic uses public Gemma defaults: tanh-approximate
GELU and split-half rotary coordinates. Identical parameter names and small
random-weight errors do not establish equivalent architecture. The old code,
warnings, numbers and outputs remain unchanged, with an adjacent qualification.

Root authorized one bounded CPU comparison to teach the missing reconciliation.
The added exercises implement a per-head Q/K projection row permutation and
the GELU(tanh) gate. Their independent immediate checks include dimension-eight
head-marked rows, an explicit scalar GELU formula, and relative-position rotary
scores whose correct and incorrect results were derived independently before
execution. V/O projections remain unchanged; attention bias is fixed to false.

The fixed comparison uses seed 5101, the original four seven-token inputs,
vocabulary 31, width 16, MLP width 32, two layers, four query/two KV heads, head
dimension four, default HF initialization standard deviation .02, eager float32
CPU attention, one thread, and no cache. One HF forward plus four learner
forwards compare unmatched, activation-only, basis-only, and both-correct
variants. The corrected gates remain maximum absolute error <=5e-6 and mean
squared error <=1e-12 over 868 logits. No seed search, stress initialization,
tolerance relaxation, training, pretrained load, download, or GPU is authorized.
The full negative variants are reported numerically; component controls prove
the convention errors even if a tiny full-model discrepancy passes a tolerance.

The exact protocol, runtime, source hashes, independent pre-run review and any
failed attempt belong in `../evidence/gemma_master_migration/`. This increment
does not alter the historical 16-cell evidence identity or establish full Gemma3,
SmolLM2, cache, CUDA, or learner acceptance. The earlier plan's no-model-execution
boundary applies to the preservation migration; this later explicit allowance
adds only the stated tiny CPU comparison.

## Recorded result and final learner surface

The single authorized comparison completed in 5.449 seconds. Python 3.14.2,
torch 2.12.1+cu132 and transformers 5.16.1 ran one HF and four learner forwards
on CPU, with both thread pools fixed to one and CUDA uninitialized. The CPU
build includes CUDA support, but no GPU execution occurred. All four variants
used one copied reference state, not separately sampled weights.

| Variant | Maximum absolute error | MSE | Ordered top-five agreement |
| --- | ---: | ---: | ---: |
| Unmatched conventions | 9.009242057800293e-5 | 6.451746914493128e-10 | 1.0 |
| GELU only | 8.448585867881775e-5 | 4.604863157453565e-10 | 1.0 |
| Q/K basis only | 4.9602240324020386e-5 | 1.73853598184337e-10 | 1.0 |
| Both conventions matched | 1.862645149230957e-9 | 3.9970586966848386e-21 | 1.0 |

Only the corrected variant passes both frozen gates. The full-model negative
variants were not required to fail by the protocol; their observed failures
support this particular configuration. The independently derived component
controls establish the convention differences directly. Perfect top-five
agreement in all four rows demonstrates why that criterion alone is weak.

The final lesson has 21 code cells, 17 student stubs, 18 forecast/reflection
pairs, balanced help/solution dropdowns, all original progressive methods,
the original token-ID play and anomaly hunt, a fresh cache-position mask play,
and the two new convention exercises. All 16 historical code cells and their
18 saved output objects/counts are preserved. The only original code routing
changes are the section-local media directory and the release module's import
of the byte-identical `legacy_solutions.py`; whitespace/student-answer markers
are checked by AST and nonblank-line comparisons. The existing missing outer
Expected-output closing tag was repaired additively. All original prose remains
available, with explicit adjacent qualifications where its reference claims
are too strong.

The independent reviewer checked the archive, methods, scalar/basis oracles,
exact frozen source and protocol before execution, actual receipt afterwards,
and all three types of plot. Independent review is not a learner walkthrough.
The proposed hub case is `../evidence/gemma_master_migration/proposed_case.json`;
root owns integration. Its numeric forecast targets corrected maximum error;
its binary transfer uses the already recorded GELU-only row. Both remain
exposed guided practice, with explicit prior-exposure reporting and no unseen
assessment claim.

## Evidence identities and freshness

- `published_before_master.tar.gz` holds all 18 archived source/asset members,
  with a validated review-bundle sidecar and exact hashes in `before.json`.
  All 16 report input hashes matched physical files before migration. The old
  report, lock, fixtures, tests and utility files retain their old bytes. This
  archive establishes what the historical report referred to, not fresh
  scientific acceptance of changed generated files.
- `output_parity.json` refers only to the 16 retained historical code cells.
  Its `fresh_model_execution: false` describes that historical carry operation.
  The historical HF output retains transformers 5.12.1 and its original
  tolerated discrepancy, warnings and code. It is not replaced by the new
  transformers 5.16.1 result.
- `cpu_subset.json` records a separate seven-cell kernel: guards, five selected
  canonical primitive cells, and verification. It checked 24 mask cases without
  constructing a model. Only the new mask-play output/count was inserted into
  the canonical lesson; the original selected primitive outputs remain
  historical. Its source receipt binds the exact selected code, not the later
  expanded master as a whole.
- `convention_protocol.json`, `convention_pre_run_review.json`, immutable
  `convention_attempt.json`, raw output, and the six-cell execution notebook
  bind the one authorized tiny-model run. New canonical code-cell counters
  2–5 belong to that separate kernel. The serialized HF config has its default
  `use_cache: true`; every actual forward explicitly overrides it with
  `use_cache=False`, as frozen in the protocol. This is not a cache test.
- The full pre-run master/notebook bytes were recovered after execution by
  undoing only additive result prose and output/provenance metadata, and were
  verified against the already frozen SHA-256 hashes. They are stored as
  `convention_frozen_master.py` and `convention_frozen_canonical_input.ipynb.gz`.
  They are hash-verified recoveries, not falsely labeled contemporaneous copies.
- After execution, new result/reflection prose and scoped provenance labels
  were added. The notebook now correctly declares fresh model execution only
  for the separate subset, and explicitly denies whole-notebook freshness.
  None of the four measured canonical source cells, their actual outputs,
  frozen protocol, runner, results or old evidence was changed. `after.json`
  distinguishes current full-file hashes from frozen executed-cell hashes.

The one-run budget has been consumed; no second model run is part of final
validation. The focused tests inspect preservation, source/output bindings,
import safety, pure tensor mask/permutation checks and receipts. Current
generation and test commands, including the original hard-capped execution
command, are recorded in the evidence directory's README. Pretrained coverage,
broader syllabus methods, full fresh execution, original report freshness,
and the deferred human walkthrough remain open.

## October 19 local reference compatibility repair

The canonical setup now names `part1_gemma_from_scratch/gemma_reference.py`, an
exact byte copy of the preserved `arena_ext/gemma.py`. This is explicit baseline,
configuration and output-type plumbing. Student Gemma implementations remain
inline and unchanged. The separately stored reference substantially duplicates
those methods; moving its import does not establish architectural independence.
The original external reference and `legacy_solutions.py` remain byte-identical.

`evidence/gemma_reference_route_20261019/` freezes the pre-edit artifacts and pins
both exact setup-source hashes. Its named bridge changes only one `ImportFrom`
module route at code ordinal zero, retaining imported names and aliases. No
numeric receipt or original protocol is rewritten. Saved October16 outputs retain
the original setup execution identity; the new import route has not been executed
as a numerical study. The static carry preserves every output, count and widget;
only the setup source changes, so full setup code-object equality is not claimed.

Root owns no-execute regeneration and publication after independent review.
The exact local reference module must be an explicit plumbing allowance in the
section shape contract. Human learner acceptance and broader model claims remain
open. No new student result, control or answer field is introduced by this repair.
