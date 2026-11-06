# Additive blind circuit prediction, 2026-10-15

Status: engineering implementation independently reviewed; exploratory diagnostic; human learner
walkthrough and scientific acceptance deferred. This implements the first proposed
blind local challenge from `2026-10-15-integrated-prediction-learning.md` without
changing the guided worksheet or its schema.

Current learner failure: reference 8.8 exposes the outcomes before the learner can
make a prediction; the existing sharpness transfer has already been answered.
Target question: numerical Δ planted-edge ROC AUC for EAP when only unembedding gain
changes 10 → 1, holding sharpness 20, attention noise σ=.03, architecture, sample,
noise seed, corruption and mean answer-logprob metric fixed.
Signature result: a paired AUC plot for exact patching/EAP/IG10, a no-noise gain1
control, signed planted-edge heatmap, and clean-task accuracy/answer logprob.
Student methods: select/aggregate compiler, induction program compiler, seeded
attention noise, exact patching, EAP, midpoint edge-mask IG10, absolute tie-aware
AUC, and result assembly. The notebook shows editable multiline definitions and
immediate independent fixture tests. Shared local 8.8 utilities supply only the
empty model, synthetic data and generic edge-forward graph.
Controls: no-noise gain1, paired noisy gain10, random ranking expectation .5.
Play cell: inspect a chosen stored edge across methods/conditions, no model rerun.
Files: new circuit_prediction unit, generator, scoped tests, and this design note.
Checks: cell syntax/schema/no-prefills, independent core-math fixtures, record
separation, stale identity rejection, replay isolation, private exclusive export,
and generator preservation. One authorized fresh CPU validation is separate.
Original ARENA chapters, existing 8.8 source, solutions, utilities and instruction
pages are preserved. No pretrained loading, GPT-2, training, GPU or cluster jobs.

## Settled design and bounds

- Frozen case: two layers, three heads/layer, d_head17, d_model81, vocab16,
  17 positions, 64 prompts, data seed0, noise seed0, float32 CPU, one thread.
- Metric averages next-token logprob over all 64×7 predictable positions;
  task accuracy uses precisely the same denominator and clean targets.
- EAP uses the keep=1 derivative. IG10 uses ten midpoint masks on the all-edge
  straight-line mask path. Exact uses all-valid minus one removed edge.
- Seven edges label the planted program. After attention-weight noise these
  are planted structural labels, not exhaustive causal ground truth.
- Schema `arena-blind-circuit-prediction`, version1, immutable sealed JSON text.
  No conversion from guided `learner_reported` or quadratic records.
- Identity binds full case, exact method text, helper and plumbing hashes,
  environment (Python/torch/NumPy/TransformerLens/platform/dtype/threads),
  ordered model-state names/dtypes/shapes/bytes, clean/corrupt tensors, valid
  and planted masks, and sender/receiver ordering.
- Freeze compiles weights without a forward pass. Explicit fresh run rebuilds
  and compares identities before inference. Commit/reveal/replay/export do no
  model work. Replay only reads stored values and can redraw saved figures.
- Exposure labels: `unknown`, `seen`, `self_reported_not_seen`. No automatic
  unseen badge. Seals detect accidental edits, not adversarial local changes.
- Student source/outputs/records are protected from generator overwrite.
  Export is opt-in, O_EXCL, mode0600. No automatic persistent learner record.
- No direct-vs-upstream normalized error metric is added; its denominator
  would need a separate frozen contract.

## Bounded validation forecast, entered before target inference

Engineering validation forecast: **Δ EAP AUC = +0.20**, confidence **0.60**.
Causal reason: lowering output gain can change log-softmax endpoint gradients
relative to finite edge effects; fixed high attention sharpness may preserve an
upstream limitation, so improvement in ranking could be small or absent. This
forecast is an engineering test attempt, not a student answer or prefilling.
The new gain1 outcome has not been inspected before writing this paragraph.
Allowed aggregate probe: one offline fresh CPU validation, one thread,
walltime ≤120 seconds, no downloads/training/GPU/cluster. Results, if obtained,
will be recorded separately after this forecast. The notebook is shipped blank.

## Open/deferred review queue

- BC1: independent source-span/semantic leak and scoring review (closed; findings repaired).
- BC2: app download/navigation integration and private record round-trip (implemented).
  The circuit schema stays separate from the quadratic app importer.
- BC3: actual human learner walkthrough (deferred by user).
- BC4: scientific adoption/claims beyond this fixed exploratory diagnostic
  (not authorized by engineering checks).

## Engineering checkpoint after review

The scoped suite now has 38 passing tests, including complete mock compiler/noise
fixtures, strict record validation, malformed/duplicate schema rejection, final
scorer overrun rejection, and reflection/transfer archive round-trip. The generated
notebook is current and blank; every code cell compiles. Two independent reviewers
closed their P2 findings (records/deadlines; learner/scientific/leakage) without
running or inspecting target outcomes. Their earlier focused runs had 34 tests;
four additional tests cover exact generated method-text execution, deadline
rejection, and fixture behavior under float32/float64 default dtypes. No further target inference was run.

The single authorized fresh validation completed in 6.34s including setup, with
2.31s attribution execution. It used Python3.14.2, torch2.12.1+cu132 on CPU,
NumPy1.26.4, TransformerLens2.16.1, float32 and one thread. Original immutable
forecast/receipt/log/plot are preserved under
`docs/mastery/evidence/blind_circuit_validation/` with their original filenames and
a hash manifest. These internal records contain spoilers. The forecast
missed; the negative/null direction is preserved. No target output is prefilled
into learner cells or metadata.

The measured helper SHA256 was
`43089571f1e91a66c120a208cfabdc62e403a2f747ecadd05f56f2f1d4bef5b0`.
The final helper SHA256 after the one-line default-dtype fixture repair is
`623c48f6d0eb51c3cf2919c1b0d4ca5a2f0eeb4b81b4e6a59828c81d48af2d2f`.
Subsequent changes hardened records/deadlines, expanded independent fixtures and
added the separate reflection archive; benchmark compiler/scoring/reference
method text was unchanged. Thus fresh evidence belongs to the measured helper
identity, while final-source evidence consists of scoped tests, independent
review, and successful model-free validation/replotting of that exact receipt.
This is not a final-source fresh-model rerun or human acceptance claim.

The reflection envelope (`arena-blind-circuit-prediction-reflection`, version1)
contains the unchanged sealed receipt, its SHA256, mismatch explanation, changed
question, new prediction, confidence and causal reason. `reflect`, `read_attempt`
and explicit private `export_attempt` preserve this separately from the original
measurement; `replay`/`plot_receipt` continue to operate on the nested receipt.
The CPU deadline is cooperative between stages and includes preparation; a slow
individual student function is rejected upon return, not forcibly interrupted.
The engineering validation additionally used an external hard120s timeout.

## Authorized final-helper known-case regression

After the checkpoint above, the parent explicitly authorized exactly one additional
offline CPU regression of the final helper, with the same case/method/cohort/dtype
and ≤120s external timeout. This was a **known-outcome engineering regression**:
`prior_exposure='seen'`, the numeric field copied the original measured delta,
and the rationale explicitly disclaimed an independent forecast, new scientific
probe, selection, or evaluation. The original first forecast and receipt were
not rewritten; all original evidence-file hashes were checked before and after.

The regression passed on final helper SHA256
`623c48f6d0eb51c3cf2919c1b0d4ca5a2f0eeb4b81b4e6a59828c81d48af2d2f`.
It completed in 5.75s including setup and archival checks; the run itself took
1.96s. Every condition's score matrix, AUC, accuracy and answer logprob matched
the original receipt exactly. The machine-produced receipt passed the final
strict validator. The separate reflection/transfer envelope exported and read
back with the original nested receipt bytes intact. Stored-value replay and
plotting succeeded after all helper model-execution entry points were replaced
with functions that raise on invocation. No helper source changes were needed.

Evidence is preserved separately in
`docs/mastery/evidence/blind_circuit_validation/known-case-final-helper-*`,
including commitment, receipt, reflection attempt, source-hash receipt, exact
helper snapshot, plot, log and regression manifest. The runner is
`known_case_regression_runner.py`. This adds final-source execution/record-path
regression evidence; it does not replace the original blind forecast, create a
second independent prediction, establish generalization, or constitute human
acceptance. The learner notebook remains blank and contains no outcome prefills.

## Subsequent runtime binding correction (current schema v2)

A cross-unit review found a binding defect: an incorrectly implemented scorer
could mutate clean/corrupt data, edge masks, or model weights after the initial
identity check, and an assembler could return an internally consistent table
unrelated to the actual returned scores. These are ordinary execution-integrity
failures; no adversarial assessment security is claimed.

Current execution snapshots and rechecks all conditions' ordered model states,
clean/corrupt tensors, sender caches, graph/model identity, graph sender/receiver
order, valid/planted masks and graph dimensions around student calls. It also
checks runtime dtype/default device/thread/deterministic settings and ordered
module configuration/training metadata. The setup explicitly selects the CPU
default device, and preparation rejects a non-CPU default before compilation.
Captured scores and clean logits are detached/cloned; assembly receives disposable
copies. Its score matrices and task behavior must match the actual captured
measurements, while the existing strict receipt validator still checks AUC and
all derived values. Completed condition tables are copied so later calls cannot
mutate them through shared aliases. The scientific compiler/scoring/reference
method text is byte-for-byte unchanged from the preserved earlier snapshot.

This adds explicit **schema version2** commitments. The v2 identity additionally
binds ordered per-module forward configuration and training metadata. Historical
v1 records retain their original exact schema validation and sealed-parent checks;
they can be replayed/exported and nested in the existing reflection archive, but
cannot start a fresh run. They are never upgraded in place. A new execution needs
a newly frozen v2 commitment. The reflection wrapper remains its separate version1
format because its unchanged payload is a sealed receipt string plus reflection.

The focused suite has 62 passing tests, including non-target stand-ins reproducing
input/cache/weight/mask/order mutations, forward-config/training/runtime-default
changes, fabricated scores/behavior, and shared result aliases. The generated
notebook remains blank and current; both preserved actual v1 receipts validate
and replay through the corrected helper. Current helper SHA256 is
`31d924d09e6e941fdb83a4e7b362d68ae8a06c79543c295b14c18f53c70704dc`.
Independent re-review is pending at this checkpoint. No target inference was run
for this correction, so the earlier final-helper known-case regression belongs
to its preserved earlier source hash; it is not fresh execution evidence for v2.
Original scientific/engineering receipts and the first forecast remain unchanged.

### Preserved failed v2 regression and narrow serialization repair

Independent review cleared the runtime-binding correction at helper
`31d924d09e6e941fdb83a4e7b362d68ae8a06c79543c295b14c18f53c70704dc`.
The parent then authorized one bounded known-case v2 regression. That attempt
failed during freezing, before commitment and before any target forward, because
TransformerLens represents derived `attn_scale` and `initializer_range` as
`numpy.float64` values. The configuration serializer had only accepted native
Python numeric values and torch dtype/device objects.

The failure is preserved under `known-case-v2-final-helper-*`, including the log,
source/test/notebook snapshots and failure manifest. No retry was performed.
A configuration-only diagnosis identified both actual types. The narrow repair
encodes a NumPy scalar with its explicit module/type tag and recursively checked
native value; nonfinite values remain rejected. No string/repr fallback was added.
The focused suite now has 64 passing tests. Its new real compiled-fixture freeze
test replaces `HookedTransformer.forward` with a function that raises, then
constructs the actual three model configurations/weight states, freezes their
metadata and validates a v2 commitment. Thus the actual serialization boundary
is covered without target inference; the additional scalar test checks finite
values and preservation of NumPy-versus-native type identity.

The repaired helper SHA256 is
`514aa1468f6aa205f6bfd96d246b0f7472421922d8eb510a451414c988dce0bb`.
Narrow independent re-review and any renewed known-case inference authorization
are pending at this checkpoint. Scientific method text and original receipts
remain unchanged; the learner notebook remains blank.

### Repaired v2 final-helper execution checkpoint

Narrow independent review cleared the NumPy scalar repair at helper
`514aa1468f6aa205f6bfd96d246b0f7472421922d8eb510a451414c988dce0bb`,
including the real compiled-fixture freeze with every model forward forbidden.
The parent renewed authorization for exactly one ≤120s offline CPU known-case
regression. This separately named attempt **passed**, with no retry inside the
attempt and no helper edits after execution.

Evidence prefix: `known-case-v2-repaired-helper-*` in the same internal evidence
directory. The helper, tests and blank notebook were snapshotted before the v2
commitment. The run used `prior_exposure='seen'`; the numeric field copied the
original measured delta and explicitly identified a known-outcome engineering
regression. It took 5.90s including setup/archive checks and 2.45s within the
execution function. All original condition score arrays, AUCs and task metrics
matched exactly. The current v2 machine receipt passed strict validation with
frozen ordered model configuration and runtime default-device identity. The
reflection envelope and nested receipt survived private mode0600 export/readback;
replay/plotting succeeded with model execution entry points replaced by functions
that raise. Hashes of all preceding evidence files, including the preserved failed
attempt, remained unchanged.

The new receipt SHA256 is
`db110965548ee9af8f977716a6875d50460274c1c1a70617d6fd0ad36f8c355b`.
The regression manifest binds the source snapshots, commitment, receipt,
reflection, plot and source-hash record. Historical v1 replay and current v2
execution now have separate evidence at their actual helper identities. This
remains exploratory known-case engineering evidence, with no new independent
prediction, scientific adoption, or human learner acceptance claim. The learner
notebook remains blank; final helper source is unchanged from the reviewed hash.
