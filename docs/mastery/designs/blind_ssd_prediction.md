# Blind SSD state-carry prediction companion

## Plan before implementation

**Current learner failure:** canonical 5.4 exposes rich historical SSD, trained recall, checkpoint and CUDA results. It provides guided practice rather than a clean numerical commitment on an unmeasured fixed case.

**Target question:** how much does weakening past-state carry change a fixed filter's final signed readout when writes and read directions compete? Learner title: **One sequence, three equivalent computations**.

**Signature result:** after commitment, a signed final-readout difference curve across carry strengths and time-by-head/state heatmaps. No monotonicity is assumed.

**Student implementations:** literal recurrence, full quadratic causal operator, chunked diagonal/off-diagonal SSD with boundary-state carry, and readout/result assembly. Immediate independent tests and parameter play use different practice inputs.

**Controls:** independent scalar oracle; three-form output/final-state parity; chunk sizes 1/3/5/12/17; causal prefix/future perturbation; zero writes; gamma-zero
current-write-only behavior; split/state-cache composition and nonzero initial state; linearity; coherent state-basis and head/readout permutations. Public-case controls run only
after commitment. Standalone test/play fixtures are distinct.

**Files:** only new `mastery_hub/units/ssd_prediction/`, `scripts/build_ssd_prediction_notebook.py`, `tests/mastery/test_ssd_prediction.py`, this design and scoped evidence. Root owns app/docs integration and shared registries. Preserve all existing lessons and blind units.

**Validation:** non-target hand-worked fixtures, strict schema/identity/alias/mutation/fabrication/failure/export/replay tests, generator preservation, independent source-level and pre-run reviews. No public-case evaluations until root records engineering forecasts and dispatches the reserved aggregate 120-second CPU budget for exactly one primary plus one separately committed transfer.

## Exact public case fixed before measurement

Batch 1, sequence length 12, heads 2, head dimension 1, state dimension 2, one shared B/C group, zero initial state. CPU float64, one thread. Constant log transition `A=log([.82,.94])` broadcasts over time. Raw transition probabilities are frozen inputs; import/freeze need not evaluate logarithms.

`X[t,head]`:

```text
[[1,-.25],[-.5,.8],[0,-.6],[.75,0],[-1,.5],[.25,-1],
 [0,.3],[-.25,0],[.5,-.4],[0,.7],[-.75,0],[.4,-.2]]
```

`B[t,state]`:

```text
[[1,0],[0,1],[1,-1],[.5,1],[-1,.5],[1,1],
 [0,-1],[-.5,1],[1,.25],[.25,-1],[-1,-.5],[.5,.75]]
```

`C[t,state]`:

```text
[[.5,1],[1,-.5],[-.5,.25],[1,1],[.25,-1],[-1,.5],
 [1,-.25],[.5,.5],[-.75,1],[1,.25],[.25,-.5],[.8,-.6]]
```

Read-after-write recurrence:

```text
state[t,h,p,n] = gamma * exp(A[h]) * state[t-1,h,p,n] + X[t,h,p] * B[t,n]
Y[t,h,p] = sum_n C[t,n] * state[t,h,p,n]
r[t] = .7 * Y[t,0,0] - .4 * Y[t,1,0]
```

The public metric is `r[11](gamma=.65)-r[11](gamma=1)`, a finite signed scalar with no [-1,1] bound. The postcommit curve evaluates gamma `[0,.25,.5,.65,.85,1]`; primary chunk size is 5. No target output has been calculated or selected during design.

Transfer changes only `A -> A/2`, equivalently square-rooting the two raw transition probabilities during execution. Every input, readout, carry strength and metric remains fixed. Transfer has a separate forecast, reflection link and immutable commitment; primary controls never execute it.

Gamma zero is handled explicitly as current-write-only. Quadratic evaluation must avoid `0 * -inf` and retain diagonal weight one. Artificial padding has transition one, zero X/B/C
and discarded readout; it must not decay the final real state or apply gamma to padded positions. Split execution carries the exact terminal state into the next segment. Coherent
head permutation also permutes readout weights. These are equivalence/control requirements, not measured claims about the public target.

## Integrity and scope

Use a separate versioned blind SSD schema, strict finite bounded JSON and duplicate-key rejection, source/helper/tests/runtime/input-byte identities, immutable forecast linkage,
before/after method-boundary checks, detached actual arrays, independently checked assembly, failure retention and exclusive mode0600 exports. Replay/reflection inspect stored
measurements and must never execute student/target methods. Notebook generation preserves learner edits, records, execution counts, outputs and widget metadata. Local seals detect
accidental drift; they do not certify adversarial security or unseen exposure.

This is an exact small fixed linear filter using the same scalar-per-head SSD structure taught in [canonical
5.4](../../../chapter5_modern_architectures/exercises/part4_mamba2_ssd/5.4_Mamba-2_and_State-Space_Duality_exercises.ipynb). It demonstrates state transport and equivalent
computations. It does not reproduce or establish the lesson's historical trained delayed-recall result, released 130M hidden-state/logit parity, CUDA parity or performance. No
training, pretrained loading, download, GPU or cluster work is included.

Current status: root adopted the design after independent source-level review; scoped implementation and non-target checks are complete. Independent implementation/pre-run review and root numerical dispatch remain pending; no target measurement has occurred.

## Adopted source-review invariants before implementation

Independent source-level review approved the exact case without evaluating its arithmetic. Gamma zero is finite current-write-only on real tokens; padding is identity and the final
state is the last real state. The state-basis control uses an explicitly orthogonal swap/sign transform on a distinct fixture, transforming both B and C and the initial state
coherently. Head swaps transform X, log decay, initial state and readout weights together while shared B/C stay fixed. No target arrays may enter pytest, shape-smoke evaluators,
import-time work or precommit play, even for gamma-one baselines. Viewing target curves or play before commitment counts as exposure. No monotonicity claim is made. Root
provisionally adopted implementation after this source review; final numerical dispatch still requires independent pre-run review and root's recorded forecasts.

## Implemented contract and pre-run verification

The blank workbook contains five editable function-source cells: recurrent SSD, quadratic SSD, chunked SSD, readout and JSON-native assembly. Each immediately runs an independent
different-input practice test. Own-answer completion reconstructs every reference method AST. Shared plumbing validates input types/shapes, hashes and records; all numerical
SSD/readout/assembly implementations are visible. The parameter play uses a different three-token fixture. The source-level design review is recorded at
`docs/mastery/evidence/blind_ssd_design_review_20261016.json`.

The committed runner has fourteen conditions: six carry strengths; four additional gamma-one chunk sizes; zero writes; a future-token perturbation; doubled writes; and a coherent
head/readout swap. Each stores actual recurrent, quadratic and chunked outputs, full real-token state histories and readouts, plus an independent scalar-loop oracle's corresponding
arrays and byte hashes. Assembly receives disposable clones, and its arrays must equal the detached measurements. A valid returned partner array survives even if another returned
array is malformed/nonfinite. Finite incorrect assembly arrays remain unscored diagnostic evidence. Structured failures preserve the immutable commitment, completed arrays,
stage/error and failed controls; they never contain a successful forecast score.

All runtime/input/source checks bracket each student call. Freeze binds raw public arrays, float64 little-endian C order, ordered shapes, method/helper/test hashes, CPU runtime and
one thread; no logarithm, SSD, readout or public-case control is evaluated by import/freeze/commit. Transfer preserves the primary method/helper/test/runtime identities and changes
only log-decay scale from1 to.5. A source/runtime edit requires a new primary commitment, not a silently changed transfer.

Portable readers validate stored array hashes and adjacent-state/readout residuals, equivalence controls, forecast scoring and parent links. They perform arithmetic validation of
saved measurements; they do not call a numerical method, scalar-loop oracle, current runtime or fresh experiment. Reflection and plotting use those stored values. Tests replace all
numerical entrypoints with raising stubs during replay/reflection/export/plot checks. State heatmaps share one signed color scale; effect curves have no imposed monotonic trend.

All records use separate `arena-blind-ssd-prediction` v1 / reflection schemas. Readers reject unknown/duplicate fields, nonfinite numbers, numeric booleans, malformed arrays, stale
layouts/identities and broken links. Accepted writers validate through their reader; a largest permitted Unicode-source primary and transfer lifecycle roundtrip is tested. Exports
are private exclusive mode0600 files. The generator is deterministic and preserves learner edits, counts, outputs, widgets, execution metadata and raw legacy trusted state.
`--check` is read-only.

Pre-run checks:

```bash
UV_CACHE_DIR=/tmp/arena-mastery-uv MPLCONFIGDIR=/tmp/arena-ssd-prediction/mpl \
CUDA_VISIBLE_DEVICES='' uv run --no-sync --offline pytest -q tests/mastery/test_ssd_prediction.py
UV_CACHE_DIR=/tmp/arena-mastery-uv uv run --no-sync --offline \
python scripts/build_ssd_prediction_notebook.py --check
```

Current result: **97 passed in6.44seconds**; generator check is recorded separately. Tests use hand-worked one-state examples, different three-token practice arrays and a distinct
zero-input diagnostic case. An autouse dispatch guard rejects the public case in every test. Public input assertions and freeze/commit tests inspect raw values/hashes only. Initial
fixture failure and diagnosis are retained in scoped implementation evidence: nbformat serialization removed the test's trusted flag before the generator could inspect it; writing
that legacy flag into raw JSON fixed the fixture without weakening generator protection.

The target and transfer remain uncomputed. The 120-second aggregate CPU allowance remains reserved, not dispatched. Independent pre-run implementation review and root's immutable forecasts/go-ahead are still required. The shipped learner notebook has no outputs, execution counts, forecast, observation or target plot. No other blind unit, existing lesson, shared app, registry or config was edited.

Root static review identified upper-triangle exponent overflow before causal masking for strong negative log decay. The prior88-test/four-file source checkpoint is archived and
member-hash verified at `docs/mastery/evidence/ssd_prediction/before_causal_mask_fix.tar.gz`. Both quadratic and chunked forms now mask future-position log weights to negative
infinity before exponentiation; the gamma-zero branch remains explicit. Nine new distinct four-token regressions cover strongly negative log decays, gamma0/.4/1 and
nondivisible/oversized chunks, with finite output/state parity against recurrence. All97checks pass; the blank generator and its read-only check pass. No public target was
evaluated. Current four-file identities are refrozen in `implementation_checks.json`.
