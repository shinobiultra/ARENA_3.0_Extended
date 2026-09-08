# Independent publication review of 16.16 development v7

Reviewer: `/root/astra_interaction_review` (Astra), 2026-09-08 UTC.

**Verdict: BLOCK PUBLICATION of frozen development v7.** The interaction methods,
learner explanations, and inspected figures are scientifically sound, but one
promised numerical detail of original review finding 6 is still unimplemented.
The current checkout also has a separate stale README binding and the disclosed
post-maintenance environment mismatch. This is an actual independent review
after development execution; it is neither pre-run approval nor approval
conditional on future fixes.

## Actionable findings

### R1 — [P2] Implement the frozen feature-order distance accumulation

`interaction_study.py:1499-1500` computes the support distance with
`delta.square().sum(-1).sqrt()`. The normative `protocol.md:188-192` instead
specifies float64 squared differences and accumulation feature by feature in
index order, followed by the square root and smallest-source-ID exact-tie rule.
The vector reduction does not implement that declared accumulation order.

This is a demonstrated native-payload discrepancy, not a hypothetical portability
concern. An independent explicit eight-step accumulation for explanation source
ID 0 checked all 256 masks and eight support donors against all 758 training
rows. Of the 2,048 retained nearest distances, **360 differ**, with maximum
absolute difference **4.440892098500626e-16**. For mask 2 / support donor index 0,
the retained value is `1.4730388718322383`; the specified sequential result is
`1.4730388718322385`. Nearest source IDs did not change among those 2,048 checked
outputs. I have not claimed that no source ID could change elsewhere.

The magnitude is roundoff and does not undermine the lesson's interaction
claims. It does prevent attesting that all exact frozen arithmetic requirements
were implemented. The existing test at `interaction_study_tests.py:335-351`
checks a duplicate-row zero-distance tie, which does not test accumulation order.

Required correction: explicitly accumulate squared feature differences in index
order; add a regression whose nonzero coordinates distinguish sequential
accumulation from a reduction; retain v7 and record the correction in the
amendment chain. Rerun the complete corrected learner and source-only fresh
reproduction. Do not change the frozen protocol or widen a tolerance to accept
the old arithmetic.

Evidence: `review_probe.py`, `review_probe.json`, and `review_probe.log` in this
directory. The probe includes the exact first eight discrepant coordinates and
1,552,384 underlying candidate-distance comparisons.

### R2 — [P1] Record the current README in the source amendment chain

`run_native.py:202-204` validates every source in the pre-run resolution plus
dated amendments. The current section `README.md` is the sole mismatch in that
chain: expected
`f7628ad83fd1f054d411f0fe9fa186eef348edcd2e11a002f4f309d397d28b61`, current
`c2fde2e74e5b74840f26254755ca831ad73969b2b09ffc5a368d3e4bbe08b14b`.
Consequently `validate_freeze()` raises
`ValueError: Resolved pre-run source changed: README.md` before a new learner
can run. The changed README was committed at `ff6d06e3`; this failure is distinct
from the later dependency update at `ce96ceaa`.

The current README accurately describes v7, so discarding its status information
is not the repair. Add an honest dated documentation/source-binding amendment
and preserve the original freeze, resolution, and old bytes. Then rerun the
unchanged freeze contract test. Evidence: `contracts.log` (19 passed, one failed).

## Current environment boundary

`interaction_reference.py:124-128` correctly rejects the current lockfile. The
current SHA256 is
`e96e946bb00f50592d32c1b3bf453fca4482a0097731120defbebcd978f019eb`; v7 expects
`03f77ffac99858cd34a653cd703728b9a868c87686a86cef78523a7afa9a6494`.
The preserved maintenance lock has the expected old hash. The installed pinned
shapiq sources pass their checks before execution reaches the lock guard;
NumPy is 1.26.4 and Torch is 2.12.1+cu132.

The current section `*_tests.py` run produced **59 passed, one optional full-fit
skip, 13 failures and six fixture errors**. All 19 unsuccessful cases reach the
old-lock guard; none was relabeled as a numerical pass. This run is narrower
than the author's historical 119-test aggregate and is not an attempted
reproduction of that count. See `current_environment_tests.log`.

Current-environment publication therefore also needs an explicit environment
amendment, complete CPU learner execution, actual fresh refit/reproduction,
normal report execution, and integration checks. Reusing v7 receipts under a
new lock label is not valid. This reviewer performed no new model fit, dependency
installation, lock change, publication, or course registration.

## What passed independent inspection

All 22 v7 execution source hashes match both their saved source snapshots and
the current section sources inspected. The completed notebook and all eleven
PNG hashes match `execution.json`. Both existing native and fresh directories
were reopened and their physical artifacts, recursive value identities,
external shards, logical aliases and pre-audit seals revalidated. The previously
executed learner and previously executed fresh refit still match exactly:
**1,797 arrays, 94,604,271 elements, 26,348 records and 22,556 scalar leaves**.
This is live verification of historical files, not a new current-environment
refit.

The prospective contract checker was read and rerun with only its output path
and reviewer fields changed. It passed **1,772 entries**, including all **1,735
required tensor paths**. That checker is reused author code, so its provenance
is explicit (`reused_prospective_validator.py`); it is supporting evidence for
this independent review, not an independently implemented algorithm.

The mathematical review checked the primary Shapley-Taylor and Faith-Shap
papers and pinned source files. The definitions and distinctions agree with
[Sundararajan et al.](https://proceedings.mlr.press/v119/sundararajan20a.html) and
[Tsai et al.](https://www.jmlr.org/papers/v24/22-0202.html): the methods have
different allocation targets, and the Faith kernel and exact endpoint
constraints are stated correctly. The pure-four `0, 1/6` versus `-1/5, 3/10`
calculation and the kernel errors `13/132` versus `1/20` are correct.

I additionally implemented a separate top-order discrete-derivative operator
from the primary papers, including Faith-Shap's highest-order derivative
formula, instead of reusing the supplied Möbius closed forms. Across all **2,992
native games / 83,776 pair coordinates per method**, maximum errors were
`1.4211e-13` (ST), `5.1803e-13` (Faith), and `1.7053e-13` (SII), all well inside
the unchanged strict bounds. Forty-two additional random method/order checks
at n=2..7, including full-order boundaries, passed with maximum error
`3.9857e-14`.

The unmodified retained-reference validator was separately called on all 22
actual reference payloads covering **3,050 game instances**. Every numerical,
bound, count and identity check before its final environment validation
completed; each then stopped at the expected current lock guard. This is
recorded as `NUMERICAL_CHECKS_REACH_EXPECTED_ENVIRONMENT_GUARD`, not overall
reference PASS. It supports the operand-preservation correction: retained
package and independent values are still intact, not replaced by residuals.
The full-order Möbius branch is mathematically the same unique interpolant and
leaves native truncated KKT solves unchanged.

## Learner and figure review

I read `lesson.md`, the actual completed learner, the actual draft exercise
page, method implementations and diagnostic/study call sites, and compared
the structure with the original 0.4 Backprop pair and instruction page. The
candidate presents one question, progresses through eight meaningful exercises,
and supplies adjacent expected outputs, useful help, interpretations and
solutions. All ten taught function ASTs in the completed notebook match the
visible implementations; all 55 code cells have sequential execution counts
and no retained execution errors. The donor-game science is implemented by the
student; shared code supplies bounded indexing and state restoration.

The signature result is the visible higher-order credit redistribution and its
effect on whole-coalition reconstruction. The native capstone includes actual
known fitted models and a concrete recipe/donor/hybrid preview. Method
disagreement, cancellation, shuffled-model interactions and nearly coincident
capture curves are explained rather than hidden. Neither reconstruction nor
ranking is presented as physical causal identification.

I opened the actual exercise HTML in the in-app browser at
`http://127.0.0.1:8871/draft_exercises.html`, expanded **all 54 dropdowns**, and
confirmed **eight open Solution dropdowns**, eleven loaded images, and no page
horizontal overflow. I visually inspected the KKT derivation and Faith closed
form in browser screenshots; equations rendered legibly. The page has 65 math
source elements. I did not infer the renderer's implementation from that count.
An initial loop skipped changing locator indices; the recovery opened the
remaining 27 dropdowns and the final all-open state was verified.

All eleven actual PNGs were individually viewed at available resolution, not
approved from a figure count or author contact sheet. `independent_figure_review.json`
records each hash and substantive assessment. The visual verdict is **PASS for
those bytes**, distinct from the overall publication BLOCK. Dense matrices and
term strips benefit from zooming, but labels, scales, units, controls and signs
are present and no relevant terms are silently removed.

The two editable plays were rerun through the reviewed author variation checker:
**12 native choices, five exact-game choices, and three rejected invalid
inputs passed**. Native play recomputes the indices from retained complete
games and inspects retained donor predictions; it does not perform a new model
fit. The display correctly distinguishes one-donor games, finite-bank means,
and exploratory audit reuse. See `reused_play_checks.py` and `play_variations.json`.

## Original six findings and completion route

1. Authoritative identity and control labels: resolved; obsolete cache prose is
   explicitly superseded, mixed-degree-two is an agreement control, and ridge /
   additive are the zero-pair controls.
2. Deterministic byte contract: resolved in the frozen manifest and retained
   exact values, seed matrices and permutations.
3. Numerical policy: resolved; strict independent bounds and separate finite
   package FSII bounds remain unchanged and actual retained arrays satisfy them.
4. Exact-suite count/order schema: resolved; eight fixed, 45 random-order and
   five separately labeled full-order instances make 58 exact instances.
5. Prospective leaf/shard map: resolved; live rehash and all 1,772 checks pass.
6. Aggregation/ties: stable lexicographic pair ordering, equal recipe-group
   weighting and smallest-ID ties are implemented; **feature-order distance
   accumulation remains unresolved as R1**.

Correct R1, record R2 and the environment transition without rewriting the old
freeze, then execute and independently review the new source-bound candidate.
Only then proceed through canonical pair/page/assets publication, registered
report and navigation/registry integration. No full-course completion claim is
supported by this section review.

Changed files in this review: only scripts, logs and review records in this
cache directory. Original ARENA preservation was freshly audited **PASS with
zero blockers** (`original_preservation.log`). No repository teaching source,
original asset, model state, frozen output, lockfile, or publication record was
edited by this reviewer.
