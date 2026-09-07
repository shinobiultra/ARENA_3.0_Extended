# Contrast-consistent search evidence

## How this fixture was produced

The complete learner is executed from its own adjacent solution dropdowns in a
fresh kernel. The executor snapshots thirteen scientific inputs and retains all
figures, fitted reporters, native activations, complete prefix logits and tokens.
Fresh reproduction snapshots fourteen inputs, independently reloads frozen GPT-2,
recaptures all four splits, refits every reporter and compares every retained
condition. All fits and calibration precede final evaluation in both paths.

## Trusted implementation

The original asymmetric loss is checked against hand-computed probabilities,
including its lack of general complement invariance. An exact four-quadrant
counterexample separates consistency from identifying a target. Independent
single-restart execution checks the batched optimizer. Centering and loss match
hash-verified author primitives at revision
`be10d7b912b17e8eacb45de794e9dca044e4ed37`; the author's complete training class
is not used. Native extraction is checked against a direct full-model forward
and individually processed prompts. Padding, pairing and vocabulary metrics
have separate adversarial numerical checks.

## Random seed

Each of the eighteen native cases retains CCS seeds 0–9 in each of three modes.
Pair corruption uses permutation seed 904. All random directions 1000–1031 are
retained with matched selected-reporter norm and bias. Fitting never uses final
labels. Calibration labels choose only the global orientation of each unlabelled
reporter. The supervised control alone uses training sentiment labels.

## Allowed tolerances

The method tests declare exact float64 oracles and tolerances alongside the
functions. Fresh tensor comparison requires matching structure, shapes, dtypes
and finite values, with `atol=1e-6, rtol=1e-6`; scalar metrics allow at most
1e-7 difference. All discrete values and tokens must match exactly. The release
contract additionally bounds the maximum absolute tensor difference at 1e-5.
These checks do not demand favorable sentiment accuracy or discard bad seeds.

## When to regenerate

Changes to executable lesson code or scientific inputs require a new complete
own-answer run and fresh reproduction. Prose-only changes require a new render
review and unchanged executable ASTs. The publisher verifies source snapshots,
all 43 native tensor artifacts in each run, actual figures and the reviewed
lesson. Retain previous runs separately, including failed or superseded versions.

## Scope and counts

The final set has 64 distinct base reviews, 512 contrast pairs across text forms,
three extraction sites and 64 reporters per case: 98,304 final reporter scores.
Fresh execution refits 1,152 reporters and compares 3,586 actual full-vocabulary
prefix outputs across training, calibration, development, final and the play.
Restricted candidate accuracy is not unrestricted token accuracy. The controlled
review labels do not establish general truth discovery, belief identification or
causal necessity. Large tensor records remain outside Git at the exact paths
and hashes in the execution/archive manifests.

A report may reuse an already completed independent reproduction when
`ARENA_CCS_REPRODUCTION_DIRECTORY` is explicitly set. This follows the publisher's
full validation path: exact own-answer identity, every current scientific input,
all source snapshots, complete condition counts and all 43 tensor hashes.
The report labels reused evidence and records that no model was reloaded in
that invocation. Reuse cannot bridge a change in scientific code or tests.
