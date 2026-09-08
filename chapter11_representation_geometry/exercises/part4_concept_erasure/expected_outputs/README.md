# Concept-erasure experiment evidence

The notebook implements and executes the method. These records support its
figures and claims; they are not a substitute for the learner.

## How this fixture was produced

`execute_learner.py` completes each student function from its own adjacent
solution dropdown and executes the full lesson in a new kernel. It snapshots
the scientific inputs and retains every figure, fitted map, probe and actual
native vocabulary output. `erasure_reproduction.py` independently reloads the
pinned checkpoint, recaptures inputs, refits all erasers and readers, and repeats
all 106 final conditions on 288 role and 384 object prompts.

## Trusted implementation

The four-point example has a hand-computed optimum. A separate NumPy KKT solve
checks the general full-rank constrained quadratic problem; singular and affine
invariants are checked on multiple random cases. Three additional cases agree
with the pinned author implementation at revision
`9f51753821316a1edacf78b52b464ab26d40e60a`, with covariance shrinkage and the trace
constraint explicitly disabled for the unregularized comparison. The author
source and license are hash-checked. Native full-forward intervention hooks
independently check the cached continuation. Fresh inference checks every
retained final full-vocabulary output and its actual winner.

## Random seed

All generated prompt rows are deterministic and defined before development.
Each native site retains 16 random directions: seeds 1000–1015 and 2000–2015.
Every direction is used for an orthogonal rank-one control and two per-example
magnitude-matched controls. No draw is selected on final outcomes. Every new
probe is fit after the edit using the separate probe-training split.

## Allowed tolerances

The exact float64 checks declare tolerances beside their independent oracles,
including `atol=2e-11, rtol=0` for singular affine invariants. The native float32
full-forward/cached comparison allows `atol=3e-4, rtol=0` across the complete
vocabulary. Fresh reconstruction additionally requires identical winning tokens;
the normal report requires maximum full-vocabulary error at most 1e-5. Probe
logits and fitted maps are compared separately, without sign-sensitive SVD
factor comparisons. These tolerances do not impose a favorable scientific
effect or remove failed interventions.

## When to regenerate

Changes to executable learner code, scientific functions, the checkpoint,
prompt construction or the protocol require a new full learner run and native
reproduction. The publisher binds code ASTs, source snapshots, every tensor
file and actual figures. Prose-only changes require a new rendered review and
must not silently change experimental outputs. Retain earlier runs, including
failures, in distinct persistent directories.

## Scope

Training-set class-mean equality is a finite distribution property. The native
study distinguishes this property from recovery by a new reader, recipient
answer changes, unrelated object damage and output-distribution KL. It does not
claim unrestricted information removal or all-layer concept scrubbing. Large
actual tensors remain outside Git in the source directories named by the
execution records and archive manifests.
