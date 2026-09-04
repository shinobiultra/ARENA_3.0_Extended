# [6.2] Gemma Scope Deep Dive Expected Outputs

This directory contains frozen fixtures for the learner-facing notebook
contract. The primary fixture describes an exact sparse-feature organism: 40
named examples, 24 held out, six known latents, unequal raw decoder norms, one
train-only confounder, one rare feature, one matched random control, and one
dead feature. Real-model evidence is separately pinned in `artifacts.lock.yml`
and recorded in `verification_report.json`.

## How this fixture was produced

The exact-organism values come from the checked-in deterministic construction
and are protected by `tests.py`. The serialized CUDA verification path can be
refreshed separately by the repository report runner.

## Trusted implementation

The trusted CPU implementation is `solutions.py`; each important value is also
checked from independent invariants in `tests.py`. The real-model record must
match the exact repository, revision, artifact path, model, and layer named in
the artifact lock.

## Random seed

No random draw is used by the exact organism. Its decoder is a fixed DCT basis
and its latent table is deterministic.

## Allowed tolerances

Integer, label, and feature-ID checks require exact equality. Float32 tensor
recovery uses `rtol=1e-5` and `atol=1e-6`; exact reconstruction MSE must remain
below `1e-12`. Real-model values are historical measurements from the pinned
run and are not recomputed by CPU tests.

## When to regenerate

Regenerate these fixtures when the organism, student contract, pinned artifact,
or claim scope changes. A failed invariant should be fixed in the implementation
or reflected honestly in the claim boundary.
