# [6.2] Gemma Scope Deep Dive

This section starts with an exact six-feature organism, then escalates to a
strictly pinned Gemma Scope 2 result.

- `6.2_Gemma_Scope_Deep_Dive_exercises.ipynb` asks students to implement
  function-preserving decoder normalization, JumpReLU encoding,
  reconstruction, held-out scoring, density diagnostics, matched ablation and
  steering, and direct logit attribution.
- `6.2_Gemma_Scope_Deep_Dive_solutions.ipynb` contains the same learner arc
  with executed CPU outputs, visible examples, and the signature figure.
- `utils.py` constructs the deterministic ground-truth organism and validates
  the identity of the committed real-model evidence. It contains no learned
  method implementation.
- `solutions.py` is the reference implementation and retains the CUDA refresh
  entry point used by the verification runner.
- `tests.py` contains immediate semantic and edge-case tests for every student
  function.
- `artifacts.lock.yml` records both the exact CPU claim and the pinned Gemma
  Scope 2 / Gemma 3 evidence boundary.

The committed CUDA report remains supporting evidence. The notebook's primary
result is computed from the exact organism, not loaded from that report.
