# [7.4] Expected Outputs

## How this fixture was produced

The canonical prompt table defines 32 lexical contexts with four number/time
combinations each. Three combinations train the AV and AR; one is held out per
context. `evaluate_mini_nla` computes all reconstructions, controls, and real
Pythia hook effects afresh. The reference file records those outputs.

**Trusted implementation:** pinned Transformers Pythia forward passes, the
explicit dual ridge solution, and row-normalized reconstruction metrics checked
against identity and collapsed-prototype controls. The AR uses the layer-4
representation at the last token; it is a small supervised adaptation.

**Random seed:** the split is deterministic and balanced across all four factor
states; AV shuffles enumerate seeds 0 through 7. AV ridge is 0.1 and AR ridge
is 0.001. No random seed is selected for a favorable reconstruction result.

**Allowed tolerances:** identity and unselected-row parity use `rtol=0`,
`atol=0` because the intervention reruns the same deterministic computation.
Distinct-row checks use Euclidean `atol=1e-2`. Float32 FVE must meet the frozen
0.75 threshold and beat the bag-of-tokens control by more than 0.10; exact text,
split membership, and token-multiset equality receive no numerical tolerance.

**When to regenerate:** rebuild the notebooks/page, execute the completed
learner and CPU tests, inspect the plot and prompt table, then rerun the strict
CUDA report after changes to prompts, code, tests, prose, dependencies, or locks.

The executed solution notebook must display the real 32-row prompt/explanation
table and a three-panel figure computed from the same run. The signature is not
read from `verification_report.json`.

Expected CUDA 13.2 behavior:

- 32/32 held-out explanation sequences are exact;
- every context's four factor states have exactly the same token multiset;
- contextual normalized FVE is about `0.816`, above `0.75`;
- bag-of-tokens FVE is about `0.642`, at least `0.10` lower;
- contextual behavior parity is `1.0`, while bag-of-tokens parity is `0.0`;
- all 32 reconstructions remain distinct;
- identity patch error and non-selected-row change are exactly zero; and
- the real `AR(edited) - AR(base)` intervention changes a candidate logit by
  more than `1.0`.

The bag-of-tokens within-factor FVE is expected to be high, around `0.841`.
This is not a failure of the test: noun and style tokens identify lexical
context. Its zero behavior parity exposes the specific order-sensitive
information it cannot recover.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python -m pytest -q \
  chapter7_activation_to_language/exercises/part4_mini_natural_language_autoencoders/tests.py
```

Counts and locality checks are exact. FVE thresholds allow small float32 kernel
differences. No alternate checkpoint, CPU fallback, synthetic replacement, or
report-based acceptance is permitted.
