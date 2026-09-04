# Hard Exercise Verification Ladders

Hard exercises are incomplete if they only test the final answer. Any exercise
with difficulty 3/5 or above should be decomposed into a visible ladder of
small checks before the real-model or full-report stage.

The generated [Hard Exercise Ladder Registry](hard_exercise_ladder_registry.md)
tracks every current difficulty-3+ extension section, the artifact sources that
prove the notebook-contract layer, and the release evidence still pending
before stronger real-model claims.

## Required Ladder

Each hard exercise should include:

1. shape tests
2. dtype and device tests
3. small hand-computed examples
4. sub-function tests
5. brute-force or reference comparison where possible
6. randomized property tests
7. integration test
8. real-model smoke test when the exercise makes a real-model claim
9. full verification report

The learner-facing notebook should place tests immediately after the relevant
exercise block. CI-only tests are not enough, because students need local
feedback that identifies the smallest failing sub-step.

## Metadata Block

Every nontrivial implementation block should expose metadata near the function
stub:

```python
EXERCISE_ID = "5.4.eap_ig.edge_scores"
GT_TIER = "GT-1"
DIFFICULTY = 4
IMPORTANCE = 5
EXPECTED_RUNTIME = "seconds on toy, minutes on real model"
REQUIRES_GPU = False
```

Function docstrings should state expected input shapes, output shapes, and the
most important invariants. Failure messages should teach a likely bug, not just
state that an assertion failed.

## Oracle Before Optimization

When possible, require both:

```text
slow_correct_version
fast_vectorized_version
```

The optimized implementation must numerically match the oracle on toy cases
before it is used on real models. Examples:

- Mamba: recurrent selective scan before parallel selective scan
- SHAPley: exact coalition enumeration before KernelSHAP or Monte Carlo Shapley
- EAP or EAP-IG: exact edge patching on a tiny graph before scalable scores
- Diffusion: explicit noising distribution before vectorized samplers
- LoRA and DoRA: explicit weight recomposition before merged modules
- VLM patching: direct activation replacement before batched cached patching

## Property Tests

Use randomized invariant tests for mathematical components:

- RoPE: norm preservation, relative-position identity, inverse rotation
- RMSNorm: controlled RMS and positive-scaling behavior
- LoRA: rank bound, merged/unmerged parity, zero adapter recovers base output
- DoRA: magnitude times normalized direction reconstructs weight
- Mamba: chunked scan equals full scan, recurrent scan equals parallel scan
- SHAPley: efficiency, symmetry, dummy, and linearity
- Integrated gradients: zero delta gives zero attribution; linear case matches
  gradient times input difference
- Diffusion noising: schedule matches analytic distribution; endpoints behave
  as specified

## Golden Fixtures

Hard notebooks should pin tiny expected outputs under:

```text
assets/expected_outputs/
```

The existing exercise layout may use `expected_outputs/`; either way, the
fixture directory needs a README or equivalent metadata describing:

- how the fixture was produced
- which trusted implementation produced it
- random seed
- allowed tolerances
- when regeneration is allowed

## Tolerances

Every numerical comparison must specify tolerances explicitly:

```python
torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
```

Do not use vague checks such as `assert looks_reasonable(output)` unless they
are followed by quantitative criteria.

Suggested defaults:

- integer and token tests: exact equality
- float32 toy tests: `rtol=1e-5`, `atol=1e-6`
- bf16 model tests: `rtol=1e-2`, `atol=1e-2`
- quantized model tests: compare behavior, top-k overlap, or KL rather than
  exact logits
- stochastic generation: fixed seed or distributional metric

## Checkpoints and Debug Mode

Long notebooks should include checkpoint cells for shape checks, toy numerical
checks, oracle-vs-vectorized parity, real-model smoke, and final causal or
verification evidence.

Complex functions should support `debug=True` and return intermediate caches
where useful, for example attention tensors, interpolated activations, edge
scores, visual token positions, or projected embeddings.

## CI Markers

Tests should be tagged by cost:

```python
@pytest.mark.unit
@pytest.mark.cpu
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.requires_model
@pytest.mark.requires_gated_model
@pytest.mark.requires_large_download
```

Unit and CPU tests should run on every commit. GPU smoke tests should be run
before merging a chapter, and slow or full tests should run before release.

## Acceptance Rule

For every hard exercise, the final report is necessary but not sufficient. The
exercise is accepted only when sub-functions are tested before the full
function, there is a toy case with known ground truth, an optimized
implementation matches a reference where possible, failure messages localize
common bugs, and partial expected outputs are visible to the learner.
