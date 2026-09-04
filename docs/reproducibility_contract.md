# Reproducibility Contract

The extension notebooks should preserve the original ARENA style: implement the idea directly, test it locally, and keep the explanation close to the code. The main addition is that every frontier notebook must provide a proof block before it asks the learner to trust a result.

## Required entry points

Each extension notebook should expose:

```python
def run_smoke_test(cpu: bool = True) -> dict:
    ...

def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    ...

def run_full_experiment():
    ...
```

`run_smoke_test` should be cheap and deterministic. It should avoid model downloads and API calls unless the section cannot be meaningfully tested without them.

The historical name `run_smoke_test` is an API hook, not an evidence claim. In
generated reports it is treated as a notebook contract and recorded with an
explicit evidence level and claim scope. See
[Verification Quality Policy](verification_quality_policy.md) for the rule that
synthetic controls are allowed only as labeled ground-truth checks, not as
proof of full real-model behavior.

`run_gpu_test` may load a small checkpoint, but it must print estimated memory, measured peak VRAM, model id, dtype, and pass/fail criteria.

`run_full_experiment` may be expensive, but it must save artifacts with enough metadata to reproduce the run.

## Architecture correctness

Architecture notebooks should verify:

```text
shape tests
dtype tests
gradient tests where training is involved
HF or official-code logit parity
generation parity for deterministic decoding
KV-cache parity where applicable
speed benchmark where relevant
VRAM estimate and measured peak VRAM
```

## Mathematical equivalence

Use explicit equivalence tests when the method has two implementations:

```text
Mamba: parallel selective scan == recurrent selective scan
Mamba: chunked scan == full scan
RoPE: rotation preserves norm
RoPE: relative-position identities hold
Diffusion LM: forward noising distribution matches schedule
JEPA: masking and EMA target-encoder updates behave as expected
```

## Interpretability validation

Interpretability claims need more than plausible examples. A claim should be tested against:

```text
held-out examples
contrastive examples
random controls
causal ablation
causal steering or patching where appropriate
OOD prompt templates or synthetic OOD splits
simple baselines
```

If a claim is only supported by max-activating examples or a single visualization, label it as a hypothesis, not a result.

## Sparse feature methods

SAE, transcoder, and crosscoder notebooks should report:

```text
L0 or active feature count
loss recovered
feature density
dead features
reconstruction MSE or KL
held-out feature validation
random-feature baseline
ablation or steering effect
```

## World-model and JEPA claims

World-model notebooks should verify:

```text
latent probe predicts true state above baseline
latent rollout predicts future state above baseline
action-conditioned rollout responds correctly to action changes
counterfactual patch changes prediction as expected
random-direction patch fails
OOD split passes
```

## Artifact metadata

Saved artifacts should include:

```text
checkpoint id
dataset id or generation script
git commit when available
seed
dtype
device
software versions
notebook section
run command or cell entry point
```

The goal is not bureaucratic logging. It is to make a later reader able to distinguish a real mechanistic result from a notebook that happened to run once.
