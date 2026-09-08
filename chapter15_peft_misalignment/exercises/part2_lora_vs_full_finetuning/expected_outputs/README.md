# Expected outputs for [15.2]

`smoke_test.json` is the exact CPU-checkable contract. It proves LoRA matrix
orientation, numerical rank, merge parity, model revision, and the 20-example
disjoint-template split without pretending to be the scientific result.

`reference_metrics.json` records both the bounded CPU preflight and the accepted
96-step real-model CUDA result. The committed `verification_report.json` binds
the exact run to the learner surface by content hash. Expected behavior is:

- rank-1 LoRA, rank-4 LoRA, and full finetuning beat the frozen base;
- random-label and same-norm random low-rank controls remain near chance;
- LoRA update spectra are more concentrated than full finetuning;
- dominant adapter-induced activation-direction ablation hurts target behavior
  more than a matched random direction;
- unrelated TinyStories next-token NLL is reported and bounded, not omitted.

The solution notebook's short CPU run remains a pipeline preflight. It is not
substituted for the full CUDA signature result.

## How this fixture was produced

`run_smoke_test(cpu=True)` computes exact matrix, merge, and split facts. The
separate `run_reference_experiment(device="cuda", steps=96, batch_size=16,
seed=0)` trains the pinned TinyStories model and produces the measured spectra,
behavior, capability controls, and activation intervention in the report.

## Trusted implementation

The LoRA orientation oracle explicitly multiplies `A=[[1,2]]`, `B=[[3],[4]]`
with scaling two and expects `[[6,12],[8,16]]`. Merge/unmerge compares actual
layer outputs and original weights. A trainable exact next-token organism
tests optimizer behavior before the pretrained-model experiment. No fixture
supplies logits or updates to the learner run.

## Random seed

The experiment seed is 0. The visible code derives fixed condition seeds from
it: rank-specific LoRA initialization, full training (+11), shuffled labels
(+101), random-label training (+19), same-norm adapter (+303), and the random
activation direction (+404). These are one declared experiment, not multiple
independent repetitions or broad uncertainty estimates.

## Allowed tolerances

The integer-valued matrix product and exact token/template identities use
`rtol=0, atol=0` semantics. Numerical merge/restoration checks use their stated
absolute-error bounds, and scientific accuracy/spectrum/intervention criteria
use the explicit `artifacts.lock.yml` thresholds. Do not replace those margins
with equality to one historical GPU run.

## When to regenerate

Regenerate after changes to module targeting, LoRA scaling, freezing, merge
semantics, prompt split, training budget, checkpoint, control, or intervention.
Run the function-level tests and complete learner figures first, then refresh
the strict CUDA report. Keep the short CPU path labeled as a mechanics preflight.
