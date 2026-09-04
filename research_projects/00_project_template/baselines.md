# Baselines

Every project needs at least one serious baseline and one negative control.

Required baseline classes:

- task baseline: a simple method that solves the benchmark directly
- text-only or input-only baseline where relevant
- probe or linear baseline where activations are involved
- random-feature, random-direction, random-region, or random-token control
- label-shuffled or template-shuffled control when labels are generated

Record each baseline with:

```text
name:
implementation:
why it is strong:
metric:
expected failure mode:
```
