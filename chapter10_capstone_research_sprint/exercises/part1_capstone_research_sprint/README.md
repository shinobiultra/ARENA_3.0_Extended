# [10.1] Capstone Research Sprint

Start with `10.1_Capstone_Research_Sprint_exercises.ipynb`. The section teaches
research process through one complete mechanistic study rather than a planning
schema.

The exact model organism computes XOR from two signed bits, appends five
template-dependent nuisance features, and applies a fixed orthogonal rotation.
Its causal XOR direction is therefore known analytically but distributed across
all eight activation coordinates. Students:

1. build balanced train and held-out template splits;
2. recover the direction with a ridge probe;
3. compare raw-bit, template-only, and shuffled-label baselines;
4. quantify the held-out improvement with a paired bootstrap;
5. perform counterfactual directional patching and ablation;
6. compare 256 same-dimensional random directions;
7. find the noise failure boundary and investigate the strongest null anomaly.

The reference run is CPU-only and takes seconds. `scripts/run_capstone.py`
regenerates `results/metrics.json`, `results/failure_cases.jsonl`, and
`reports/capstone.md`. The optional serialized GPU contract repeats the same
exact operations; it does not substitute a different scientific experiment.

The claim is limited to method validation on this exact model organism. It is
not evidence that a released transformer contains a single stable XOR direction.
