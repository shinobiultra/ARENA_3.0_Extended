# Refusal Directions and Safe Steering

Start with [the learner notebook](9.1_Refusal_Directions_and_Safe_Steering_exercises.ipynb). In twelve exercises, fit a direction from actual Qwen activations, distinguish classification from behavior, implement persistent interventions and generate the final comparison yourself. The [solution notebook](9.1_Refusal_Directions_and_Safe_Steering_solutions.ipynb) contains the same lesson with complete answers and measured outputs. `lesson.md` is the canonical paired source.

Allow one to two study days. The full experiment uses the pinned Qwen2.5-0.5B-Instruct checkpoint in FP32 with SDPA on CUDA, at most 48 new tokens, and a roughly 3 GiB model-workload budget. Runtime depends on competing GPU workloads; the exact completed run is recorded in `expected_outputs/learner_execution.json`. Begin after original ARENA 1.3.2, Function Vectors and Model Steering. Original ARENA files remain unchanged.

## What you implement

Training-only mean direction and threshold; tie-correct AUROC; frozen evaluation; 32 genuine training-label permutations and 32 same-norm random directions; centered training PCA; correctly indexed native-Qwen residual caching; raw-vector addition and unit-direction ablation; hooks that persist across every decoding step; greedy generation with KV cache and independent EOS stopping; an inspectable refusal-marker rubric; paired transition denominators; development-only selection across all fifteen candidates.

The numerical exercises expose an evaluation-threshold leak and include exact geometric tests. The real-model path uses 64 fitting prompts, 64 development prompts and 72 reserved prompts, with 32 exact benign questions in each evaluation split. Addition changes one decoder input at every position and call. Ablation changes embeddings and every attention/MLP residual write. The notebook's own functions perform the inference; data loading, plot layout and independent tests are helpers.

## Results to inspect

Six figures show development layer/position separation, all fifteen candidate effects and capability costs, training geometry against nulls, final score distributions, final intervention/capability curves and actual benign generations. Selection freezes before final prompts load. All 85 conditions retain the first eight random/permutation draws, three addition amplitudes and two ablation strengths.

The reference development choice is layer 16, suffix offset -1. On reserved prompts, multiplier-one addition raises benign marker rate from 1/36 to 27/36; full ablation reduces refusal-category marker rate from 26/36 to 0/36. Addition at multiplier two reduces exact capability from 28/32 to 6/32. A permutation reaches AUROC about .999 and some permutation ablations have substantial effects. Keep these results: ranking does not establish a unique behavioral axis, and a stronger intervention is not automatically better.

Three play cells vary the training geometry site, generate the birthday/arithmetic examples under target and random interventions, and compare cached with uncached decoding. Their outputs come from the real model or its actual cached activations.

## Run and validate

From the repository root, use the project environment:

```bash
uv run python -m pytest -q chapter9_alignment_interpretability/exercises/part1_refusal_directions_safe_steering/tests.py
uv run python chapter9_alignment_interpretability/exercises/part1_refusal_directions_safe_steering/build_notebooks.py --check
uv run python chapter9_alignment_interpretability/exercises/part1_refusal_directions_safe_steering/execute_learner.py --scope numerical --output-directory /tmp/refusal-numerical-new
```

For a full fresh learner run, use `--scope full` and a new output directory. To publish its outputs, first run the builder without `--check`, then run the executor with `--publish`. Publication requires all 26 code cells, unchanged section sources, exact paired method agreement, raw measured evidence and all six figures. Numerical-only execution cannot publish. `solutions.run_full_experiment()` performs the build, full own-answer execution/publication and separate CUDA recheck.

`direction_evidence.py` recomputes the stored measurements, frozen selection, denominators and control inventory. `direction_cuda.py` independently refits from training only and regenerates all 34 unit-strength target/random/permutation conditions plus baseline. It compares individual completion hashes, markers, lengths and benign generated tokens, and rechecks all 65 frozen score directions. Stored evidence is not described as a fresh model run. Supporting reports follow the learner results.

## Scope and limits

This is a scoped method study connected to [Arditi et al.](https://arxiv.org/abs/2406.11717) and [their implementation](https://github.com/andyrdt/refusal_direction). It does not reproduce their complete model suite, datasets, generation budget or harmfulness evaluation. The English marker rubric makes visible mistakes; absence of its markers is neither safe behavior nor a correct answer. Capability checks cover four familiar task templates with new inputs, not broad general ability. Sensitive benchmark rows store IDs, complete hashes and outcomes; benign text and generated tokens remain inspectable.

The previous prototype and its misleading acceptance contract are archived with hashes under `expected_outputs/legacy_proxy`. Its evaluation-dependent threshold, reversed-label shuffle and prompt-only generation intervention are superseded. They do not contribute to acceptance of this learner.
