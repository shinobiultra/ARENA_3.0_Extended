# Expected outputs

The seed-0 CPU signature run should show:

- falling training loss and rising held-out shortest-action accuracy;
- an actual held-out maze whose constrained rollout is legal and reaches the goal;
- separate legal-path, goal-reach, and exact-optimality rates;
- raw illegal-move failures rather than silently masking them;
- layerwise distance, local legal-degree, and action probes with randomized-label
  and explicit Manhattan-distance controls;
- query-state patching beside wrong-position, matched-norm random, unrelated-state,
  and same-action other-maze controls; the same-action control need not fail;
- a zero-overlap train/held-out wall-template check;
- explicit obstacle-density and unseen-size OOD results.

`reference_metrics.json` records actual CPU and CUDA runs separately, actual test
output, learner execution counts, and input hashes. The original numerical
thresholds remain fixed. CPU passes those thresholds; the larger CUDA run fails
the middle-layer patch gates. Neither run supports a route-specific or internal
search claim, and course acceptance remains false.

After rebuilding and executing the learner notebook, regenerate the record with:

```bash
uv run python chapter14_jepa_world_models/exercises/part4_maze_transformer_world_model/regenerate_reference.py --verify-cuda-repeatability
```

`--cpu-only` explicitly records CUDA as untested. It does not silently substitute
CPU execution for GPU evidence.

The learner-facing evidence is the rendered maze, curves, bars, and patching
heatmap in `instructions/assets/maze_transformer_signature_result.png`. A path
image without legality, goal reach, and optimality metrics is not sufficient.
