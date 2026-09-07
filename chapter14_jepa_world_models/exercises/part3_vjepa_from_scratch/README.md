# [14.3] V-JEPA from Scratch: A Persistence-Shortcut Lab

This section asks whether a learner-built V-JEPA trained with the paper mechanics predicts future dynamics or
succeeds through persistence. Students implement canonical forward and reversed clip keys, Conv3D tubelets,
spatial masks repeated through time, visible-only encoding, target layer normalization under stop-gradient,
plain L1 prediction, EMA, tie-aware retrieval, balanced probes, and collision auditing.

The three-seed CPU reference is intentionally a negative result. On the out-of-distribution full-future mask,
V-JEPA reaches `0.531 [0.414, 0.672]` retrieval versus persistence at `0.703 [0.609, 0.852]`; exact dynamics
reaches `1.000`. These are observed ranges, not confidence intervals. The notebook does not claim dynamics
beyond persistence.

Erasing all future pixels while retaining context still gives `0.721 [0.602,
0.945]` retrieval. The teacher attends to the full clip, so its final-slice
features retain past information. The signature figure and editable play cell
show this blank-future control beside persistence. Clip-level latent retrieval
alone cannot establish future prediction.

All 15 exercise groups and 20 code cells were completed from their own dropdown
answers and executed in a fresh CPU kernel. Immediate tests include exact
future erasure, nonmutation, analytic context carryover, indivisible tubelet
rejection, and the zero-rank constant-feature control. This remains an unreleased
generated-world lab; the roadmap's broader video and intervention tasks remain.

Run CPU tests:

```bash
CUDA_VISIBLE_DEVICES="" uv run pytest -q chapter14_jepa_world_models/exercises/part3_vjepa_from_scratch/tests.py
```

Regenerate all three CPU seeds, metrics, and both PNG assets:

```bash
CUDA_VISIBLE_DEVICES="" uv run python chapter14_jepa_world_models/exercises/part3_vjepa_from_scratch/regenerate_reference.py
```

Rebuild the paired notebooks and instruction page after changing learner-facing code:

```bash
CUDA_VISIBLE_DEVICES="" uv run python chapter14_jepa_world_models/exercises/part3_vjepa_from_scratch/rebuild_notebooks.py
```

No external checkpoint, dataset, network access, or GPU is required.
