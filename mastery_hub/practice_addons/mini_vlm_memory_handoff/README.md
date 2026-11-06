# MiniVLM: make room before optional Qwen section 5

You can read all of **12.3 Mini VLM from Scratch** and its saved figures without loading any model. Stop before **5️⃣ Is a Real VLM Doing the Same?** if you are studying the MiniVLM first. Reading those historical results is exposed practice.

If you run the optional Qwen stage, its original cleanup loop deletes a few names but leaves GPT-2, the trained MiniVLM, projectors, SigLIP and other large tensors resident. `empty_cache()` cannot free storage still owned by those live objects. The roughly 4.4 GB Qwen weight size and 6 GB budget in the historical lesson do not guarantee a successful run: activations, caches and allocator headroom also consume memory.

The companion [stage_memory.py](stage_memory.py) provides visible, editable plumbing to move the earlier models, features and analysis caches to CPU. It keeps the existing model handles and tensor values; it preserves tensor aliases shared between named caches and inputs. `patch_curves` stays exactly as it was for the later comparison. CPU offload needs enough host RAM. Extra tensors or aliases you created yourself, notebook output history and external references may retain GPU storage.

## Use in your working notebook

This companion requires a **full local checkout** containing `mastery_hub/practice_addons/mini_vlm_memory_handoff/stage_memory.py`. The canonical Colab setup unzips only the chapter exercises, so it does not include this package. Adding `root_dir` to `sys.path` exposes files already present; it does not install or download the companion. Use this recipe in the full local checkout.

Use a saved working copy of the exercise notebook. Find the first code cell under **Qwen3-VL-2B on the same scenes**. Replace only its initial cleanup block, from `# Free what we no longer need` through `t.cuda.empty_cache()`, with this code:

```python
import gc
import sys

# The lesson setup defines root_dir; expose the companion package from that checkout.
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
from mastery_hub.practice_addons.mini_vlm_memory_handoff.stage_memory import move_minivlm_stage

# Keep the earlier stage available on CPU before loading the larger model.
move_minivlm_stage(globals(), "cpu")
gc.collect()
if device.type == "cuda":
    t.cuda.empty_cache()
```

Keep the following `qwen, qwen_processor = utils.load_qwen(device)` line and the remainder of the cell unchanged. The existing setup defines `root_dir`; this snippet adds it to Python's import path (the historical setup adds only the chapter exercise directory). If you copied this notebook outside the checkout, first open it beside the original exercise notebook so its setup finds the lesson files.

**Replace the old cleanup; do not run both.** The original `globals().pop(...)` loop would delete handles after they were moved. Apply this handoff at the boundary before the first Qwen load, when the names still refer to MiniVLM objects. It deliberately excludes Qwen and its named inputs; it also protects `qwen` if an earlier model name aliases that selected model. It does not unload an already active Qwen stage or manage arbitrary tensors from later cells.

Open the short helper source if you want to inspect or extend its explicit owner list. The core training, evaluation, patching and readout implementations remain in the lesson.

## Return to MiniVLM play

Retain the original `device` value. Before returning to `ask`, `patch_demo` or the earlier analysis, finish the Qwen experiment. If `qwen` still exists, park its model on CPU as shown below. The final Qwen cell already executes `del qwen`, so skip that move when the final cleanup has run. Also move or clear the Qwen-only inputs, generated tensors and activation caches you created: later cells use names including `example_inputs`, `generated`, `inputs`, `inputs_clean`, `inputs_corrupt`, `cache`, `lc`, `lk` and `logits`. A model move alone does not move those tensors.

After handling those Qwen-only tensors, conditionally park the model and restore the earlier stage to its original device:

```python
if "qwen" in globals():
    qwen.to("cpu")
move_minivlm_stage(globals(), device)
gc.collect()
if device.type == "cuda":
    t.cuda.empty_cache()
```

Run this before rerunning the desired MiniVLM play/analysis cells, so models, cached features, masks and token IDs agree with `device`. Later Qwen cells reuse some short temporary names such as `obj_mask`; rerun the corresponding MiniVLM analysis cell to recreate any overwritten temporary. If you ran the original deleting cleanup, deleted objects must be recreated from the earlier cells. The historical run did not save recoverable trained MiniVLM weights, so restarting may require training again.

## What is checked

CPU-only tests use fake module/tensor owners and AST inspection to check offload, nested cache aliases, restore behavior and exclusion of the selected Qwen model. They import no model packages and execute no model, training loop, checkpoint download or GPU operation. No GPU peak-memory or out-of-memory guarantee is established.

This additive companion changes no canonical master, exercise/solution notebook, reading page, method, numerical result, cohort, checkpoint pin or saved figure. The main solutions notebook remains the exact historical 61-cell execution; this helper has no scientific execution outputs and establishes no fresh whole-notebook or learner acceptance.
