# Mini VLM checked recovery practice addon

Open [the exercise notebook](exercises.ipynb). It asks when an entire answer-recovery curve is
defined and keeps the learner's own metric and patching callbacks visible.

The tiny literal CPU controls require no model, data, training, RNG, cache, network or GPU. The
final learner-run and plotting cells require a separately authorized, already prepared 12.3 session;
do not run them to initialize missing state. The source notebook has blank execution counts and
outputs.

## Continue inside your existing 12.3 notebook

A separate addon tab has a separate kernel; opening it does not share the trained model or
cached cohort with the lesson. Finish the tiny CPU exercises here first. For the final adapter,
copy your **completed code cells**, including their immediate controls, into your already
working 12.3 notebook in its existing live session. Copy the setup import, failure/helper
class definitions, `validate_recovery_baselines`, `checked_learner_recovery`,
`checked_learner_sweep` and the three immediate test cells, in their original order. Keep
your own metric and patching implementations; copying a solution dropdown is exposed practice.

Use the lesson's code boundary containing `patch_sets = { ... }` followed by
`patch_curves = patching_sweep(...)`. In your working copy, split that cell **after the
closing brace of `patch_sets` and before `patch_curves = ...`**. Insert your completed addon
cells, then the addon cells labelled `FUTURE LEARNER RUN ONLY` and
`FUTURE SUCCESS-ONLY DISPLAY` there. The cohort/cache cell starting
`pairs = utils.make_counterfactual_pairs(100, seed=7)` and the method tests must already have
run successfully in this same session. Use this boundary before the later Mini-to-Qwen
cleanup, which removes the Mini caches. The historical notebook remains the reference;
record the prospective checked result separately in your working copy.

Check these live prerequisites before running the final adapter. Let `B = len(pairs)`,
`S = SEQ_LEN`, `D` be the model width and `V` vocabulary size; the default lesson uses
`B=100`, `S=60`, `D=768`.

| Existing name | Required state |
| --- | --- |
| `t`, `plt`, `tests` | Existing Torch, matplotlib pyplot and 12.3 tests bindings |
| `logit_diff_recovery`, `patched_answer_logits` | Your completed functions with their original immediate tests passing |
| `vlm`, `N_LAYERS`, `ANSWER_POS` | Live Mini model and matching cache/answer-position convention |
| `pairs` | Original complete selected cohort, in its original order; no post-failure filtering |
| `corrupt_embeds` | `[B, S, D]` tensor |
| `clean_resids[layer]` | `[B, S, D]` for every layer in `range(N_LAYERS)` |
| `clean_logits`, `corrupt_logits` | `[B, V]` tensors for this same cohort/model |
| `clean_ids`, `corrupt_ids` | Integer `[B]` answer token IDs |
| `patch_sets` | All six existing named boolean `[B, S]` masks, including background, other-object and full-image controls |

Shapes, row order, device and model/cache identity must agree. If a prerequisite is absent,
stop at the tiny CPU controls; this handoff does not authorize loading, retraining, restoring
or initializing a missing model. A shared notebook filename or another tab showing old
outputs does not establish a shared live session. Retain the original pair IDs before any
measurements and preserve cohort failures rather than removing failing rows.

Complete the three inline guard exercises and immediate controls before opening their solution
dropdowns. Forecast first; then use the play fixture to make a zero/reversed gap or nonfinite
recovery fail with original pair indices. The expected control table has gaps 4 and 2. No tiny
positive gap is excluded; finite recovery outside [0,1] is retained.

The existing [12.3
exercise](../../../chapter12_vlm_interpretability/exercises/part3_mini_vlm_from_scratch/12.3_Mini_VLM_from_Scratch_exercises.ipynb)
and its saved historical Mini reference stay unchanged. The [prediction
workbook](../../units/extension_predictions/exercises.ipynb#section-12-3) remains the
forecast/reflection companion. This addon is not new scientific execution or human learner
acceptance.

All guard logic and reference answers are source-visible in the notebook. No hidden metric module,
reference callback fallback, automatic model initialization or output-file writer is supplied. The
callback is required to be pure with respect to its inputs; validation is not a sandbox for
arbitrary user code.

`source_manifest.json` binds this addon, the adopted design, and the unchanged current canonical
inputs. A future run needs its own callback/cohort/model/runtime provenance and new output location.
Do not overwrite historical evidence or call these engineering checks scientific acceptance.
