# Live foundation signatures for blank Practice copies

`lesson.additions(section)` returns six fresh notebook cell dictionaries for `"0.6"` or `"1.6"`. Append them only to the new blank Practice copy after its existing cells. Every new code cell has `execution_count=None` and `outputs=[]`; the provider does no plotting, imports only stdlib, and performs no file access or computation at import time. Other section identities raise `ValueError`.

The six-cell arc is forecast → live plot → help/expected interpretation → play forecast → changed-condition plot → limitations. Core mathematical methods remain the student's existing definitions in the main notebook. There are no new method stubs or hidden reference implementations.

For 0.6, literal CPU fixtures call the learner's `label_leakage_report` and `cherry_pick_report`.
Two panels show label-encoding versus inverted-feature accuracy and selected-tail versus
representative/population means. The inverted feature is explicitly **anti-correlated**, not an
independent null. Play changes only the selected indices and recomputes the learner callback. No
detector thresholds change.

For 1.6, the learner's `estimate_inference_memory` is called across four contexts with explicit
illustrative tensor counts. Curves show parameters, KV cache, activations, total and a deliberately
omitted-KV underestimate, with an illustrative 3 GiB budget. Labels use **GiB**, matching the
original `1024**3` conversion despite `_gb` field names. Play doubles only batch size. No tensors
representing the illustrated model are allocated; there is no model loading, GPU allocation,
profiling or hardware measurement.

All plotted data are callback return values. Changing the student's implementation changes the
figure. Missing/unimplemented callbacks fail openly rather than being replaced by a saved answer,
imported solution or historical report. Expected values occur only in reveal-after-forecast
explanatory prose. No image, output, execution count, report acceptance or historical hash is
supplied by this provider.

Integration is separate from review: this candidate does not modify `read.py`, the blank-copy helper, canonical masters, existing notebooks, evidence or reports. Root's blank-copy worker owns V2 integration after V1 is frozen. These additions do not make a real-model research or course-readiness claim.
