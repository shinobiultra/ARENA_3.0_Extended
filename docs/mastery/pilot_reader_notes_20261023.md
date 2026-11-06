# Reading the 7.1 and 12.3 results

These notes clarify two interpretations and one reflection prompt in the public lessons. They are dated 23 October 2026. The historical source notebooks, outputs and practice records remain unchanged. On the updated instruction pages, the relevant note appears beside the passage it qualifies. If you open a raw notebook, use the notes below alongside it.

<a id="7-1-interpretation"></a>
## 7.1: what the Patchscope grid tells us

<!-- reader-note:7-1-interpretation:start -->
**Reading note — 23 October 2026.** The grid shows where a donor vector can be used in this inspection prompt.
Both the computation left after insertion and the compatibility of the donor and target representations affect success.
The cutoff motivates a hypothesis about where recall happens; it does not establish necessity of blocks 1–5 for native factual recall.
Keep that distinction in mind when reading the historical explanation below.
What further intervention would distinguish the two explanations?
<!-- reader-note:7-1-interpretation:end -->

This qualifies the grid interpretation in the canonical exercise's cell 99 and its summary in cell 109 (zero-based). The blank public practice copy already includes this distinction after the grid.

<a id="7-1-summary"></a>
### Carry the distinction into the summary

<!-- reader-note:7-1-summary:start -->
**Reading note — 23 October 2026.** In the historical summary below, read the target-layer cutoff as a result about this inspection prompt and donor/target compatibility. It does not establish necessity of those blocks for the model's native recall computation. A successful readout and a necessity test answer different questions.
<!-- reader-note:7-1-summary:end -->

<a id="7-1-controls-reflection"></a>
## 7.1: a control need not succeed

<!-- reader-note:7-1-controls-reflection:start -->
**Reflection clarification — 23 October 2026.** The historical reference reports 0% for each of the three controls.
The old practice prompt asks you to identify a country where a control was correct; there may be none.
If a control is correct in the result you inspect, discuss one such country.
Otherwise record **no control successes** and inspect a patched-success/control-failure case.
Report what you ran or read; do not invent a successful control.
This clarification applies to the old blank practice reflection and prediction workbook, whose raw notebooks remain unchanged.
<!-- reader-note:7-1-controls-reflection:end -->

The affected entries are blank public practice cell 108 and general prediction workbook cell 49 (zero-based). Forecasts, confidence and observations remain your own records. Reading the reference is exposed practice, not an unseen test.

<a id="12-3-recovery"></a>
## 12.3: check the whole cohort before averaging recovery

<!-- reader-note:12-3-recovery:start -->
**Before making a new recovery curve — 23 October 2026.** The formula below divides by each pair's clean-minus-corrupt logit gap. Check that gap before averaging. For example:

| Original pair | Clean LD | Corrupt LD | Gap |
| --- | --- | --- | --- |
| 7 | 3 | -1 | 4 |
| 9 | 0 | 0 | 0 |

Pair 7 has a defined positive gap.
Pair 9 has no clean-versus-corrupt contrast to normalize, so its recovery is undefined.
Under the prospective practice policy, this stops the whole cohort before any mean or figure; silently dropping pair 9 would change the question.
Reversed or nonfinite gaps and nonfinite recovery also stop the cohort.
A finite tiny positive gap is retained, and a finite recovery outside [0, 1] is allowed; neither is automatically a well-behaved effect.

Complete the **checked recovery practice notebook** before changing scenes or running a new cohort sweep.
It keeps your own recovery and patching functions visible and supplies small controls for the guards.
Its final model cells require an already prepared 12.3 session; opening the addon does not prepare or train that model.
The saved historical Mini curve retains its original normalization and is not revalidated by these notes.
<!-- reader-note:12-3-recovery:end -->

Open [the checked recovery practice notebook](../../mastery_hub/practice_addons/mini_vlm_checked_recovery/exercises.ipynb) from your checkout. Its first exercises use literal CPU fixtures; the final learner run has separate prerequisites. The relevant canonical exercises begin at cells 141–147 (zero-based).

These notes add no experimental result. They do not rewrite raw notebooks, rerun models or establish human learner acceptance.
