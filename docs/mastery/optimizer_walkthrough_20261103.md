# Optimizer learner walkthrough — Q001

The first learner walkthrough failed on usability: the UI was too spread out
and unintuitive. The current compact candidate needs a repeat of the affected
steps below. Q001 remains failed/pending direct human acceptance. Full #39
completion remains the v1.0 release requirement; passing this walkthrough would
open the next Alchemy protocol review, not approve a release or authorize training.

Open [the local optimizer](http://127.0.0.1:18506/). This preview uses the existing
repository `.venv` and its configured local progress directory. Check **Study
tools → Local progress and privacy** for that location. The author's separate
native checks used an isolated session on port18508; those records are not your
walkthrough. To launch a personal session from the repository root, follow
[Install the study reader](install_study.md), with its optimizer command.

The main workspace shows essential settings, playback and one selected chart.
Use **Loss / Trajectory / Update size** to switch the result; expand **Advanced
settings** and **Chart options · baseline and scale** when needed. The sidebar
contains **Study tools**: worked example, notebook downloads, original sources,
other companions and timer. Open the sidebar using its top-left control if it is
collapsed. On a phone the chart requires vertical scrolling.

1. In **Explore**, open sidebar **Study tools → Start here · the model and a
   worked example**. Before changing controls, hand-compute
   one baseline update. Check the displayed convention and the worked result:
   `x₁=(0.9,0)`, loss `0.405`. Explain why the high-curvature coordinate stops
   after that update.
2. Use **Play**, **Pause**, **Previous step**, **Next step**, the inspection
   slider and **Reset playback**. Confirm the selected points agree across the
   trajectory, loss, update norm and table by switching the result view. While
   playing, change a scientific
   control: inspection must pause and return to step zero. Check **Reset to
   baseline** restores learning rate `0.1`.
3. In **Challenge**, commit a stability prediction, probability and reason before
   reveal. Change a defining input and confirm the old commitment is invalidated.
   Commit again, reveal, inspect the score explanation and test a control that
   would contradict your explanation. A wrong forecast is useful evidence; it
   does not prevent inspecting the result.
4. Download the companion exercises under sidebar **Study tools → Start here**. Save them in
   `mastery_hub/units/u02_optimization/`, select your study kernel, and implement
   the Hessian, momentum step, loop and stability reasoning. Use the immediate
   tests and hints. Run a zero-learning-rate or rotated-coordinate control and
   explain what a failure would mean.
5. Before revealing a **changed transfer regime**, state a new prediction and
   why the old rule might fail. Previously inspected settings count as practice.
   Compare the prediction with the actual curve, then revise one causal claim.
6. Use **Optional reflection, saved attempts and replay** to export and replay;
   use **Import an exported JSON attempt** for import. Save progress, export an
   attempt, reload, import and replay it. Confirm the
   scientific inputs and recorded result survive, while the main and replay
   inspection controls remain separate. After reload, the saved prediction,
   confidence and reason must match the locked record, including a nondefault
   “Not strictly stable” answer. Use the text/table alternatives and
   keyboard navigation. Note any clipped labels or inaccessible control.
7. Optionally enable the study timer. It starts disabled. Start, stop, resume,
   disable and reset it; confirm these actions leave the scientific settings and
   prediction unchanged. No calendar event or reminder is created.

Report the first blocked step, confusing instruction, incorrect result, assistance
needed, and whether the complete loop worked. Do not report a pass solely because
the page opened. Preserve negative feedback for repair and repeat the affected
steps after a correction.

## Current evidence

The final v3 candidate passed an independent **119 focused checks in17.58s**;
the earlier compact v2 passed117 focused checks plus seven independent edge
checks. These are source/state checks in the existing Python3.14.2 environment.
See [the exact v3 review](../../.scratch/studiable-20261102/v1-quality/reviews/compact_ui/v3-restore/REVIEW.md)
and [the final verification record](evidence/quality_20261103/final_verification.json).

[Native browser evidence](evidence/quality_20261103/native-ui-behavior.json)
records real Play progression15→21, Pause21, Next22, keyboard result-view
selection retaining22 and Reset0. Challenge displayed zero plots before
commitment and after input invalidation, one after reveal, and restored the
saved nondefault forecast/reason as disabled fields after reload. At708×1052,
the essential controls and entire curve fit; at390×844 there was no horizontal
overflow but the chart required scrolling. [The screenshot](evidence/quality_20261103/optimizer-compact-v2-708.png)
is author navigation evidence. A dark-preference probe retained the configured
light theme; dark-theme qualification is not claimed.

One display limitation remains: immediately after the first Commit render,
the prediction fields can still look editable until the next rerun, although
the immutable saved forecast already controls reveal. The independent review
records that behavior and the successful reload correction. Browser evidence
does not establish your understanding or pass Q001; record fresh learner feedback.

The original ARENA chapters and canonical extension notebooks were preserved.
The timer and playback inspect the existing analytic quadratic trajectory; no
model or GPU run is launched by these controls.
