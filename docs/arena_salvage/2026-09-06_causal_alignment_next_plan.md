# Completed 11.3 checkpoint

The full native learner described by this plan is now published. The latest
own-answer run is `~/.cache/arena_extended/das/full-own-answer-v3`: 63 cells,
32 executed code cells, nine implementation groups, thirteen immediate tests,
twelve inspected figures and three plays, PASS in 306.270 seconds. Independent
`fresh-native-v4` reproduces every one of 16,512 final outputs across 43 conditions
with zero full-vocabulary error or changed winners. All scientific inputs and
frozen model weights remain unchanged.

DAS seeds attain 71/96, 73/96 and 72/96 changed cross-name role interventions;
whole donors attain 72/96. All 32 random subspaces, shuffled targets, native
errors and the failed prescribed play remain visible. This covers the two
selected causal-abstraction/DAS requirements. It does not cover IIT, uniqueness,
minimality or superiority over whole donors. See the current `HANDOFF.md` for
normal-report hashes, final global gates, archived earlier failures and the next
selected concept-erasure plan.

The original plan and intermediate foundation checkpoint follow as history;
their statements that no native DAS experiment exists are superseded above.

---

# Next selected learner: causal abstraction and Distributed Alignment Search

Selected existing requirements: `representation.causal_probes_das` and
`representation.causal_abstraction`. The roadmap places both after observational
probes. Section 11.2 now gives the concrete motivation: high training CCA and
partial donor-distribution recovery do not guarantee the intended name change.
Do not promote Interchange Intervention Training merely for training an alignment;
IIT changes the low-level model and is a separate selected requirement.

Current learner failure: neither a trainable intervention nor an explicit
high-level/low-level interchange comparison is taught. Ordinary probe accuracy
can stay high on unused or redundant information.

Target section claim: implement an abstraction map, perform the corresponding
high-level and low-level interventions, measure their agreement, and optimize an
orthogonal distributed alignment using intervention outcomes. Distinguish this
from fitting coordinates with CCA and from changing the model's weights.

Signature result to create: an exact commuting-diagram counterexample on a
known rotated organism, followed by held-out GPT-2 name-role interchange accuracy
and actual token-distribution curves versus random/equal-rank/label-shuffled
controls. A trained subspace must change the intended behavior on new examples;
otherwise retain a clear negative result and avoid a discovery claim.

Student-implemented functions: high-level structural equations; abstraction and
interchange operators; orthogonal subspace replacement; differentiable constrained
alignment; paired intervention loss; held-out interchange agreement; specificity
and off-manifold checks. Use exact finite differences for gradient tests and
known rotations for intervention oracles. The core functions belong in the
notebook with immediate feedback, help and own-answer dropdowns.

Controls/baselines: no-op, whole-position donor, random subspaces of equal rank,
wrong intervention labels, a predictive-but-unused feature, ordinary CCA alignment,
training-only fitting, development-selected layer/rank, all new final cases,
multiple seeds, unrelated-variable preservation. Treat baseline model errors and
native-donor failures separately from alignment failure.

Play cells: swap a different high-level variable, rotate the known organism,
change rank or a real recipient/donor pair and rerun the native model. Show
actual tokens and probabilities, not only an aggregate report.

Execution route: reuse pinned GPT-2 CPU weights and the visible native plumbing
from 11.2. Its inference-mode hooks need a separate differentiable implementation
for alignment training; test that gradients reach alignment parameters while
model weights remain frozen. Define fresh final prompt families before model
execution so the new method is not evaluated solely on 11.2's inspected final
set. Settle development behavior before writing the frozen final protocol.

Files to create after inspecting the closest original ARENA IOI section:
a new chapter-11 part3 learner, methods, tests, native differentiation plumbing,
data/protocol and matched notebook/instruction surfaces. Preserve all original
chapters and freeze 11.1's queued geometry sources. Keep all model runs, logs,
source snapshots and large actual tensors under `~/.cache/arena_extended`.

Tests: exact structural equations and abstraction interventions; orthogonality,
zero/full-rank limits and unrelated-variable preservation; independent gradient
oracle; native no-op and donor parity; full own-answer execution; fresh native
model reproduction; rendered dropdown/math/figure review; original preservation.

Primary sources to read in full before implementation:
- Geiger et al., [Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations](https://proceedings.mlr.press/v236/geiger24a.html), 2024.
- Geiger et al., [Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability](https://jmlr.org/papers/v26/23-0058.html), 2025.

This is a next-work plan. No DAS model has been trained, no final dataset has
been opened for it, and neither requirement is promoted by this document.


## Completed exact-model foundation checkpoint

The original 1.4.1 IOI exercise and solution notebooks were inspected before
creating the draft. The primary DAS definitions/objective and experimental
appendix, and the causal-abstraction paper's approximate-transformation,
interchange and feature-learning definitions were read. No claim is made that
all unrelated sections of the 64-page theory paper were reviewed.

New source directory:
`chapter11_representation_geometry/exercises/part3_causal_abstraction_das`.
`lesson.md` has 29 cells / 14 code cells, five exercise groups, eight immediate
mathematical/gradient/training tests, four controlled figures and an angle play.
The complete encoder/readout and all seven method functions are visible.
`build_notebooks.py` creates drafts; `execute_learner.py` executes own answers.
No draft notebook or instruction page was added to course navigation.

Own foundation v3 at `~/.cache/arena_extended/das/own-foundations-v3` passes in
4.083 s with all six inputs unchanged. Its four figures were inspected (three
are unchanged from inspected v2; the first heatmap layout was improved).
`expected_outputs` archives the development run, first parse failure and both
successful foundation candidates with verified hashes. This is exact-model
evidence only. Both selected full requirements remain UNIMPLEMENTED.

The organism rotates eight coordinates. The readout uses a routing bit and an
independent side bit, ignoring a perfect copy of the routing bit. All 16
base/source bit combinations occur; train has 256 pairs, held-out has 1024 with
fresh Gaussian nuisances. Three rank-one DAS fits (300 updates, .03 learning
rate) attain 100% held-out IIA and preserve the side decision. The unused copy
gets 50% overall IIA but 0% changed-routing IIA. All 32 random rank-one draws
are retained (changed IIA mean .06372, max .59375). Shuffled targets attain
.23633/.34570/.47852 changed IIA and remain visible. A mean direction is also
perfect here; DAS is not necessary for this organism. Learned edits have about
.21-.23 coordinate MAE despite perfect output classes, illustrating the limited
identification provided by output-only intervention objectives.

Next: implement a native differentiable GPT-2 continuation. Freeze model weights
but retain autograd from logits through the patched activation and QR basis.
Clone cached inference tensors outside inference_mode before using them in
backpropagation. Check no-op/full-donor parity, finite differences and hook cleanup
on a tiny native batch before training. A promising high-level task replaces
which of the first two names is repeated: derive the counterfactual answer from
the base names and the source's repetition role. Use donors with different names
to test whether the subspace transfers a role instead of merely copying a name.
Compare the second-subject and final positions on development data. Keep native
baseline errors and whole-donor failures separate; never filter final examples
based on learned success. Define new final templates before any DAS model
access. No pretrained DAS model or final prompt set has been run yet.
