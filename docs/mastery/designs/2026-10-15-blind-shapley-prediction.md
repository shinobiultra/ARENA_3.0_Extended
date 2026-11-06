# Additive blind Data Shapley prediction, 2026-10-15

Status: authorized exploratory implementation; independent review and human learner acceptance remain distinct. Uses research-workflow collaboration and the ARENA salvage standard.

Current learner failure: the reference-rich 16.9 notebook supplies its baseline answers before a forecast. Its denominator suggestion was not executed; no subset-mean outcome has been computed or inspected at this design checkpoint.

Target question: **Δφ7 = exact Data Shapley of player 7 under subset-mean training minus fixed-n training**, measured in mean-validation binary-cross-entropy nats. A finite numeric
forecast has no artificial [-1,1] bound. This changes the learning algorithm/game, not the quality of a Shapley estimator. The eight training and eight validation rows,
labels/player order, zero initialization, 40 GD steps, learning rate 0.2, float64 CPU and one thread are fixed. For nonempty S, only denominator 8 versus |S| changes; empty S
leaves zero initialization.

Signature result: paired credit bars for all eight examples plus player 7's mean marginal contribution by predecessor coalition size. Their unweighted mean equals φ7. LOO from cached coalition tables is a baseline, not another training run.

Student methods: vectorized coalition GD, validation utilities, exact factorial-weighted Shapley, marginal-by-size, and visible result assembly. Each has a different small practice
fixture and independent autograd/permutation checks before commitment. Target data may be inspected but training/utility evaluation waits until after the immutable forecast.
Reference implementations are available only through a postcommit helper call; adopting changed source needs a new commitment.

Controls: full-coalition weights/utility equal across denominators; empty and zero-learning-rate models/utility/credits zero; duplicate 0/1 symmetry; exact efficiency; actual label
0↔6 relocation in both policies swaps their credits and leaves others unchanged. Players 0/1 are exact duplicates; 6 has the same features and opposite label; 7 is the rare
positive. All controls run after commitment. The label relocation uses four nonzero-LR coalition games total (1024 fits × 40 batched updates); two additional all-coalition zero-LR
checks are bounded in the same run.

Records: separate schema `arena-blind-shapley-prediction`, version 1, strict bounded JSON with duplicate-key and nonfinite rejection. A sealed immutable commitment binds exact
case, raw ordered tensor hashes and shape/dtype, zero initialization, exact method source, helper/test hashes and environment. Stale identity is rejected before any method
execution or training. A machine-produced receipt contains the unchanged commitment, all ordered coalition utilities/final states, Shapley/LOO/marginal results, hashes, checked
controls, forecast error and timing. Replay/plot/export do no training or current-environment validation. Seals detect accidental edits, not adversarial local tampering or prior
exposure. Exposure is unknown/seen/self_reported_not_seen; never auto-unseen.

A separate reflection envelope keeps the unchanged original receipt and its hash alongside causal reflection and a new transfer forecast. Export is explicit, exclusive creation with private mode 0600. The notebook generator protects learner edits and saved execution artifacts; notebook ships blank.

Authority: only new unit/generator/tests/design/scoped evidence files. Original ARENA and existing 16.9 remain unchanged. One aggregate fresh CPU validation ≤120 seconds is
authorized after a recorded engineering forecast; one thread, offline, no GPU, downloads, cluster or new architecture. Finish code and practice fixtures before that run. Preserve
failed attempts and null outcomes. Root performs independent review and integration; human acceptance is deferred.

Queue: SP1 independent scientific/leak/receipt review pending; SP2 human learner walkthrough deferred; SP3 broader scientific adoption/generalization not authorized. All implementation choices above are settled by the lead's assignment and independent audit; no unresolved prerequisite blocks this scoped implementation.

## Pre-run engineering forecast and frozen checkpoint

The numerical target has not been run or inspected at this checkpoint. Engineering forecast: **Δφ7 = +0.010 mean-validation BCE nats**, confidence **0.55**. Reason: dividing by
smaller coalition size increases gradient magnitude in small coalitions, which may increase the rare positive's early marginal contribution; saturation and redundancy with other
positives could instead offset or reverse this. This is an engineering forecast for a fixed toy, not a student answer or notebook prefill.

Both independent pre-run reviews closed their P1/P2 findings. The helper is frozen at SHA-256 `b06426d5c9757328578ccff543ef3cec4cbca7e8fd23ec6a83c4aa6fdc4ed97b`; tests at
`f2c425d213b69a3a414e3b6a9a7dd655f86f75caf87acdbbb8f3e1d5e75aebb5`. The non-target suite passes 53 tests in 2.50 seconds, including exact generated-source execution, input/runtime
mutation detection, failed-table retention, largest accepted source/Unicode writer-reader parity, private export/reflection, replay isolation, and deadline failure with captured
actual states. The blank notebook is current. No source changes are planned after the single target run; additional outcome reruns are not part of this allowance.

## Final-source bounded validation checkpoint

The single authorized target validation passed on the frozen helper/test identities above. It used Python 3.14.2, torch 2.12.1+cu132 on CPU, float64, one thread, offline model
flags and an external 120-second timeout. Fresh paired/control training plus record validation took 0.266659 seconds; the complete runner including source archival, private
exports/reflection, model-free replay and plotting took 0.710816 seconds. It performed 1024 nonzero-learning-rate coalition fits and 512 zero-LR checks; all eleven frozen controls
passed. No failed target attempt or retry occurred.

Spoiler-bearing commitment, receipt, reflection, exact helper/tests/generator/blank-notebook snapshots, source identity, figure and log are preserved under
`docs/mastery/evidence/blind_shapley_validation/`. Source and learner notebook bytes were unchanged throughout. The saved receipt SHA-256 is
`4cd3e38b874b5d84aff4fe92d9ceac2dcda203683908a398ec7bfc9ff5651127`; the paired figure SHA-256 is `7ffd13137713e191ba63b8ca524e16bc79efcae39ed1574714f7cdf9e13c02d5`. The manifest
binds other saved artifacts; `validation.log` is excluded from its in-progress file hash map because it was still being written. Exact numeric outcomes remain outside the blank
learner notebook and this learner-facing summary.

The paired credit/marginal figure was visually inspected: labels, units, both denominator conditions, player ordering, and zero references are readable. Stored observations were
replayed and replotted after numerical method/freeze/run/environment entry points were replaced by raising stubs. Private exclusive receipt and reflection exports read back with
the original nested commitment/receipt bytes intact. No helper repair followed measurement, so execution evidence covers the final helper identity directly.

SP1 independent scientific/leak and records reviews are closed for this increment. SP2 human learner walkthrough remains deferred; SP3 broader scientific adoption/generalization remains unauthorized. This is a reviewable additive engineering result, not a new unseen assessment claim or universal data-value conclusion.
