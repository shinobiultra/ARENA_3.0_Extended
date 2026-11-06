# Study cautions — 23 October 2026

<!-- study-caution-source:13-2-reference-check:start -->
**Study caution — 23 October 2026.** Start by implementing `causal_patch_sweep(model, clean_labels, corrupt_labels, initial_noise, schedule)` below. The no-argument `tests.test_trained_toy_causal_sweep_beats_preregistered_controls()` runs `solutions.run_toy_experiment`: a green check covers the reference implementation only and does not validate your sweep, even if your stub still raises `NotImplementedError`.

Continue to “Forecast before the result — toy-bottleneck” and the following code cell: `toy_result = causal_patch_sweep(` calls your function. That later call is not a replacement for a missing callback test. Before interpreting the peak, compare your complete heatmap with the same-size random-channel, displaced-location, wrong-time, shuffled-label, unpatched and pixel-patch controls. Which control would expose a sweep that overwrites the object without localizing its cause?
<!-- study-caution-source:13-2-reference-check:end -->

<!-- study-caution-source:14-1-conflicting-signature:start -->
**Study caution — 23 October 2026.** Read all seven positions in the key-swap panel: the third category, “swap: original”, has zero height; the fourth category, “swap: alternate”, has full height. Both retained figures agree internally with the saved tables. Each figure plots seed 0; the saved seed rows are:

| Retained seed | Correct context (2nd) | Original after swap (3rd) | Alternate after swap (4th) |
| --- | --- | --- | --- |
| 0 | 1.0 | 0.0 | 1.0 |
| 1 | 1.0 | 0.0 | 1.0 |
| 2 | 1.0 | 0.0 | 1.0 |

The nearby “Historical asset context” warning preserves a historical axis-reading mistake: the empty third slot was mistaken for the adjacent full fourth bar. Keep that old text and both figures as history, and use the explicit category order above to read them. This is internal agreement of retained outputs, not a fresh scientific validation. Before drawing a causal conclusion, identify the key-only intervention and its non-key-pixel control: what should change, and which zero-difference check would rule out another image edit?
<!-- study-caution-source:14-1-conflicting-signature:end -->

<!-- study-caution-source:15-3-failed-parity:start -->
**Study caution — 23 October 2026.** For a first session, work through “Exercise: exact blockwise NF4”: reconstruct values from the codebook and block scales, then check padding and zero-block behavior before comparing the backend.

Stop before treating the full pretrained comparison below as accepted. Its BF16 dequantized-matmul error is `0.03125 > 0.01`; the gate is unchanged and the full comparison is not accepted. Passing behavior and resident-state checks do not erase that failure. Compare trained adapters with random labels and same-norm random adapters, and keep the measured storage denominator explicit. Which discrepancy comes from arithmetic precision, and which comes from counting different resident tensors? Diagnose that distinction without relaxing the error limit.
<!-- study-caution-source:15-3-failed-parity:end -->

<!-- study-caution-source:17-1-finite-domain:start -->
**Study caution — 23 October 2026.** Start with the complete mod-13 table and its legal Fourier modes, then save and reload checkpoints before measuring their mechanism. This finite checkpoint/Fourier study is available here: compare legal-mode ablations with random-label and same-size random-mode controls.

All 169 input pairs appear in both training and evaluation, so held-out grokking is not established. Stop before calling a late rise in accuracy delayed generalization to unseen pairs. Ask which checkpoint intervention shows that the measured modes matter on this finite table, and what separate held-out split and measurement would be needed to support a grokking claim.
<!-- study-caution-source:17-1-finite-domain:end -->

<!-- study-caution-source:8-6-node-metrics:start -->
**Study caution — 23 October 2026.** Read these curves as summaries of planted node contributions. The visible code uses `full_score = contributions.sum().item()` and `selected_score = contributions[selected].sum().item()`, then `faithfulness = selected_score / full_score`, `completeness = 1.0 - abs(full_score - selected_score) / abs(full_score)`, and `minimality = (contributions[selected].abs().min() / abs(full_score)).item()`. The denominator must be nonzero; the threshold sweep assigns minimality zero for an empty selection. These definitions do not independently test general edge-circuit completeness or minimality.

The same-size control is intentionally conditioned on disjoint nuisance nodes; it is not an unconditional random-graph null. Overlap is not inherently answer leakage in an unconditional null, despite the retained Help wording below. Ask what the comparison establishes when useful selected nodes are excluded from the control's pool. For the intervention background, revisit original [IOI](https://learn.arena.education/chapter1_transformer_interp/21_ioi/) and [SAE Circuits](https://learn.arena.education/chapter1_transformer_interp/22_sae_circuits/), then read [Sparse Feature Circuits](https://arxiv.org/abs/2403.19647) to separate this planted exercise from the paper's model study.
<!-- study-caution-source:8-6-node-metrics:end -->

<!-- study-caution-source:8-7-primary-reading:start -->
**Study caution — 23 October 2026.** Revisit original [IOI](https://learn.arena.education/chapter1_transformer_interp/21_ioi/) for activation and path-patching background, then read the authors' [Circuit Tracing methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) alongside the graph exercises here.

Start by naming the source, target and intervened component in this lesson's planted graph. Which path intervention would distinguish a useful graph hypothesis from a decorative attribution picture? The planted result and the retained position-graph preflight do not establish the full paper method on meaningful model components.
<!-- study-caution-source:8-7-primary-reading:end -->

<!-- study-caution-source:9-3-primary-reading:start -->
**Study caution — 23 October 2026.** Use original [Intro to Mech Interp](https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/) for the intervention background, then read [Language Models Don't Always Say What They Think](https://arxiv.org/abs/2305.04388) as the primary paper connection.

Begin with the exact A/B toy and ask which intervention would make rationale information causal rather than merely predictive of the answer. Keep the retained negative Pythia result and its matched-donor control in view; a clickable paper does not turn this controlled toy into a broad natural-language CoT benchmark.
<!-- study-caution-source:9-3-primary-reading:end -->

<!-- study-caution-source:9-4-primary-reading:start -->
**Study caution — 23 October 2026.** Read [Emergent Misalignment](https://arxiv.org/abs/2502.17424) for the broader paper connection, and revisit original [Intro to Mech Interp](https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/) before interpreting an activation intervention.

Use this lesson's known-onset safe toy to practise the measurement first. Which held-out detector control and behavior-only baseline would disqualify the claimed internal warning? Its benign proxy drift and retained next-token preflight do not demonstrate the paper's harmful-finetuning phenomenon or a broad safety conclusion.
<!-- study-caution-source:9-4-primary-reading:end -->

<!-- study-caution-source:9-6-reading-boundary:start -->
**Study caution — 23 October 2026.** Start with the isolated rank-one affine-write algebra and its controls, then read the authors' [ROME](https://rome.baulab.info/) and [MEMIT](https://memit.baulab.info/) materials to distinguish the course edit from official protocol parity and mass editing across layers.

Keep a reading session to the existing toy algebra and retained outputs. The preserved Run All path loads a pinned model and writes legacy expected outputs; do not use it merely to inspect the results. The existing playground rescores the same fixed matrix: it is not MEMIT and does not train a new multi-token edit. What retention control would be needed before claiming that several edits coexist safely?
<!-- study-caution-source:9-6-reading-boundary:end -->
