# [16.6] SHAP vs Activation Patching

This learner lab asks one concrete question: when SHAP and activation patching
disagree, did a method fail or did the experiment change its causal players?

The notebooks expose a four-token ReLU organism with exact token and hidden-unit
ground truth. Students implement coalition evaluation, exact Shapley values,
cached-activation patching, strict unit alignment, matched controls, and an
interaction sweep. The primary result is generated visibly in the notebook and
saved as `shap_vs_activation_patching_exact_signature.png`.

The exact result is:

- token Shapley: `[2.20, 1.80, 0.25, 0.00]`;
- token noising patch losses: `[3.40, 3.00, 0.25, 0.00]`;
- token patching credit overcount: `2.40`;
- post-ReLU hidden Shapley and patching: `[1.00, 0.60, 0.25, 0.00, 2.40]`;
- wrong-location effect: `0.00`;
- shuffled-label cosine: `0.515`;
- matched-random p95: `2.166`, below the true gate effect `2.400`.

`verification_report.json` is supporting release evidence and becomes stale
when learner or solution inputs change. The repository report runner must
regenerate it on CUDA before release; the notebook never loads it as teaching
content.
