# [9.4] White-box Evals and Monitors

This section asks one narrow question: can internal evidence detect a planted
CoT-unfaithfulness state when its output is exactly identical to a clean state?

The learner path starts from a fully declared safe model organism with six named
features, five training contexts, and four disjoint held-out contexts. Students
implement the activation matrix, context split, mean-difference monitor, AUROC,
threshold sweep, output-only baseline, audit dashboard, negative controls, and a
named-feature intervention directly in the notebook.

The visible signature result is generated from those implementations. It shows:

- the exact planted feature matrix;
- held-out white-box and output-only scores for every failure kind;
- four CoT-unfaithfulness failures caught only by the white-box monitor;
- fitted, output-only, shuffled-label, and random-direction AUROCs of
  `1.00`, `0.80`, `0.60`, and `0.20` respectively.

The solved notebook contains all taught implementations inline and executes end
to end. `solutions.py` retains the pinned Pythia-70M CUDA path as a real-model
mechanics preflight. That preflight uses hidden states and next-token logits on
safe generated monitor records; it generates no completions and does not justify
a deployment-monitor claim.

Supporting verification files remain secondary to the learner notebooks. The
course result is the computed figure and auditable caught-example table, not a
JSON report.
