# [7.2] Feature Verbalizers

This section teaches the complete explain, predict, falsify, and revise loop on
an exact planted feature before escalating to a real transformer direction.

- `7.2_Feature_Verbalizers_exercises.ipynb` contains seven graded exercises.
  Students implement the planted oracle, example selection, executable semantic
  rules, counterexample mining, controls, intervention scoring, and final result.
- `7.2_Feature_Verbalizers_solutions.ipynb` mirrors the learner progression with
  every taught implementation inline and an executed three-panel signature
  result plus the full held-out prompt table.
- `tests.py` gives immediate semantic feedback and enforces notebook structure,
  visible toy ground truth, inline solutions, and parseable code cells.
- `solutions.py` is the importable reference and contains the serialized CUDA
  escalation; it is not the learner-facing implementation surface.
- `artifacts.lock.yml`, `expected_outputs/`, and `verification_report.json`
  preserve supporting release evidence. They are not the lesson.

The exact CPU result uses 12 training, 6 revision, and 24 final held-out prompts.
The revised verbalizer reaches 1.0 held-out and contrastive accuracy, while the
initial text-only story reaches 0.542 and the base-rate, random-keyword, and
lookup controls each reach 0.667. Adding the planted feature direction changes
the score by 1.25; an orthogonal direction changes it by 0.0.

The real-model handoff loads the pinned
`NeelNanda/GELU_1L512W_C4_Code` checkpoint through TransformerLens on CUDA,
constructs a residual direction from safe final-token activations, validates a
held-out keyword explanation, and compares target-direction steering with an
orthogonal random direction. That evidence must be rerun by the parent CUDA pass
after learner-facing changes.
