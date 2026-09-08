# [9.2] Chain-of-Thought Faithfulness Verification Assets

Generated support files for the roadmap verification contract.

- `artifacts.lock.yml` pins the toy contract plus the Pythia-70M CUDA preflight.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records the expected Pythia hidden-answer
  metrics and control slots.
- `9.2_Chain_of_Thought_Faithfulness_exercises.ipynb` is the learner notebook.
- `9.2_Chain_of_Thought_Faithfulness_solutions.ipynb` runs the visible tests and
  asserts the committed CUDA verification report fields.

The graded local path loads pinned `EleutherAI/pythia-70m-deduped`, extracts
hidden states on safe A/B private-answer prompts, trains a thresholded
hidden-answer direction, checks held-out hidden-answer prediction, compares
against visible-rationale/text-only and label-shuffled controls, patches hidden
state through the LM head, and records measured VRAM. It uses hidden states and
answer-token logits only; it does not generate completions or claim a broad CoT
faithfulness benchmark.
