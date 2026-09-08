# 11.3 Causal Abstraction and Distributed Alignment Search

`lesson.md` contains the complete developing learner: nine implementation
groups, 13 mathematical/native-plumbing tests, independent pretrained gradient
checks, 12 controlled figures and three play experiments. The core structural
interventions, QR subspace swaps, native continuation, observational probe and
minibatch alignment optimizer are visible, with immediate tests, help and
solution dropdowns.

The exact model demonstrates a perfectly predictive feature whose intervention
does nothing. The pretrained GPT-2 study transfers which of two people repeats
between prompts with disjoint names. It uses a development-selected rank-eight
subspace before block 9 at the final token, three fitting seeds, three matching
shuffled-target fits, 32 random subspaces, observational baselines and whole
donor replacement. All 96 final prompts and 384 intervention pairs remain.

The first complete own-answer run passed in 292.081 seconds; its independent
fresh run reproduced all 16,512 final outputs across 43 conditions with zero
full-vocabulary logit difference and zero winning-token mismatches. DAS gets
71/96, 73/96 and 72/96 changed cross-name interventions correct. Whole-position
donor replacement also gets 72/96. The lesson makes no superiority, uniqueness,
minimality or full causal-abstraction-theorem claim. Model weights stay frozen;
this is not Interchange Intervention Training.

The published student and solution notebooks contain 63 cells and 12 actual
figures. The final own-answer run passes in 306.270 seconds; its fresh native
reproduction passes in 47.099 seconds with zero logit differences. Browser review
verified equations, numbered exercises and help/answer dropdowns. All twelve
final figures equal the inspected previous render. Normal
verification explicitly uses CPU, including on hosts whose GPU is occupied:

```bash
CUDA_VISIBLE_DEVICES='' uv run python scripts/run_extension_verification_reports.py --section 11.3
```

The initial normal-report attempt failed while the generic runner initialized
CUDA, before any section test or native reproduction. That resource failure is
retained separately; the section's scientific experiment uses CPU throughout.

Run the complete learner from its own answers using:

```bash
uv run python chapter11_representation_geometry/exercises/part3_causal_abstraction_das/execute_learner.py --output-directory ~/.cache/arena_extended/das/my-full-run
```

Use a new output directory and the pinned GPT-2 checkpoint in the Hugging Face
cache. The full-vocabulary tensors are several GB; keep them in the persistent
output directory, outside the repository. Allow about four hours to implement
and interpret the lesson; the reference experiment runs on CPU in about five
minutes. `build_notebooks.py --draft-output DIRECTORY` writes unexecuted paired
draft notebooks. `das_reproduction.py --learner-directory DIRECTORY
--output-directory NEW_DIRECTORY` independently checks every final condition.
