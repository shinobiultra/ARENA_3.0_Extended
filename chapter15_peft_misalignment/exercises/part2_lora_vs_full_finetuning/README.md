# [15.2] LoRA vs Full Finetuning

This section is a real-transformer ARENA lab, not a linear PEFT proxy. Students
insert LoRA into pinned TinyStories-1M GPT-Neo attention projections, train two
adapter ranks and a matched full-model baseline on the same safe codebook task,
then test whether similar held-out behavior is carried by different weight and
activation mechanisms.

Learner-facing files:

- `15.2_LoRA_vs_Full_Finetuning_exercises.ipynb`
- `15.2_LoRA_vs_Full_Finetuning_solutions.ipynb`
- `../../instructions/pages/02_[15.2]_LoRA_vs_Full_Finetuning.md`

Verification files:

- `tests.py` contains immediate semantic tests for every core method.
- `solutions.py` mirrors the visible notebook implementation for reproducible
  CPU and serialized CUDA execution.
- `artifacts.lock.yml` pins the model, revision, task, controls, and claim.
- `expected_outputs/` records exact toy and bounded reference expectations.
- `verification_report.schema.json` defines the parent-generated CUDA report;
  the committed report binds its measurements to these files by content hash.
  A report supports the learner result; it is not the lesson.

The signature panel compares held-out behavior, actual selected-transformer
weight spectra, layer-6 activation drift, and causal projection ablation. It
must show base, rank-1 LoRA, rank-4 LoRA, full finetuning, random-label LoRA,
and a same-norm random low-rank update. An unrelated TinyStories next-token
replay task measures capability damage.

Claim boundary: the task tests generalization to disjoint prompt templates for
the same ten subjects and marker rule. It is not evidence that PEFT matches
full finetuning on broad language-model adaptation, unseen entities, alignment,
or downstream benchmarks.
