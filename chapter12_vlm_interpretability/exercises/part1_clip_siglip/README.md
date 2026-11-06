# [12.1] CLIP and SigLIP from Scratch

The full lesson builds image–text similarity, CLIP and SigLIP losses, a tiny dual
encoder, pretrained retrieval and joint-space analysis through visible student
implementations and immediate tests. Start with the losses and tiny-model parts;
the pretrained comparison has its own checkpoint and data requirements.

Open [the lesson reader](http://127.0.0.1:18505/?section=12.1&part=1), or use
[the exercise notebook](12.1_CLIP_and_SigLIP_from_Scratch_exercises.ipynb).
The matching solutions notebook retains historical outputs, rather than proving
a new run in your current environment.

## One guided checkpoint: what changes when distractors arrive?

With the matching-pair logit held fixed, will adding frozen distractor captions change its
softmax probability while leaving its pairwise sigmoid probability unchanged?

Start with the [guided prediction workbook](../../../mastery_hub/units/extension_predictions/exercises.ipynb#section-12-1)
before viewing reference results. Record your forecast, confidence, reasoning and prior
exposure. Then compare the controls and commit a changed-case prediction.

Open [the exercise notebook](12.1_CLIP_and_SigLIP_from_Scratch_exercises.ipynb) to implement the methods and use its
immediate tests. Follow the notebook's setup and runtime instructions in your ARENA
environment. Run the small play cells separately from any training or model-loading cells.
[The solutions notebook](12.1_CLIP_and_SigLIP_from_Scratch_solutions.ipynb) contains worked code and available result evidence.

## Methods and controls

The visible implementation includes `similarity_logits`, `zero_shot_probabilities`.

- original candidate set
- shuffled caption labels
- fixed matching-pair logit
- top-1 retrieval

## Evidence and limits

Analytic fixed-logit normalization comparison. Optional learned embeddings are historical;
monitored training evaluation is not independent TEST.

- Softmax and sigmoid probabilities have different normalization and are not interchangeable
  calibrated beliefs.
- Tiny-model test_images were monitored every 100 training steps; they are not untouched TEST.

Reading a saved result and executing the experiment are different observations. Record
which you used, with its source and configuration. The workbook labels this as guided
practice; historical or already exposed results do not earn unseen-mastery credit.

[Integration and verification record](../../../docs/mastery/continuation_2026-10-16.md).
