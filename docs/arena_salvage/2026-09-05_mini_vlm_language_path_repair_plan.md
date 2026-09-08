# Complete the Mini VLM image-to-language pilot

## Current learner failure

The 55-cell paired 12.3 notebooks and instruction page were inspected alongside
original ARENA 1.1 Transformer from Scratch (module tests, causal masking,
training and generation progression). The current lab has useful explicit
visual rows, a learned connector and two causal attention blocks. However:

- Its vision features are engineered mean RGB and non-white occupancy, not
  the patch embedding plus ViT or frozen OpenCLIP requested by the roadmap.
- A whole question is one categorical ID. The decoder predicts one of seven
  answer classes, with no word-token sequence, teacher-forced captioning or
  autoregressive text generation.
- Several solution dropdowns have broken Python indentation. The final
  "Try It Yourself" is only a Markdown fence, not an executable play cell.
- Training/evaluation reuse the same fixed geometry and labels with a style
  change. The current whole-path examples do not establish transfer to
  independent rendered instances or held-out attribute combinations.
- The "random region" is explicitly sampled from non-object tokens. That is
  a useful matched background null, but is not an unrestricted spatial null.

The roadmap passages are 3692–3758 and 6277–6332. The current registry's TAUGHT
row is broader than the implemented word-level learning/generation path.
Do not use the existing report to claim those missing methods are complete.

## Target section claim

Students build a small image-to-language model from RGB pixels: learned patch
embeddings, a bidirectional vision transformer, a connector, explicit image
and text token positions, causal decoder blocks, teacher-forced caption/VQA
losses, and greedy autoregressive generation. They then causally test which
visual information changes actual generated words.

Keep the RGB/occupancy and exact additive-patching examples as clearly labeled
warm-ups. Do not relabel engineered features as a trained vision encoder or
a one-token answer head as a captioning language model. Preserve all original
ARENA chapters, and reuse provided rendering/tokenization plumbing where useful.
No additional VLM family or unrelated topic is introduced by this repair.

## Signature result to create

A fixed gallery shows images, word-level questions, generated answers/captions,
truth, and paired counterfactual outputs. Adjacent plots show learning and
transfer across three seeds, both modalities and shuffled-pair controls. A
stage-by-region patching heatmap connects local RGB evidence, contextualized
vision tokens and downstream decoder states to actual answer-word changes.

## Student-implemented functions

Patch embedding with exact pixel-to-row order; bidirectional visual attention;
vision position embeddings and encoder; connector; multimodal sequence and
padding/causal masks; causal decoder; shifted teacher-forced loss masking;
answer/caption metrics; training; greedy decoding and EOS handling; region
indices; exact activation replacement; and stage-aware causal evaluation.
Methods stay in the notebook with sub-function tests, expected outputs, help,
complete working solution dropdowns and interpretation.

## Controls and baselines

Use independent renderer seeds for training, development and final evaluation,
with explicit familiar classes versus held-out combinations/styles. Select
training settings on development only. Show every declared model seed.
Compare joint models to inference ablations and appropriately trained unimodal
or shuffled-pair baselines. Image-only captioning may succeed; the key modality
necessity comparison is answering different questions about the same image.
Do not demand an impossible universal image-only failure.

For patches, retain object rows, equal-size background rows, unrestricted
same-size random rows, matched counterfactual images, and full-sequence
controls. Random rows may include the object; keep those outcomes. A ViT mixes
object information into background tokens, so later background patches need
not be inert. Treat that result as token mixing, not a reason to choose a more
convenient control. At later decoder layers, replacing all *visual* rows is
not equivalent to replacing the whole sequence: question states may already
contain image information. Use a true full-sequence donor to check parity.

## Independent verification

Compare patch embedding to unfold-plus-linear values and gradients; attention
to explicit small float64 forward/gradient oracles; reject future/pad leakage;
verify shifted loss and ignored visual/prompt/pad positions; test greedy EOS
and prefix preservation; compare at least two optimizer updates independently;
and verify exact native single-stage/whole-sequence patching with hook cleanup.
Complete the learner solely from its own dropdowns in a fresh kernel. Run the
actual training/generation/causal experiment, inspect figures and outputs,
then refresh source/figure manifests, report, scope and preservation checks.

## Play cells

Edit a word-level question on a fixed image; compare generated answers with
image/text controls. Change image color/shape and predict the generated word
before rerunning. Select a patch stage and region, inspect the generation and
contrast object/background/unrestricted random/full-sequence effects. Each
activity has an executable cell, not just a suggested Markdown snippet.

## Files to modify

The existing 12.3 lesson surfaces, paired notebooks, section solutions/tests,
README, assets and evidence; add a readable lesson/builder/executor and bounded
vision-language method/data modules as needed. The 8.3 fourth IOI search remains
active independently; do not interrupt it or edit its frozen intervention
source. This document is preparation only: no 12.3 implementation or release
status has been upgraded at the time of writing.
