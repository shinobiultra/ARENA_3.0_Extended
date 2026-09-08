# [12.3] Expected learner outputs

These fixtures describe the actual image-to-word lesson. The generated
notebook galleries and plots are the main result; raw evidence supports
rerunning and checking the experiment.

## Exact warm-up

The RGB/occupancy encoder produces 36 ordered four-feature rows. Its weights
are fixed because those features are engineered. An additive contribution game
places known clean/donor evidence in object rows. Replacing those rows flips the
signed answer margin exactly; two same-size background-only draws preserve it.
Those warm-up draws are not described as unrestricted spatial controls.

## Word model and independent checks

The main model has 127,392 parameters: learned 4×4 RGB patch embeddings, two
bidirectional vision blocks, a connector, explicit image markers, word
embeddings, two causal decoder blocks and a 24-word unembedding. Its actual
answers include word tokens and EOS; captions contain multiple words.

Independent float64 oracles check explicit pixel windows, attention values and
gradients, position/marker order, future/pad masking, shifted loss, native stage
interventions, ragged EOS generation, answer leakage and two full AdamW updates.
These tests run on the learner's function/class bodies rather than treating an
imported model as the student implementation.

## Training and final generation

Keep seeds 0/1/2 for joint, text-only, image-only and shuffled-pair training.
Each receives 600 AdamW updates, batch 48, learning rate .003, weight decay
.0001 and norm clipping at 1. Evaluate development every 100 updates and keep
the earliest maximum exact generation score. Every checkpoint history and
seed remains available. Use 288 training and 108 development images, with
both VQA questions and caption requests on each image.

Freeze the twelve choices before final evaluation. The 144 familiar solid and
144 familiar muted images share geometry/noise; the muted palette changes.
The 96 compositional muted images contain red triangles, blue circles and
green squares, which are excluded from all fitting and checkpoint selection.

The measured three-seed CPU means are:

| Final condition | Joint complete sequence | Joint color | Joint shape | Joint caption |
|---|---:|---:|---:|---:|
| Familiar solid | .8958 | 1.0000 | .8519 | .8356 |
| Familiar muted | .8580 | .9907 | .7917 | .7917 |
| Excluded combinations, muted | .3275 | .9028 | .0799 | .0000 |

All trained controls are visible in the figures. Familiar-muted exact sequence
means are .2593 text-only, .3133 image-only and .2670 shuffled-pairs. Image-only
models can succeed on one request type; without question words they cannot
reliably distinguish all three requests. Do not require every image-only answer
to fail. Do not retune models from the excluded-combination failure.

## Causal stage and region study

For every model seed, use the first four instances of every familiar combination
in the fixed final recipe and both VQA questions: 72 matched pairs. Cycle only
the queried attribute. The object mask is the union of occupied clean/donor
pixels. At each stage retain object, eight background, eight unrestricted
random, all-visual and full-sequence conditions: 285 experiments in total.

All pairs remain recorded. Conditional targeted flips use the joint-correct
endpoint denominators 50/72, 51/72 and 45/72. The object-input patch reproduces
donor generation. Background input patches are exactly inert. Unrestricted
random patches may overlap the object and can change answers. After vision
attention, background rows may contain object information. Final-decoder
visual-only patches are structurally inert; full-sequence patches reproduce
actual donor generation at every stage, including incorrect donor answers.

## Provenance and regeneration

`completed_word_study.json.gz` and `learner_execution.json` are written only by
a successful full publication run. The manifest binds all section Python
sources, the canonical lesson, both notebooks and all seven figures. The
validator recomputes reported metrics from raw generated tokens and rejects
missing seeds, altered checkpoint selection, omitted random draws and wrong
conditional denominators. Numerical-only or mixed-source runs cannot publish.

`word_candidates` preserves independent experiments and explicitly scoped
candidate learner executions. It is not the release evidence. No model weight
files are committed. A separate fresh four-mode seed-0 CUDA experiment is
reported as such and retains its own final results, even if they differ from
CPU training. Full regeneration reruns the learner from its own answers before
refreshing the report. Original ARENA chapters remain unchanged.
