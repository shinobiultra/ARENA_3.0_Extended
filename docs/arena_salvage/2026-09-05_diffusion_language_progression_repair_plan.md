# 5.5 full diffusion language-model progression: repair plan

Current learner failure: the 51-cell paired notebooks already teach a useful,
live CUDA copy-grammar denoiser with ten implementation exercises, controls,
trajectories and a real play cell. Preserve that useful warm-up. Its vocabulary
is ten integers and its only task copies two prefix digits. It does not yet
cover the selected roadmap's balanced parentheses, arithmetic strings,
TinyStories, templated QA, block/temperature sampling or matched AR comparison.
Its confidence-remasking sampler is an iterative heuristic; it should not be
silently equated with an absorbing-state ancestral diffusion transition.

Inspection completed before editing: both 5.5 notebook surfaces, instruction
page, reference training/sampling code, registry requirements and exact
Extension-Roadmap.md:405-468. Original ARENA 1.1 Transformer from Scratch was
used as comparator, especially its model assembly, training, actual text
sampling and KV-cache exercises. No original files will be edited. This plan
is preparation while the final pilot validations run; no new 5.5 experiment
has been started or claimed complete.

Target section claim: derive and test masked noising and reverse transitions,
train tiny language models on controlled string tasks and a pinned real text
subset, then compare what their generated words, errors and commitment traces
show at matched sampling budgets. Keep natural-text and arithmetic failures;
do not select a model or sampler from final outcomes.

Signature result to create: actual token/word denoising trajectories alongside
an AR continuation, with the input, predicted words, grammar/answer score,
entropy, stable commitment and measured cost. A quality-versus-model-calls and
latency plot must retain every sampler/seed and all relevant controls. TinyStories
continuation is open-ended; reference edit distance is not a correctness score
for valid alternative text, and a diffusion loss bound is not AR perplexity.

Student-implemented functions: target-only noising with an explicit eligible
mask; weighted denoising objective; carry-over parameterization excluding MASK
predictions; exact absorbing reverse probabilities; ancestral and separately
labeled confidence-remasking sampling; temperature schedule; block generation;
AR shifted loss and cached generation; training/checkpoint selection; token
entropy, stable commitment, edit distance, grammar/answer scoring and fair
latency measurement. Core bodies stay visible in the notebook; dataset/tokenizer
I/O and plot layout may be helpers. Reuse original ARENA concepts without
modifying their notebooks.

Controls/baselines: exact finite-state probability and gradient oracles before
training; unchanged conditioning tokens; random-order or shuffled supervision
with the same token marginals; untrained model; three matched training seeds;
all declared denoising budgets; a parameter-matched AR model on identical data
and tokenization, including KV caching for timing. Report forward calls, batch
size, output length, hardware synchronization and warm-up. Do not claim a speed
advantage from comparing a cached diffusion path with an uncached AR path.

Data/protocol: deterministic generated train/development/final splits for
parentheses, arithmetic and context-dependent QA; no supplied answer length or
padding pattern may reveal a held-out answer. Use fixed token blocks or an
explicitly justified EOS treatment. Pin TinyStories source revision and exact
chosen story IDs/hashes; check normalized-text and token-block duplicates.
Fit any tokenizer on training text only. Freeze data recipes, training budgets,
seed set and development selection before final scores. Keep all trials.

Play cells: modify a QA fact and trace when the answer changes; vary sampler
steps/temperature/block width on fixed model weights; compare the same real
text prefix under AR and diffusion. Each play shows actual generated tokens,
not a lookup in a saved result.

Files to modify after pilot validation: this existing section's canonical
lesson/builder, new inspectable methods/data/training modules, semantic tests,
paired notebooks, page, figures and bounded evidence; lock/report/README and
registry only after the full own-answer run passes. Archive the current useful
copy-grammar result if a larger lesson supersedes it. The existing pinned
DiffusionGemma NVFP4 generation proof stays narrowly scoped; do not claim
released-model diffusion patching without a working hookable runtime.

Tests to run: exact forward noising frequencies; weighted-loss expectation and
gradient oracle; reverse-transition normalization and token carry-over; no MASK
output at zero noise; conditioning/padding isolation; same-seed sampling and
per-example EOS; cached/uncached AR parity; matched training update oracle;
dataset disjointness; meaningful counterfactual controls; full own-answer clean
kernel and inspected figures; separate fresh CUDA evidence; preservation,
course/style, scientific evidence and roadmap completeness audits.

Primary-source preparation: [MDLM authors' implementation](https://github.com/kuleshov-group/mdlm)
and [paper](https://arxiv.org/abs/2406.07524v2) provide an implementation check
for the selected masked-diffusion topic. Local read-only source snapshot is
`/tmp/mdlm-source-review`, commit
`c112c526d193436838c98d81455ee51f90309470`. Its reverse transition preserves
unmasked tokens and distributes masked-token mass between remaining masked
and predicted clean values. This is distinct from the current heuristic.
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) is pinned
for preparation to `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`; no subset has
been downloaded or trained yet. The current selected-topic registry status
remains TOY_ONLY. No method coverage has been promoted by writing this plan.

## Development update — real data and sampler diagnosis

The preparation-only statements above are historical. The pinned source files
were downloaded and SHA256 checked, training-only BPE was fit, and duplicate
actual tokenized contexts were excluded across splits before model training.
The v3 data inventory is 5,632 train, 512 development and 512 final rows; model
runners have loaded only train/development. All prior data candidates remain.

The first matched four-mode 1,500-update candidate completed but generated weak
text. A real sampler bug was diagnosed: confidence remasking rewrote already
visible positions using logits unsupervised at those positions. The corrected
confidence/uniform paths preserve visible tokens, as the ancestral path does.
A regression oracle supplies deliberately wrong visible logits. All 28 semantic
tests pass. The old candidate and exact sources remain; all three corrected
samplers were evaluated on the same unchanged weights and development inputs.
QA improves to 24–31/128, while arithmetic and stories remain weak. A second
matched candidate increases all four models equally to 12,000 updates with
checks every 1,000. No final-based tuning, new easier dataset or scope promotion
has occurred. Expanded canonical learner publication remains pending.
