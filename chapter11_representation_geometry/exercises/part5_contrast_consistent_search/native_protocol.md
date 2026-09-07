# CCS native protocol — v1 design and v2 development grammar correction

V1 was defined before native model access. Its complete development run is
retained at `~/.cache/arena_extended/ccs/native-development-v1` (197.444 s,
all 18 cases, unchanged sources/weights, no model gradients, no final evaluation).
V2 changes only the first development sentence frame from `What a {adjective}
{item} this turned out to be.` to `This {item} turned out to be {adjective}.`
This fixes the incorrect indefinite article before vowel-initial adjectives.
Train, calibration and final strings are unchanged. No site, seed, hyperparameter
or final example selection changes. The next full learner must freshly capture
the corrected development text; v1 results retain the original wording.

## Question and claim boundary

Can an unlabelled consistency fit recover the sentiment explicitly stated by a
customer, and does an independent reader's opinion change which feature it
selects? This is controlled adjective-based classification using actual GPT-2
activations. It is not a benchmark of general truthfulness, latent beliefs or a
replication of the original paper's larger models and datasets.

## Frozen data

`ccs_data.py` defines every row before native extraction. Base reviews: 256 train,
64 labelled calibration, 64 development and 64 final. The train split fits CCS
without its labels; supervised baselines may use those labels explicitly.
Calibration labels orient each fitted reporter and never update its weights.
Final frames and all eight final adjectives are new relative to the other
splits. There are no repeated base review strings between splits.

Each base appears in plain, neutral-reader and distracting-reader variants and
two prescribed wrappers. A distracting reader's positive/negative opinion is
crossed with every base review, so it is exactly independent of the customer's
label. The neutral-reader sentence controls for adding a second reader without
giving an opinion; it is not a token-length-matched control. The 64 final base
reviews yield 512 contrast pairs across wrappers and variants. These are paired
measurements of 64 reviews, not 512 independent semantic examples.

Both full prompts append one candidate answer, ` negative` or ` positive`.
Check exact tokenization and retain all tokens. A prefix without either answer
supplies the full-vocabulary model-output baseline. Scores for the two candidate
answers are conditional on that restricted pair; retain and display the native
full-vocabulary winning token separately.

## Frozen model and sites

GPT-2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, CPU float32,
eager attention, evaluation mode, frozen weights, four CPU threads. Fit reporters
in float64. Extract the last non-padding token in each answer-completed prompt.
The three prescribed hidden-state indices are 4 (input to block 4), 8 (input to
block 8) and 12 (final normalized residual stream). Keep all three sites, both
wrappers and every variant; do not choose a winning final site.

## Fitting and comparisons

Separate training branch means, no variance normalization. Each of the 18
site/wrapper/variant cases uses ten CCS restarts, seeds 0–9, 1000 AdamW steps,
learning rate .01, weight decay .01, full-batch independent objectives. Keep every
restart, and designate the minimum final unlabelled training loss as selected.
Calibrate each reporter's global orientation with the separate calibration
labels, then hold that orientation fixed on development and final. Also show
the explicitly labelled final-orientation oracle as a diagnostic, never as the
primary result. Do not treat nominal CCS scores as calibrated probabilities.

For every case retain raw uncentered CCS, a shuffled-pair CCS fit (minus branch
permuted on training data only, fixed seed 904), paired-difference PCA, a
supervised linear probe on concatenated branches, and 32 random directions.
Random directions have the selected centered CCS weight norm and bias, with
draw seeds 1000–1031. Orient them on calibration using the same rule. The
supervised probe uses training-feature standardization and fixed L2 coefficient
.001 and 150 LBFGS iterations, matching the visible prior lesson's baseline.
Store every fit, orientation, score, loss and example count. Native-error cases
are retained. When an unsupervised baseline reads the target well, preserve that
result; a control need not fail on every representation.

Development diagnoses token alignment, resource use and implementation errors.
Any scientific protocol change must be recorded before final access and must
retain prior results. No final data may select a hyperparameter, seed, site,
wrapper, prompt example or claim.

## Signature and execution

Before first final model access, fix the actual-text gallery to base IDs
`final:0:0:0:0`, `final:0:0:0:1`, and `final:0:0:1:0` (the first three base
reviews in row order). Use customer-wrapper, layer-8 fits for that illustrative
gallery and the default play; every other site/wrapper remains in the full
results. The play's initial text is the development sentence `Having finished
the show, I can say it was amazing.` with both Morgan opinions. These choices
use only already observed development, never final outcomes.

Show loss-versus-target/distractor recovery, every restart, wrapper/site transfer,
raw/centered/corrupted-pair controls, native output probabilities and actual
review texts. The fixed native play changes a review and the other reader's
opinion using an already fitted reporter. Full own-answer execution, independent
fresh model recapture and reporter refits, tensor/output comparisons, rendered
math/dropdown inspection and normal course gates are required before release.

No native CCS result had been observed when v1 was written. At the v2 correction,
only train/calibration/development have run; final native data remain unevaluated.
