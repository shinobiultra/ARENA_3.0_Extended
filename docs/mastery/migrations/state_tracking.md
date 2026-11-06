# Canonical 5.5 state-tracking migration

The complete existing Mamba state-tracking lesson is now authored through
`infrastructure/chapters/chapter5_modern_architectures/master_5_5.py` and published at
`chapter5_modern_architectures/exercises/part5_state_tracking/`, with the matching
`05_[5.5]_State_Tracking_in_SSMs_and_Transformers` instruction page. Original ARENA
and the legacy `part4_mamba_state_tracking` publication remain physically unchanged.

This migrates an existing lesson, not the entire expanded 5.5 syllabus. The visible
learner study remains a one-layer tiny Mamba trained on exact bounded bracket
depths. It does not establish a Transformer/Mamba-3 comparison, permutation-group
study or additional SSM family. The separate old release record contains a tiny
Transformer comparison and pinned 130M extraction preflight; neither is a newly
executed study or the visible notebook's transplanted-state result.

## Original material and identity

`published_before_master.tar.gz` was created before edits. Its SHA-256 is
`6dae34e4d4934fc3f731bdf6cc1ab10a3cb5b52d9a977142e71613e911e778bc`.
It contains twenty exact source/publication files; `before.json` records sixteen
physically unchanged legacy publication/support files and all fifteen historical
report-source hashes, each matching at capture. No trained learner checkpoint or
full continuation-logit cache is among the retained publication files.

All fourteen original solution code cells preserve AST and nonblank source lines,
execution counts 1–14, thirteen output objects and five inline PNGs after only four
setup-route adaptations: canonical section, canonical test import, canonical media
directory, and the exact local prerequisite backbone import. Python-only root
resolution and `MAIN` guards make module import inert; they disappear from the
notebooks. The fourteen exercise cells retain the original student implementations
and stubs after setup routing. All five answer dropdowns contain their own original
solution cell. All original teaching prose is retained with only section formatting,
difficulty-block conversion, image routes and historical-dropdown labeling.

The original exercise generator default is `max_depth=3`, whereas the original
solution default is `4`; both remain intact, and the new prose calls this out.
Every measured call explicitly supplies `max_depth=3`. The old metadata identity
`5_4_mamba_state_tracking` is retained as historical provenance. It does not become
current 5.4, whose registered topic is Mamba-2/SSD.

## Visible methods and supplied architecture

The learner implements or can directly modify task generation, classifier/training,
cache collection, standardized ridge fitting, probe scoring, cache capture, SSM-state
transplantation, equal-norm random edits, continuation, matched-pair construction,
causal scoring, confident-error ranking and error aggregation in the notebook.
Seven original immediate test function bodies are exact; the local tests lazily
resolve the canonical learner module and use a separate scalar recurrence oracle.

`mamba_backbone.py` is precisely ten unchanged definitions, the minimum dependency
closure of `MambaConfig` and `TinyMambaModel` in archived `arena_ext/mamba.py`.
`backbone_provenance.json` binds every complete definition's exact source and AST.
The original module SHA-256 is
`765279bd314b97920844b150c36282d948dfefa2535b3282d46de435da122249`.
This architecture is supplied prerequisite plumbing. It is not a copy of the newer
`part3_mamba_from_scratch` implementation and is not represented as student-authored.
The lesson links the actual existing 5.3 notebook as architectural background;
it does not claim a 5.3 canonical master exists.

Root integration should allow exactly `mamba_backbone.py` as prerequisite plumbing
in this canonical directory. Existing `utils.py` has no implementation. Optional
release wrappers retain their explicit lazy delegation to the original module;
importing the canonical module performs no training, plotting, CUDA discovery or
legacy-module import and preserves RNG/thread state.

## Learner result, controls and play

The preserved signature has four panels: exact latent depth, behavioral ID/OOD
accuracy with majority baseline, probe accuracy with shuffled labels, and the
source/targeted/random future-state trajectories. Other figures show the concrete
bracket sequence, training and position accuracy, the original length-48 play, and
error rates by true depth and position. All five were visually inspected and retain
legible axes, controls and nonempty results. Three original physical assets are also
byte-preserved. The figures are expected HTML image outputs in the exercise
notebook and actual retained image outputs in the solutions notebook.

Seven new forecast/reflection pairs cover boundaries, length generalization,
linear accessibility, causal cache sufficiency, the combined figure, the play and
confident errors. They distinguish behavior's final eight positions from the
probe's OOD-late positions 16–31. The play retains batch128, length48, seed3100,
its final-eight metric and the first position below .8 batch accuracy. Changing
length64 is a proposal, not a newly observed transfer.

The causal pair is one source and one donor of length14; after edit position5,
their convolution history and remaining eight tokens agree. The targeted edit
changes only the SSM state; source convolution state stays fixed. The seed0
Gaussian edit is normalized to the exact donor-minus-source SSM-state norm.
The full donor forward checks cache sufficiency, while cumulative tokens give
exact labels. This does not identify a sparse feature, native depth direction or
population-level effect.

## Execution evidence and preserved failures

No original learner training, twelve-step immediate training test, trained-state
probe study, long-sequence play, pretrained load, GPU study or full notebook was
executed for this migration. The historical output provenance is explicit on each
solution cell and at notebook level. A no-execute repeat build retained the same
code and outputs.

Root authorized one at-most-30-second CPU float32/one-thread primitive subset,
with CUDA hidden, offline dependencies and optimizers/release paths forbidden.
`cpu_primitive_protocol.json` was frozen before execution. The exact pre-run focused
test source is retained as `cpu_primitive_test_initial.py`. Inputs were two bracket
batches of4×8 (seeds0/1), an8×4 literal one-hot probe fixture, four handcrafted
error tokens, one untrained3×9 full/cache comparison and one untrained matched-cache
transplant. The numeric primitive function bodies remain AST-identical to that
freeze, and all three canonical scientific modules still match their frozen hashes.
These checks validate primitives, not the historical trained result.

`pytest-initial.txt` records thirteen passes and two failures in3.63 seconds. Both
failures concerned static presentation assertions: the converted difficulty block
uses quoted YAML, and expected figures are HTML outputs rather than Markdown cells.
The actual prose/figures were present. Assertion-only corrections yielded three
static passes in1.05 seconds, without repeating scientific checks. A subsequent
static proposal assertion omitted the word `transplant` from the preserved output
label; that failure is retained, and the corrected exact-label check passed in
1.01 seconds. No numerical threshold, target or algorithm changed. The first
converter failure (exercise heading placement) is also retained alongside the
successful initial and repeat build logs.

## Guided proposal and acceptance boundary

The strict twenty-key `proposed_case.json` remains evidence-only until root adopts
it. It asks for targeted minus matched-random donor-trajectory match over exactly
eight suffix positions, using original notebook cell13 / code ordinal8. Printed
match fractions0.000 and1.000 are exact at denominator8; rounded probabilities,
probe accuracies and behavioral figures are not reconstructed as exact floats.
Source/output/figure identities are pinned in `proposed_case_identity.json`.

The binary transfer changes donor delta strength from1 to.5 using the visible cache
type and continuation helpers. Its random control is constructed against the
half-donor cache with the same seed, hence matches the half-strength norm. The
original `run_state_transplant` has no strength parameter. The transfer has not been
executed or inferred. Missing trained weights require a separately frozen and
authorized reconstruction before observation; a new primary replication would
remain distinct from the historical primary.

Historical reference exposure remains seen/unknown guided practice. Source checks,
CPU primitives and independent review do not supply human learner acceptance or
finish the expanded syllabus. The learner walkthrough remains deferred.


## October 19 proposal-only correction

Removed the actual historical target match values from precommit metric metadata,
retaining denominator eight and discrete resolution 1/8. The separate proposal
identity and generator use the same answer-free wording. Prior proposal/identity/
generator bytes are sealed in `pre_resume_20261019/`; fourteen historical cells
and results remain untouched. The proposal schema/identity/leakage checks were
included in the 15-pass structural selector; no historical study or primitive
was rerun. Independent proposal-only review precedes root adoption.
