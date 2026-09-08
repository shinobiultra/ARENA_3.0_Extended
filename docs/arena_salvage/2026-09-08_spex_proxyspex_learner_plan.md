# SPEX and ProxySPEX: learner plan before native outcomes

This continues the already-selected `shapley.spex_proxyspex` requirement. It
remains UNIMPLEMENTED until both methods and their actual LLM comparison have
passed execution, independent reproduction and learner review. No new topic or
model family is being added to the selection.

## Current learner failure

The existing course has no learner-visible sparse Fourier recovery or boosted
tree spectrum extraction. The completed 16.17 lesson estimates pairwise SII in
an eight-feature oracle. It cannot show how sparse higher-order recovery avoids
enumerating a much larger feature space. The source preflight has now executed
both actual author methods, including meaningful failure cases; that evidence
is preparation, not a lesson or a native LLM result.

Before this plan, root inspected the original 1.1 notebook pair and instruction
page, and 16.17's pair, page, lesson and builder. The original introduction,
objectives, numbered learning arc, difficulty/importance/time blocks, immediate
tests and adjacent help/solution dropdowns are the writing model. Original
ARENA files remain immutable.

## Target section claim

Use two connected notebooks so each has one concrete question:

- **16.18 — Find the Interactions Without Trying Every Subset.** Can a few
  carefully chosen groups of model queries identify a sparse signed spectrum?
- **16.19 — Explain the Proxy, Check the Model.** When does extracting the
  exact spectrum of a boosted-tree proxy explain the queried model accurately?

The first teaches SPEX's hash bins, shift signatures, error correction and
iterative recovery. The second teaches ProxySPEX's actual boosted-tree route,
exact extraction, coefficient selection and refinement. Fourier coefficients
and Shapley interactions must remain explicitly distinct. Their common native
comparison will query GPT-2 on fixed word-mask games; it will not enumerate the
full power set or present logical cached calls as direct inference timings.

## Signature results to create

16.18: a visible collision becoming a decoded signed interaction, recovery
traces against known sparse truth, error versus requested/unique query cost,
and actual word-labelled GPT-2 interaction spectra beside held-out prediction
curves. Dense spectra and incorrect degree limits should fail visibly.

16.19: separate plots for model-to-proxy error, proxy-to-spectrum extraction
error, and truncation/refinement error; a tree path becoming signed Fourier
terms; then common GPT-2 mask-audit comparisons against SPEX and query-matched
simple baselines. Preserve and explain author truncation failures instead of
silently calling corrected behavior the original implementation.

Every headline curve needs its exact-truth, constant, additive, shuffled or
misspecified comparator. A sparse-support recovery plot needs signed truth and
false-positive/missed-support information, not only aggregate RMSE.

## Student-implemented functions

16.18: Boolean characters and the normalized fast Walsh transform; GF(2)
subsample/shift queries; systematic BCH encoding and syndrome interpretation;
error-locator recurrence and root search; confidence-guided candidate search;
signed amplitude/bin-consistency checks; iterative collision subtraction;
query accounting; Fourier-to-SII/Shapley conversion. Finite-field lookup tables
and model loading may be supplied as plumbing, but decoding and recovery must
be visible and editable. Group closely related short functions into exercises
with immediate sub-function tests; do not hide the algorithm in imports.

16.19: residual-based boosted regression trees using a documented tree
primitive; exact recursive path-indicator Fourier expansion including repeated
feature splits; ensemble accumulation; energy-based support selection with a
stated boundary convention; training-only cross-validated coefficient
refinement; separate stage-error accounting. Actual LightGBM author/reference
runs are comparators, not replacements for notebook-owned implementation.

## Controls and baselines

Analytic controls: additive, signed parity, sparse polynomial, dense spectrum,
degree misspecification, bin collisions, parity corruption, and a shuffled
response control. Small exhaustive games establish signs and conversions;
large sparse games establish scaling without an exhaustive value table.

Native controls: common held-out word masks, constant and additive predictors,
shuffled training responses, and fixed subject-only counterfactual prompts.
Report all predeclared prompts, including incorrect unmasked model answers and
poor explanations. Native approximation fidelity is separate from semantic
correctness and from a causal account of natural language.

## Play cells

16.18: edit the support/sign, collision pattern or degree budget and watch the
decoder; edit a native prompt index/query budget and inspect recovered words.
16.19: edit a tree path or retained-energy setting, then compare reconstructed
responses and extracted terms; edit a native prompt/budget and separate the
three error stages. Plays must call the student's functions and generate
figures, with exact cache reuse and extra query counts disclosed.

## Files to modify

Only new development directories `part18_spex` and `part19_proxyspex` under
`chapter16_shapley_attribution_baselines/exercises`, this plan, and a prospective
native protocol in the first directory. Each directory will own its method
module/tests, prose, cell definitions and notebook builder. Shared native
inputs, mask oracle and study harness belong to 16.18 and must have a documented
interface. Canonical pairs/pages/assets and existing shared registries are
publication work after execution and review, not preflight artifacts.

## Tests to run

All Boolean basis functions through five features; independent finite
differences for attribution conversion; complete bounded BCH cases and actual
author parity; shift/subsampling identities; collisions, negative amplitudes,
misspecified degree and finite-budget failure; exact tree enumeration versus
extracted spectra, repeated feature splits and pure parity; stage-separated
author and maintained-package comparisons with exact source identities.

Before any native results, freeze prompts/model revision/masking/scalar output,
seeds, held-out masks, query budgets, selection rules, resource caps, output
inventory and failure reporting. Then execute every own-answer learner cell,
repeat from source with no fitted learner input, inspect actual figures and
rendered equations/dropdowns/plays, run registered section/integration gates,
and recheck original preservation. Never promote the joint requirement from
analytic tests or an import-only reference.

## Scope and limits fixed before writing

The native task is a bounded GPT-2 word-mask game, not a claim of frontier-model
or thousand-token native performance. High-dimensional scaling can be shown on
analytic games with known spectra and must be labelled accordingly. Fixed
position masking is a defined model intervention, not natural deletion. The
student must be able to explain where sparsity, low interaction degree and
proxy fidelity enter; weak results stay visible.
