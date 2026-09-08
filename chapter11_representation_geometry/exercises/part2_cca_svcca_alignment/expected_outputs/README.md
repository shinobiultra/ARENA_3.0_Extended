# Actual alignment evidence

The lesson displays results from student functions and actual model interventions.
These files support that lesson; they are not the learner's main output.

## How this fixture was produced

`execute_learner.py` completes only the notebook's own solution dropdowns and runs
every cell in a fresh kernel. It snapshots eleven inputs before execution. GPT-2
uses the pinned revision in `alignment_protocol.json`, FP32 eager attention and
four CPU threads. The complete learner retains all final examples, candidate
ranks and 16 random draws per pair. Its large actual tensors stay in the named
persistent run directory, with hashes in the accepted execution record.

## Trusted implementation

The mathematical tests use independent known-covariance, exact linear-map,
rigid-transform, primal ridge and categorical-distribution oracles. The fresh
CPU path reloads GPT-2, recaptures native activations, refits every training map
and repeats all conditions on the first and last final batches. It compares
full-vocabulary logits and winning tokens. No reference module supplies hidden
answers to the student's implementation cells.

## Random seed

Synthetic checks use 801–805; visible experiments use 811, 820–824 and 831.
Training-pair permutation uses 910. Sixteen same-norm directions use 941 plus
100 times the layer-pair index plus draw index. The live play uses 987. Templates,
names, split sizes and rank selection are fixed by the protocol and source.

## Allowed tolerances

Whitening/canonical/transport mathematical checks use rtol=1e-10 and atol=1e-10
on FP64 exact oracles; KL uses 1e-12. Native no-op logits must match exactly.
Fresh CPU comparisons require no changed winning tokens and maximum full-logit
error at most 1e-5. The saved learner scores must match actual full-vocabulary
logits within 1e-7; the accepted run's observed error is recorded separately.
These bounds do not certify successful causal transfer; failed transfers stay.

## When to regenerate

Regenerate a full own-answer learner and fresh model comparison when any declared
source, model revision or protocol changes. Create new persistent output
folders. Never tune ranks using final outcomes. Retain failed runs and source
snapshots. Publish through `publish_study.py` only after both execution and fresh
comparison pass, and inspect all nine images before accepting the section.
