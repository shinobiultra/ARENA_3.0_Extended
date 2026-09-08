# Method Disagreement Examples

## Agreement case: additive and trained finite game

Exact Shapley and analytic mechanistic scores agree on the trained neural coalition game: Spearman=1.000, top-2 overlap=1.000.
Deleting the top Shapley feature drops the model value by 3.800, above the non-top baseline 0.200.

## Disagreement case: XOR interaction

Ordinary single-feature Shapley has max absolute value 0.000 on XOR, so it misses the mechanism when players are individual features.
Pairwise Shapley interaction recovers the causal pair with value 2.000. This is a tested player-set disagreement, not a visual story.

## Negative control: shuffled trained model

The shuffled-label control fits its own targets but fails agreement: Spearman=-0.800, top-2 overlap=0.000.
