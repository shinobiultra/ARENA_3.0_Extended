# Maze Transformers and World Models

Implement an exact BFS oracle, causal next-action transformer, state probes and residual patches.
The paired notebooks retain the complete legacy lesson and all saved outputs, including the failed
stronger internal-search and route-specific interpretations. Added forecasts and reflections are blank.

The CPU signature uses260 training layouts,70 held-out layouts,10 epochs and width56. Its path metrics
use the first60 held-out layouts; probes use state prefixes capped at1200 per split. The full maze and
updated current cell are provided every step. There is no learned transition model or latent planning.
The larger historical CUDA configuration failed its original patch/control gates. CPU and CUDA prefix
cohorts differ; they are separate historical configurations rather than a scaling comparison.

The original student implementation stubs, immediate tests, all prose and dropdown bodies, the six-panel
signature, raw/constrained/template behavior baselines, random-label and Manhattan probe controls, signed
patch controls and original play cells remain inspectable. Pure constants and record classes were made
visible in setup; reporting wrappers appear in a nonexecuting appendix and generated Python.

Historical files have historical_* names, including expected_outputs/historical_reference_metrics.json
(protocol maze-14.4-policy-controls-v2 dated2026-09-05T01:03:24.991679+00:00). Those exact bytes retain
original identity, failures, input hashes and accepted:false. No active canonical acceptance report exists.

This migration performs source-only conversion and static preservation checks. It runs no model, taught
method, numerical test, data/cache access, GPU or cluster work. No new numerical validity or learner
acceptance is established. Topology reconstruction, adjacency-head analysis, broader world models and
human walkthrough remain open. Public worked outputs are exposed guided practice, not unseen mastery.
