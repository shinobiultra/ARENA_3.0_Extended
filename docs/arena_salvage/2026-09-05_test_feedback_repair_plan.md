# Visible test feedback and fixture provenance

Current learner failure: seventeen hard-exercise findings remain across 6.3,
7.3/7.5, 8.2/8.3, 9.3, 10.1, 12.1/12.3, 16.7/16.8, and 17.1. Some passing
tests give no immediate feedback. Several notebook-completeness tests inspect
exception-name strings, missing empty or ellipsis-only implementations. Five
fixture READMEs omit precise provenance. Their notebook pairs, tests, experiment
entry points, and existing fixture notes were inspected; original ARENA's
immediate sub-function tests remain the comparator.

Target section claim: preserve each existing scientific claim while making
sub-function feedback immediate and fixture provenance traceable. This batch
does not promote toy or preflight evidence into completed roadmap coverage.

Signature result to create: no new scientific result. Retain existing live
graphs, retrieval/patching matrices, intervention curves, and training figures;
refresh their supporting reports only after the changed checks pass.

Student-implemented functions: all existing taught functions remain local to
their notebooks. A shared AST checker is maintenance plumbing that recognizes
empty, pass-only, ellipsis-only, and explicitly unimplemented function bodies.
It does not grade scientific behavior; the existing numerical oracles do that.

Controls/baselines: preserve every existing numerical and causal control. Test
the checker on unfinished and completed functions, including nested methods,
and confirm prose mentions cannot masquerade as executable learner work.

Play cells: retain each current parameter/intervention experiment.

Files to modify: affected section tests and five expected-output READMEs;
shared notebook checker and its tests; explicit report dependency entries in
affected locks; refreshed reports. No original ARENA edits.

Tests to run: checker unit tests; all affected section tests; hard-exercise and
style audits; serialized CUDA report refresh; freshness and preservation audit.
