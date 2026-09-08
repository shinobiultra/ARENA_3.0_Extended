# SAGE fixture and execution provenance

## How this fixture was produced

The protocol, manifest and data helper were frozen before any native fit. The own-answer notebook trains six
CPU float64 MLPs and one ridge model from the pinned UCI source, fixes rankings on explanation recipes before
audit loading, and computes every prescribed coalition, sample/control and default play. Published figures
come from those actual producing cells. `learner_execution.json`, `native_summary.json` and
`fresh_reproduction.json` point to full source snapshots and model outputs; reports support the notebook.

## Trusted implementation

Independent permutation enumeration and constrained weighted least squares recover SAGE from actual model
games. NumPy reproduces 64 native network predictions; unmodified author MarginalImputer and utils at commit
0e34e6c7e2a6b1666e39043e83b9929e68ee68f3 reproduce 64 real restrictions. Byte hashes and MIT provenance are in
author_reference.json. Analytic Gaussian/XOR and small hand-computed loss/path fixtures check the student
functions. Explicit nonaliasing output storage protects predecessor masks in the independent permutation
reference.

## Random seed

The frozen protocol records network seeds 161210..161212, matched shuffled-target seeds 161220..161222,
training background hashes with seeds 161260..161262, target permutation 161270, random feature orders 161280,
direct sampling 161290 and 32 repeated schedules 161300..161331. The analytic inner-bias illustration uses
161340. No native outcome chooses a seed, model, background or split.

## Allowed tolerances

Fresh tensor comparisons require identical structures, shapes, dtypes, integers and identifiers, and
rtol=1e-7, atol=1e-8 for floats. All 3,786,320 tensor elements matched exactly in the published run. The
publication gate requires maximum absolute difference at most 1e-10 and all 464 tensors / 61 scalars.
Independent reference checks use stricter operation-specific float64 tolerances; every permutation path must
conserve its endpoints.

## When to regenerate

Regenerate after changing any study or learner source. Reexecute the complete own-answer notebook, run a fresh
independent training/reproduction, inspect every changed figure and the browser math/dropdowns, and publish
only the exact reviewed source/output pair. Then run the normal report and course gates. Do not change the
byte-frozen protocol/data retrospectively; preserve old attempts and explicitly declare any new experiment.

Full native .pt payloads stay in the cache with checksummed external-artifact manifests. Readable source and
execution/visual evidence are archived here and in development_evidence. Failed own-answer v2 and diagnostic
traces are retained.
