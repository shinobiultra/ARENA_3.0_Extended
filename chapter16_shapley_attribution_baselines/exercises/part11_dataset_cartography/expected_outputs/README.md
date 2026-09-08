# Reproduction evidence for 16.11

## How this fixture was produced

The published 71-cell lesson is executed in a fresh CPU kernel, completing each
exercise only from its own following answer. Thirty code cells produce twenty
figures, all training traces, selections, fresh parameters and held-out results.
A separate process reloads the pinned UCI source and retrains the full experiment.
The main evidence is 140 real training runs, with two additional separate mapping
checks and two play fits. Figures are computed from these fits in the notebook.

## Trusted implementation

NumPy softmax and population statistics independently check real-map coordinates.
An ordinary torch.nn/torch.optim.SGD loop checks small independent training
cases. Full separately scheduled clean MLP and linear runs check all recorded
logits and final parameters against the independent model-axis implementation.
The fresh complete run compares every saved tensor and metadata value; hashes
bind the learner, source snapshots, images and publication to their execution.

## Random seed

The pre-fit protocol specifies mapping seeds 161100–161104, retraining seeds
161200–161204, exact corruption hashes, every random-subset seed, two logging
conventions and all budgets. Default real play uses 166000/166001. No evaluation
result selects a seed or model. Retain failed and earlier presentation evidence.

## Allowed tolerances

Ordinary and separate SGD parity uses `rtol=2e-5`, `atol=2e-5` for float32 arithmetic;
the actual separate-run maximum absolute error is below 7e-6. Independent
NumPy map checks use `rtol=1e-6`, `atol=1e-6`. Full learner/fresh comparison uses
`rtol=1e-6`, `atol=1e-7`, exact dtypes/shapes/IDs/scalar types; the development result
is zero error over 675 tensors and 41903198 elements, plus 8904 scalar values.

## When to regenerate

Regenerate after changes to any method, data, protocol, lesson, source reference,
executor or comparison code. Re-execute the learner, reproduce separately,
inspect every figure and the rendered math/dropdowns, then publish and refresh
the normal verification report. A formatting change still needs source-bound
evidence. Archive existing evidence first; do not overwrite old execution runs.

The publication JSON files are supporting evidence. The exercises, solutions,
image galleries and controlled plots are the teaching artifact. Large tensor
payloads remain in the named persistent cache, with exact sizes and hashes in
the archive manifests; archives do not pretend these payloads are embedded.
