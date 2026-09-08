# Influence Functions and TracIn

Which training images helped a prediction, and what changes if we remove them?

Open [the exercises](16.9_Influence_Functions_and_TracIn_exercises.ipynb),
[the executed solutions](16.9_Influence_Functions_and_TracIn_solutions.ipynb),
or [the instruction page](../../instructions/pages/09_[16.9]_Influence_Functions_and_TracIn.md).
The notebook has eleven exercise groups, thirteen student-implemented functions,
immediate checks, eighteen figures and two editable experiments. Allow several
hours for the derivations and investigations. The full reference notebook ran
in about fourteen seconds on the development machine using four CPU threads;
other machines and the first data download may take longer.

Start from original ARENA 0.3 Optimization. Derive weighted logistic objectives,
Hessian-vector products, conjugate gradients, local influence and exact versus
Taylor training-path accounting. Then compare real UCI digit images with
independently retrained leave-one-out models and frozen deletion policies.
The affine ghost-gradient exercise checks actual complete parameter gradients;
it does not claim GhostSuite package integration or tied-token support.

The reference study keeps clean and twelve-flipped-label conditions, 128 training
images, 48 ranking queries and 64 final queries from the separate source-test
writer group. Every prescribed seed, checkpoint choice, perturbation and random
control remains. The final gallery shows cases where a checkpoint proponent is
harmful under actual deletion. Accuracy and prediction loss are both visible.
Neither influence nor TracIn is presented as a Shapley value.

Run from the repository environment (`uv sync` if needed), select its Python
kernel, and work through the exercise notebook. For a fresh own-answer run:

```bash
uv run python chapter16_shapley_attribution_baselines/exercises/part9_influence_and_tracin/execute_learner.py --output-directory /tmp/arena16-9-own-run
```

The output directory must be new. It contains the completed notebook, figures,
source snapshots and all native model states, probabilities and counterfactual
fits. The normal verification runner also retrains the complete study on CPU.
See [the frozen protocol](native_protocol.md) and
[the evidence guide](expected_outputs/README.md) for provenance and reproduction.

Data: Alpaydin and Kaynak (1998), UCI Optical Recognition of Handwritten Digits,
[DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49), CC BY 4.0. Pixel counts are
scaled by 1/16; deliberate label changes retain the source labels and IDs.
Original ARENA chapters and the existing 16.7 lesson are unchanged.
