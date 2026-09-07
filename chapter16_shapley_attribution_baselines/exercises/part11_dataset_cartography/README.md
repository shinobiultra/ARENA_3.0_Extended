# [16.11] Dataset Cartography

Can a model's learning history tell you which training examples to keep?

Start with [the exercises](16.11_Dataset_Cartography_exercises.ipynb). The
[solutions](16.11_Dataset_Cartography_solutions.ipynb) include the complete
executed results. [lesson.md](lesson.md) is the readable source; the
[instruction page](../../instructions/pages/11_[16.11]_Dataset_Cartography.md)
contains the same narrative and figures.

Nine exercise groups teach ten functions: stable example-ID alignment,
assigned-label probabilities, confidence/variability/correctness, independent
network forward passes, SGD logging, region ranking, error retrieval, set
overlap, class-matched sampling and held-out evaluation. Twenty figures connect
the calculations to actual digit images, learning histories, noise detection
and fresh retraining. Two editable plays change a probability history and
retrain a real selected subset against a matched random control.

The frozen experiment trains 20 mapping models and 120 fresh subset/full-pool
models on CPU, including all five seeds, label controls, both logging conventions
and an equal-update comparison. It evaluates all models on 823 same-source and
1797 separate-writer images. No held-out prediction chooses a subset, checkpoint
or seed. The reference notebook takes about thirteen seconds on the development
machine; allow four to six hours for self-study.

Low confidence retrieves 88–90% of the 300 injected errors within a 300-image
inspection budget. High variability is not a universally better selection:
the one-third subset underperforms its class-matched random control, including
after matching optimizer updates. The original images, failures and class
omissions are retained. This is a real small-network method experiment, not a
RoBERTa NLP or whole-paper reproduction.

## Run and inspect

From the repository root:

```bash
uv run pytest -q chapter16_shapley_attribution_baselines/exercises/part11_dataset_cartography/cartography_methods_tests.py chapter16_shapley_attribution_baselines/exercises/part11_dataset_cartography/cartography_contract_tests.py
uv run python chapter16_shapley_attribution_baselines/exercises/part11_dataset_cartography/execute_learner.py --output-directory /tmp/cartography-own-new
uv run python chapter16_shapley_attribution_baselines/exercises/part11_dataset_cartography/reproduce_native.py --learner-directory /tmp/cartography-own-new --output-directory /tmp/cartography-fresh-new
```

Output directories must be new. The own-answer executor fills each stub only
from its own following Solution dropdown. It then trains the actual models in
a fresh kernel. The reproducer reloads the source archive and performs every
fit again; it does not consume learner parameters as training inputs. It checks
all 675 tensors, 41903198 elements and 8904 scalar/ID values. The development
run agrees exactly. Independent oracles additionally compare NumPy coordinates,
ordinary torch.nn/SGD and separate versus scheduled training.

[protocol.md](protocol.md), [protocol_freeze.json](protocol_freeze.json) and
[data_manifest.json](data_manifest.json) record the pre-experiment choices.
[expected_outputs/README.md](expected_outputs/README.md) explains the evidence.
Original ARENA chapters remain unchanged.

## Sources

- [Swayamdipta et al. (EMNLP 2020)](https://aclanthology.org/2020.emnlp-main.746/)
- [Author implementation](https://github.com/allenai/cartography)
- [UCI Optical Recognition of Handwritten Digits](https://doi.org/10.24432/C50P49), CC BY 4.0
