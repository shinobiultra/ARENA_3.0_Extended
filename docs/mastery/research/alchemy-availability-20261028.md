# Deep Learning Alchemy availability — 2026-10-28

Research for [decision ticket #5](https://github.com/shinobiultra/ARENA_3.0_Extended/issues/5), under [map #4](https://github.com/shinobiultra/ARENA_3.0_Extended/issues/4). This is a source-availability assessment, not implementation or learner acceptance. Checked the official site, its source tree, and the assignment release branches; no training, installs, tests, or downloads of datasets/checkpoints ran.

## Finding

Four assignments now have published PDF and code links. Assignment 4 is newly actionable for planning the stability companion; the local October 21 catalog/inventory is stale on that point. Assignments 5–7 remain pending. Publication does not establish external checkpoint access, successful execution, reuse rights, or companion readiness. [Official schedule](https://deep-learning-alchemy.github.io/#schedule)

| Local unit | Local source status | Checked assignment availability | Recommendation |
| --- | --- | --- | --- |
| `u01_basics` | published | [A1 PDF](https://deep-learning-alchemy.github.io/assets/assignments/a1_basics.pdf), [A1 code](https://github.com/deep-learning-alchemy/assignments/tree/baafc3f2d364e5ddbba75f0fe8d1cb29326920e5) | Retain published |
| `u02_optimization` | published | [A2 PDF](https://deep-learning-alchemy.github.io/assets/assignments/a2_hyperparameter_scaling.pdf), [A2 code](https://github.com/deep-learning-alchemy/assignments/tree/069e18c52c93a0c6626b00f1c79de070ba30275f) | Retain published |
| `u03_landscapes` | published | [A3 PDF](https://deep-learning-alchemy.github.io/assets/assignments/a3_optimization2.pdf), [A3 code](https://github.com/deep-learning-alchemy/assignments/tree/57c60bbeef200ee20f5bdd847bfefaf7646fee78) | Retain published |
| `u04_stability` | pending_upstream | [A4 PDF: Transformers vs RNNs](https://deep-learning-alchemy.github.io/assets/assignments/a4_architecture1.pdf), [A4 code](https://github.com/deep-learning-alchemy/assignments/tree/4be74a55db2c9d87df59c637c9795377f3320b9b) | Propose published |
| `u05_capacity` | pending_upstream | Scheduled October 29; no assignment links | Retain pending |
| `u06_generalization` | pending_upstream | Scheduled November 10; no assignment links | Retain pending |
| `u07_exotic` | pending_upstream | Scheduled November 19; topic remains unspecified | Retain pending |

The schedule also links Architecture 1 lecture 7/8 slides. The homepage announces DNA from Assignment 6 onward; this is not a published assignment protocol. These observations are bounded to the [official site snapshot](https://github.com/deep-learning-alchemy/deep-learning-alchemy.github.io/tree/ad2a4967d4df985cf35562091df1f78eb5aea176); “pending” means no release found in the checked official surfaces.

## What “usable” permits us to conclude

- **A1:** the pinned release README says problem-specific files named in the handout are not bundled; it supplies a default run and LR-tuning template. A published handout is therefore not a turnkey implementation. [A1 setup](https://github.com/deep-learning-alchemy/assignments/blob/baafc3f2d364e5ddbba75f0fe8d1cb29326920e5/README.md)
- **A2:** the release has 78 supplied source runs with an offline CSV path. Target experiments, policy functions, fitting and interpretation remain student work. The handout contains an example-question answer key, so reading it changes answer exposure. [A2 runtime guide](https://github.com/deep-learning-alchemy/assignments/blob/069e18c52c93a0c6626b00f1c79de070ba30275f/experiments/a2/README.md), [A2 handout](https://deep-learning-alchemy.github.io/assets/assignments/a2_hyperparameter_scaling.pdf)
- **A3:** starters cover connectivity, schedule/EMA interventions, sharpness and Hessian probes. Supplied Hessian checkpoints reside in the course Modal environment; independent users need staff copies or their own checkpoints. The personal-workspace guide downloads data, not these checkpoints. [A3 guide](https://github.com/deep-learning-alchemy/assignments/blob/57c60bbeef200ee20f5bdd847bfefaf7646fee78/experiments/a3_optimization/README.md)
- **A4:** the release describes 105 P1/P2 reference outcomes and public W&B links, with checkpoints in the course Modal environment. P3/P4 training runs and checkpoints are not supplied. It includes architecture definitions, diagnostics and local-GPU entry points; availability of course checkpoints to this learner remains unverified. [A4 guide](https://github.com/deep-learning-alchemy/assignments/blob/4be74a55db2c9d87df59c637c9795377f3320b9b/experiments/a4_stability/README.md)

The official repository documents [local CUDA/Slurm](https://github.com/deep-learning-alchemy/assignments/blob/4350166ac1cedba0222a3124a2a2a54b65fb635d/gpu/README.md) and [personal Modal workspace](https://github.com/deep-learning-alchemy/assignments/blob/4350166ac1cedba0222a3124a2a2a54b65fb635d/README_MODAL_OWN_WORKSPACE.md) paths. Documentation was inspected, not executed. Its Python 3.10 pin and course compute defaults are source facts, not instructions to change this project's environment or authorize jobs.

## Rights and attribution

GitHub metadata reports `license: null` for both the [assignment repository](https://api.github.com/repos/deep-learning-alchemy/assignments) and [website repository](https://api.github.com/repos/deep-learning-alchemy/deep-learning-alchemy.github.io). Recursive trees at the four release commits and both current main commits contain no blanket LICENSE/COPYING/NOTICE file. A2 has a scoped third-party [Muon MIT notice, copyright Keller Jordan 2024](https://github.com/deep-learning-alchemy/assignments/blob/069e18c52c93a0c6626b00f1c79de070ba30275f/experiments/a2/licenses/Muon.txt), attributed in its runtime guide; that does not establish a blanket Alchemy license. [Assignment branches](https://api.github.com/repos/deep-learning-alchemy/assignments/branches), [assignment main tree](https://api.github.com/repos/deep-learning-alchemy/assignments/git/trees/4350166ac1cedba0222a3124a2a2a54b65fb635d?recursive=1), [website tree](https://api.github.com/repos/deep-learning-alchemy/deep-learning-alchemy.github.io/git/trees/ad2a4967d4df985cf35562091df1f78eb5aea176?recursive=1)

Retain the user's canonical-originals-plus-provenance choice: link official originals, identify CS312/Stanford Fall 2026 and assignment/revision, and write original companions. Do not infer permission to copy handouts, code, quiz answers, images or measurements from public access. GitHub also distinguishes public visibility from licensing. Dataset, checkpoint and figure-specific rights remain unaudited. [GitHub licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)

## Implications for M2–M4

These are recommendations against the existing local plan; no curriculum/status changes were adopted here.

- **M2:** A1–A3 supply verified source entry points. Pin each release separately: current assignment main omits A2's dedicated tree. Preserve Q001's M1 learner walkthrough and Q002's frozen organism, TRAIN/DEV/TEST, metrics, seeds and aggregate budget before empirical work. Supplied results can inform a protocol, but are exposed reference data rather than untouched evaluation.
- **M3:** plan `u04_stability` against A4's actual transformer/RNN depth/time questions. Refresh its source status/date and the source inventory through a separate reviewed change; retain all companion statuses. Keep capacity/generalization pending until their releases can be checked, including the announced modality change.
- **M4:** Alchemy publication does not implement or validate causal interpretability transfer or a capstone. Retain the planned measured interventions, random controls, held-out predictions and independent review.

Open items: A5–A7 release contents; external access to A3/A4 reference checkpoints; dataset/checkpoint/figure reuse terms; runtime compatibility; Q001 learner acceptance and Q002 experiment protocol/budget. None requires guessing an unreleased assignment.

## Provenance and comparison boundary

Observed release heads: `a1_release=baafc3f2d364e5ddbba75f0fe8d1cb29326920e5`, `a2_release=069e18c52c93a0c6626b00f1c79de070ba30275f`, `a3_release=57c60bbeef200ee20f5bdd847bfefaf7646fee78`, `a4_release=4be74a55db2c9d87df59c637c9795377f3320b9b`. Assignment main: `4350166ac1cedba0222a3124a2a2a54b65fb635d`. Website main: `ad2a4967d4df985cf35562091df1f78eb5aea176`; homepage Git blob: `e9f9d17ccffee0f38f8b7969e41086cb1e4cee36`.

All four canonical PDFs parsed successfully. The website tree records PDF Git blobs A1 `28187b017a5c4c3131c9fc8ca973125d54bdba75`, A2 `8dd8b83025d0d6e2cf2c562de7ce2ee2636b0aa4`, A3 `4d48c82144259e83c300a53fb8fd35ff86a46841`, A4 `656b29acb21a3b43b93222740b8cc5efe73790d1`. These are Git object IDs, not SHA-256 checksums.

The comparison used read-only local files below. They are absent from the remote extension base `458a7b3ac327af538abac3400fa0fe8c64b146f9`; their contents were not published with this report.

| Local path | SHA-256 |
| --- | --- |
| `mastery_hub/catalog.json` | `f31687428b9889f69690d564396f05bbf2d6f1a066e5cbed73e63e63ffc86876` |
| `docs/mastery/source_inventory.md` | `64fddb3992b07287c9a508bc8378d8e034f145dd85e4c9cd69ec61682d82f0ac` |
| `docs/superpowers/specs/2026-10-14-alchemy-mastery-hub-design.md` | `e99df119311c86b01cbc11736efbc74006a06e2b7fa423219f68b9d62cdf9d29` |
| `docs/superpowers/plans/2026-10-14-alchemy-mastery-hub.md` | `d42167c148bd792d73868447116d8a6330e07ca94d94c46ad31f874d3c9b7009` |
| `docs/mastery/implementation_record.md` | `5c651cafe68680100f66f24664693b6d6a87c8f753c944d16e27c6c5dd31fd6d` |

Only this research report is published on the disposable research branch. Original ARENA, shared checkout/index, source, notebooks and evidence were not changed. No upstream materials were vendored.
