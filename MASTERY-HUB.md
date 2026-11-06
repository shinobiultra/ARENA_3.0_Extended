# ARENA Mastery Hub

**Status: 86 configured study lessons, 86 guided checkpoints, the CPU optimizer pilot and six blank prediction notebooks; learner walkthrough deferred (Q001).**
The later Alchemy companion labs remain planned. Original ARENA lessons are preserved.
The 26 newly authored CPU study editions close the configured lesson-file gaps; native/model
qualification and learner acceptance remain pending. No configured lesson is currently missing.
See [launch instructions](mastery_hub/README.md), [review evidence](docs/mastery/pilot_review.md),
the [November 2 study coverage and limits](docs/mastery/study_coverage_2026-11-02.md),
and the [historical implementation record](docs/mastery/implementation_record.md).

## Start here

To study the extension, launch the **lesson reader** from the repository root:

```bash
uv run --no-sync --offline streamlit run mastery_hub/read.py --server.address=127.0.0.1 --server.port=18505 --server.headless=true --browser.gatherUsageStats=false
```

Open [the reader](http://127.0.0.1:18505/) and choose one lesson. It offers blank
Practice, intact references, an exact notebook save directory and a bookmark for
your current part. The separate optimizer app is an optional lab; it is not the
entrypoint to the 86-lesson course. See the [study guide](mastery_hub/STUDY-GUIDE.md)
for prerequisites and first-session stopping points.

Keep the excellent original resources as the source material. Add an original, interactive companion that makes understanding testable: predict an outcome, change the experiment, inspect what actually happens, explain the surprise, and transfer the idea.

| Resource | Role |
| --- | --- |
| [Original ARENA course](README.md) | Canonical implementation and interpretability foundation; no rewritten lessons. |
| [Existing Frontier Lab roadmap](Extension-Roadmap.md) | Existing extension priorities, methods, and verification requirements. |
| [Deep Learning Alchemy — Stanford CS 312](https://deep-learning-alchemy.github.io/) | Canonical external course; open the original materials rather than a copied or rewritten version. |
| [Mastery Hub design](docs/superpowers/specs/2026-10-14-alchemy-mastery-hub-design.md) | Learning experience, scientific safeguards, curriculum map, and release boundaries. |
| [Mastery Hub implementation plan](docs/superpowers/plans/2026-10-14-alchemy-mastery-hub.md) | Ordered engineering tasks, exact file targets, interfaces, tests, and acceptance gates. |

The new plan supplements, rather than replaces, `Extension-Roadmap.md`. The current master-file rewrite is tracked in [the extension syllabus](infrastructure/extension/syllabus.yaml). This document does not promote existing lessons.

## Across the extension

The [original-materials shelf](docs/mastery/original_materials.md) keeps canonical source links and provenance.
The [guided prediction workbook](mastery_hub/units/extension_predictions/exercises.ipynb) adds a method-specific forecast,
confidence/reasoning, controls, reflection and changed-case transfer for all 86 configured extension lessons.
Those lessons link back to the workbook. Migrations retain the complete original teaching material,
with historical results distinguished from fresh executions.
The new 26 editions teach methods through small controlled CPU organisms and independent fixtures;
their toy results do not establish the corresponding pretrained/model claims.

The 0.6, 1.6 and 16.3 lessons now have separate blank Practice copies. Their
original notebooks, including saved outputs, remain intact and directly linked.
The copies retain visible methods, controls and play; their historical solution
outputs do not establish a fresh successful execution.

Separate [7.1 blank public Practice](mastery_hub/units/pilot_practice/7_1_exercises.ipynb) and [9.1 blank public Practice](mastery_hub/units/pilot_practice/9_1_exercises.ipynb) put a local manual forecast before the preserved lessons, with reflection after controls and intact-original links. These are guided public copies, not additional blind companions or guided checkpoints. Published text and answer dropdowns remain exposed; see [how to use the copies](mastery_hub/units/pilot_practice/README.md).

Six blank companions require commitment before explicit CPU execution and support private reflection/export/replay:
[8.8 circuit](mastery_hub/units/circuit_prediction/exercises.ipynb),
[Data Shapley](mastery_hub/units/shapley_prediction/exercises.ipynb),
[SAGE](mastery_hub/units/sage_prediction/exercises.ipynb),
[SSD](mastery_hub/units/ssd_prediction/exercises.ipynb),
[8.7 graph/path](mastery_hub/units/graph_prediction/exercises.ipynb), and
[6.1 fixed-forward SAE rules](mastery_hub/units/sae_prediction/exercises.ipynb).
The graph/path companion studies graph attribution and path recovery in a fixed public synthetic setup,
separate from the 8.8 readout-gain circuit challenge. Its distinct DEV mathematics tests passed;
the target experiment remains unexecuted, with no empirical result. Its own strict receipt supports
exclusive private export and replay without rerunning methods. Public setup and passing DEV checks do not establish unseen mastery.

The SAE companion compares four fixed forward rules through seven visible student
methods. Separate DEV checks cover implementation, controls, records and replay;
its public target remains unexecuted. Fixed-rule mechanics do not rank learned
SAEs. Learners commit before an explicit run; the app only offers the notebook.

Original ARENA stays unchanged. Further challenge variants and native/model qualification remain unfinished;
these entrypoints do not establish full-course readiness. See the
[integration design](docs/mastery/designs/2026-10-15-integrated-prediction-learning.md).

## The learning loop

**Read the original → form a mental model → predict with confidence → experiment → visualize → implement and verify → explain the mismatch → predict an unseen case.**

Each companion must include a worked example, meaningful live controls, a learner-visible implementation, immediate tests, a negative control, a held-out prediction challenge, a research connection, and honest limitations. An attractive plot or a passing JSON report alone is not enough.

There are two deliberately distinct modes:

- **Explore:** freely change parameters, inspect results, compare runs, and try to break an explanation.
- **Challenge:** commit a prediction and reasoning before revealing the result; changed settings create a new attempt rather than silently changing the question.

Progress records evidence such as successful replay, an unseen prediction, and an explanation of a failed control. Opening a page is not mastery.

## Build order

| Stage | Deliverable | Current status |
| --- | --- | --- |
| 0 | Source inventory, additive design, implementation plan | Implemented: canonical link index, source receipts and additive design |
| 1 | CPU optimizer pilot: interactive quadratic lab, notebook, controls, prediction record, and replay | Implemented and independently reviewed; human walkthrough pending |
| 2 | Small real-model bridge plus basics/scaling and optimization companions | Planned |
| 3 | Architecture and generalization companions, mapped to verified upstream releases | Planned; release-dependent |
| 4 | MechInterp transfer challenges and an evidence-backed capstone | Planned |
| 5 | Learner review of the finished release, then a sustainable recurring study schedule | Deferred until release acceptance |

Unreleased or unspecified upstream material remains explicitly pending. The pilot is a quality gate, not permission to mass-produce shallow chapters.

## Preservation and scope

The original ARENA baseline is `f9f034bdb5b8748f44e8b4533b5c5bea68dc8bc0`. The planning baseline is the extension branch at `458a7b3ac327af538abac3400fa0fe8c64b146f9`.

The original planning proposal remains recorded unchanged in the design and plan linked above. The implementation adds the CPU pilot, repairs incomplete extension prose, and reconciles extension navigation and preservation paths. Original notebooks, course pages and assets, dependency pins and historical model execution outputs are preserved. Stanford content stays upstream with attribution; additional resources will be inventoried from actual repository/user links rather than guessed.

No recurring reminders or calendar events are created by this plan. Build and verify the agreed learning release first; arrange the study schedule afterward.

For a concrete starting route and exact exercise, reference and reading links to all 86 covered lessons, open the [study guide](mastery_hub/STUDY-GUIDE.md). It distinguishes the 26 new CPU study editions from native evidence and explains the pilot reading notes before you interpret their results.
