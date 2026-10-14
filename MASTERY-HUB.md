# ARENA Mastery Hub

**Status: planning only.** The companion labs described here are not implemented or course-ready. This addition preserves every existing repository file.

## Start here

Keep the excellent original resources as the source material. Add an original, interactive companion that makes understanding testable: predict an outcome, change the experiment, inspect what actually happens, explain the surprise, and transfer the idea.

| Resource | Role |
| --- | --- |
| [Original ARENA course](README.md) | Canonical implementation and interpretability foundation; no rewritten lessons. |
| [Existing Frontier Lab roadmap](Extension-Roadmap.md) | Existing extension priorities, methods, and verification requirements. |
| [Deep Learning Alchemy — Stanford CS 312](https://deep-learning-alchemy.github.io/) | Canonical external course; open the original materials rather than a copied or rewritten version. |
| [Mastery Hub design](docs/superpowers/specs/2026-10-14-alchemy-mastery-hub-design.md) | Learning experience, scientific safeguards, curriculum map, and release boundaries. |
| [Mastery Hub implementation plan](docs/superpowers/plans/2026-10-14-alchemy-mastery-hub.md) | Ordered engineering tasks, exact file targets, interfaces, tests, and acceptance gates. |

The new plan supplements, rather than replaces, `Extension-Roadmap.md`. The source of truth for existing lesson readiness remains [the existing status registry](docs/arena_style_rewrite_status.yml); this document does not promote any existing or proposed lesson.

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
| 0 | Source inventory, additive design, implementation plan | This documentation proposal |
| 1 | One complete optimizer pilot: interactive quadratic lab, notebook, controls, prediction record, and replay | Planned |
| 2 | Small real-model bridge plus basics/scaling and optimization companions | Planned |
| 3 | Architecture and generalization companions, mapped to verified upstream releases | Planned; release-dependent |
| 4 | MechInterp transfer challenges and an evidence-backed capstone | Planned |
| 5 | Learner review of the finished release, then a sustainable recurring study schedule | Deferred until release acceptance |

Unreleased or unspecified upstream material remains explicitly pending. The pilot is a quality gate, not permission to mass-produce shallow chapters.

## Preservation and scope

The original ARENA baseline is `f9f034bdb5b8748f44e8b4533b5c5bea68dc8bc0`. The planning baseline is the extension branch at `458a7b3ac327af538abac3400fa0fe8c64b146f9`.

No original notebooks, course pages, assets, dependency pins, model results, or readiness declarations are changed by this proposal. Stanford content stays upstream with attribution; additional resources will be inventoried from actual repository/user links rather than guessed.

No recurring reminders or calendar events are created by this plan. Build and verify the agreed learning release first; arrange the study schedule afterward.
