# PROTOTYPE: the learner's single route (wayfinder ticket #30)

Throwaway storyboard for [What single route does a learner take through the whole course?](https://github.com/shinobiultra/ARENA_3.0_Extended/issues/30). It lives on the branch `prototype/study-route-30`, which is never merged.

**Open `route-storyboard.html` in a browser** (download it, or `xdg-open prototypes/study-route-30/route-storyboard.html` from a checkout of this branch). Flip between the three routes with the black bar or ←/→. `?variant=A|B|C` selects one directly. Click a lesson to see its prerequisites and where they sit on that route. Lessons whose prerequisites come later get a red outline.

The page shows:
- the route as stages, with core (●) and optional (dashed) lessons, original ARENA lessons (grey), extension lessons (blue), Alchemy companions as side trips (purple), 0.7 as a shelf (yellow) and PRH as a placeholder (red);
- pacing in days and weeks, plus a check that prerequisites come before the lessons that need them;
- four mocks of how the next step is shown: the top and the end of an extension page, the reader, and an extension chapter home;
- what a learner sees first, today and proposed;
- the open choices SR1–SR8.

## Data

- Sections and titles come from `infrastructure/extension/syllabus.yaml` and `infrastructure/core/config.yaml` in the main checkout, at `integration/extension` 9527393e. They were read on 2026-11-06 and nothing there changed.
- **Minutes** are the sum of each page's "You should spend up to N minutes" lines. **Days** follow one rough rule: one day per 6 h of those minutes, rounded to half days, with a minimum of one day. Afternoons such as 10.2 count as half a day. Original 1.2 has only four time lines, so it is fixed at one day.
- **Prerequisite edges are a draft.** They come from syllabus briefs, the few pages that already state a "Prerequisites:" line (5.6, 5.7, 7.6, 9.7, 11.6, 13.6, 14.6, 14.7) and the sections pages cite. Batches confirm them.
- **The "core" picks are a draft** that follows original chapter 1's own convention: "1.1 and 1.2 are compulsory… the remaining sections are optional… in any order". The Spec lists the real picks, and walkthroughs can move them.
- Alchemy unit titles come from `mastery_hub/catalog.json`. The catalog retires under #27, so it is used here only for the titles.

## What the three routes cost (core route, from the storyboard)

| | A originals first | **B interleaved (recommended)** | C spine + tracks |
|---|---|---|---|
| Core route | 46.5 days ≈ 9.3 weeks full-time | same lessons, so same total | same |
| Everything, optional included | 128 days ≈ 25.6 weeks (Alchemy companions not sized) | same | same |
| First chapter 5–18 lesson finished | day 25 | **day 9.5** | day 10.5 |
| Average core days between a prerequisite and the lesson that needs it | 8.8 (max 31.5) | **3.6** (max 20.5) | 2.5 within a track (max 32.5) |
| Prerequisites that come later on the route | 0 | 0 | 0, but 23 prerequisites sit in another track |

The three routes contain the same lessons, so the core size is SR3's question, not SR1's. SR1 is about order.

## Route B, the recommendation

| Stage | Name | Core | Optional | Side trips | Core days | All days |
|---|---|---|---|---|---|---|
| 1 | Start | 0.0 | — | — | 1 | 1 |
| 2 | Fundamentals (original ch0) | 0.1, 0.2, 0.3, 0.4 | 0.5 | u01, u02 (+ optimizer lab), u03, 0.7 shelf | 4.5 | 5.5 |
| 3 | Transformers (original 1.1, 1.2) | 1.1, 1.2 | 1.6 | u04 | 2 | 3 |
| 4 | Is it real? | 0.6 | — | — | 1 | 1 |
| 5 | Modern architectures (ch5) | 5.1, 5.3 | 5.2, 5.4–5.9 | u05 | 2 | 9.5 |
| 6 | Features (1.5.4, 1.3.3 → ch6) | 1.5.4, 1.3.3, 6.1, 6.2 | 6.3–6.5 | — | 5 | 8 |
| 7 | Circuits (1.4.1 → ch8) | 1.4.1, 8.1, 8.2 | 1.4.2, 8.3–8.8 | — | 3 | 10.5 |
| 8 | Reading activations (ch7) | 7.1 | 1.3.4, 7.2–7.6 | — | 1 | 7 |
| 9 | Geometry (1.3.1 → ch11) | 1.3.1, 11.1 | 11.2–11.6 | — | 2 | 7 |
| 10 | Research Practice I (ch10) | 10.1 parts 1️⃣–4️⃣, 10.2 | 10.3 | — | 1.5 | 2 |
| 11 | RL (original ch2) | 2.1, 2.2, 2.3 | 2.4, 2.5 | — | 3 | 5 |
| 12 | Evals (original ch3) | 3.1, 3.2, 3.3 | 3.4, 3.5 | — | 3 | 5 |
| 13 | Alignment (ch4 → 1.3.2 → ch9, ch15) | 4.1, 1.3.2, 4.4, 9.1, 15.1 | 4.2, 4.3, 4.5, 9.2–9.7, 15.2–15.4 | — | 5.5 | 17.5 |
| 14 | Multimodal (ch12 → ch13) | 12.1, 12.3, 13.1 | 12.2, 12.4–12.7, 15.5, 13.2–13.6 | — | 3 | 14 |
| 15 | World models (1.5.3 → ch14 → **PRH slot**) | 14.1 | 1.5.3, 14.2–14.7 | — | 1 | 8.5 |
| 16 | Attribution (ch16) | 16.1, 16.2 | 16.3–16.11 | u06 | 2 | 12.5 |
| 17 | Dynamics & intrinsic (1.5.2 → ch17, ch18) | 1.5.2, 17.1, 18.1 | 1.5.1, 17.2–17.4, 18.2 | — | 3 | 8 |
| 18 | Finale | 10.1 part 5️⃣ "Your capstone" | — | u07 | 3 | 3 |

The PRH slot is a placeholder. [Where should the PRH research chapter sit…](https://github.com/shinobiultra/ARENA_3.0_Extended/issues/235) decides it; that ticket's proposed direction is "geometry entry with JEPA-section revisit".

## Open choices

| ID | Choice | Recommendation | Alternatives |
|---|---|---|---|
| SR1 | Route shape | **B**: one ordered route. Original chapters stay whole blocks, and the optional ch1 sets sit right before the extension chapter that builds on them. | A originals first, then 5–18 in number order; C a short core spine, then elective tracks in any order |
| SR2 | Where a learner starts | **One "Start here"**: course home → original 0.0 → 0.1, with one skip line for ARENA graduates ("start at 0.6"). Retire the five "pick your first question" entries in STUDY-GUIDE.md and the reader. | keep several entry points |
| SR3 | Core versus optional | **ARENA convention**: each chapter home names its core lessons; the rest are optional, in any order. ☆ Bonus, Alchemy side trips and 0.7 are always optional. | everything core; core = only what chapter 10 needs |
| SR4 | How the next step is shown | **Pages:** only the hand-written "Prerequisites:" line. **Reader:** a route sidebar plus "Next on the route →" on a lesson's last part. **Chapter homes:** an ARENA-style TLDR naming what comes before and after. | also a "Next on the route" line at the end of every extension page |
| SR5 | Where the route lives | **A `route:` block in syllabus.yaml** (stages → section numbers, originals included, with a core flag). STUDY-GUIDE.md is rewritten as the route page (start, stage table, days) and loses its status and provenance sections. The reader reads the same block. | a separate route.yaml; a hand-maintained guide |
| SR6 | Where chapter 10 sits | **Split.** 10.1 parts 1️⃣–4️⃣ and 10.2 come after the interpretability core. 10.3 and later Micro-Lab rungs are optional afternoons after later stages. 10.1 part 5️⃣ is the finale. | all of chapter 10 at the end |
| SR7 | Alchemy companions | **Optional side trips at their matching stop**: u01–u03 at 0.3, u04 at 1.1, u05 at 5.2, u06 near 16.9, u07 to be placed. MASTERY-HUB.md lists all seven. | one Alchemy block after chapter 0; a parallel "training science" track |
| SR8 | Pacing | **Days, ARENA-style**, from each lesson's time lines. The route page shows days per stage and core and full totals in weeks. No timer, no tracking, nothing stored. | hours only; no pacing on the route page |

Display only, nothing to decide: 0.7 is a shelf that first uses link into, as #234 settled. PRH is a placeholder slot that adopts #235's answer.

Nothing was run, downloaded or trained.
