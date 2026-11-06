---
# 🛠 Prototype (issue 246): what `write_note()` in 10.2 would write. One file per claim, Obsidian-compatible, local only.
claim: "One vector steers four behaviours at four lengths; v1 + v2 steers a third"
source: https://www.cs.columbia.edu/~johnhew/do-not-rely-linearity.html
author: John Hewitt
read: ‹date›
tier: reproduce            # re-run | reproduce | replicate | generalise
model: Qwen/Qwen3-1.7B@‹revision›
artifacts: john-hewitt/odd-steering-expts@04ad6c1
verdict: ‹›                # replicated | partly replicated | not replicated | inconclusive
share: private             # private | sent to author | fieldnote
tags: [fresh-claim, steering, linearity]
---

# ‹The claim in one line›

**Expected (before running):** ‹from the registered predictions›
**Observed (including the control):** ‹the numbers, with intervals, and the random band›
**Interpretation / strongest alternative explanation:** ‹›
**Next smallest discriminating experiment:** ‹one runnable thing›

## Claims and verdicts

| Claim | Rung | Published (1.7B) | Ours [95% CI] | Against random [95% CI] | Verdict |
|---|---|---|---|---|---|
| Dial: French at length 71 | existence | not published | ‹› | ‹› | ‹› |
| Dial: code style at length 55 | existence | not published | ‹› | ‹› | ‹› |
| Dial: sycophancy at length 98 (proxy) | existence | not published | ‹› | ‹› | ‹› |
| Composition: v1 + v2 → code style | existence | +0.45 | ‹› | ‹› | ‹› |
| Linear representations often fail | prevalence | (argument) | n/a | n/a | not tested here |

## What I did differently from the post

- ‹own hook and scorers; proxy for sycophancy; no sentiment; validation prompts for the sweep; …›

## Critique checklist

1. Primary source: ‹›
2. Rungs vs evidence: ‹›
3. Numbers at my scale, artifacts, licence: ‹›
4. Samples, seeds, intervals: ‹›
5. Metric validity: ‹›
6. Representative examples: ‹›
7. Random direction of the same size: ‹›
8. Size against normal activations: ‹›
9. Leakage and overfitting: ‹›
10. Forking paths: ‹›
11. Quality cost: ‹›
12. Strongest alternative and the cheapest test: ‹›

## Figure

![[dial.png]]

## Log

‹the ResearchLog, printed in order›
