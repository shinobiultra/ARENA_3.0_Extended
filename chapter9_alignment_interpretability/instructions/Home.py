import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = r"""## TLDR (if you're in a hurry!)

**Prerequisites:** [1.2] Intro to Mech Interp and [1.3.2] Function Vectors & Model Steering. Chapter 4 of original ARENA (Alignment Science) is good background.

- **9.1 Refusal Directions** is the best entry point.
- 9.2 (truthfulness and hallucination), 9.3 (chain-of-thought faithfulness), 9.4 (emergent misalignment) and 9.5 (white-box monitors) can be done in any order.
- **9.6 ROME/MEMIT** and **9.7 Unlearning** are about *changing* what a model knows, and how to check whether you really did.

## Description of material

This chapter applies interpretability to safety-relevant behaviour: refusal, honesty, faithfulness of reasoning, drift during fine-tuning, monitoring, editing and unlearning. All experiments use small open models and benign proxy tasks, and report aggregate behaviour rather than harmful content. A recurring theme is the gap between a method that *works on the benchmark* and a method that does what it claims: steering that damages capabilities, monitors that can be evaded, edits that don't generalise, and unlearning that only suppresses."""

render_chapter_home(__file__, INTRO)
