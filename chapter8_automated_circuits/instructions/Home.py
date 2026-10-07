import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = r"""## TLDR (if you're in a hurry!)

**Prerequisites:** [1.2] Intro to Mech Interp and [1.4.1] Indirect Object Identification (essential); chapter 6 for 8.6 and 8.7.

- Do **8.1 → 8.2 → 8.3 → 8.4** in order: activation patching, then its fast linear approximations (attribution patching, EAP, AtP*), then automated discovery with ACDC and the metrics for judging a circuit.
- **8.5 Causal Scrubbing** is about *testing* a circuit hypothesis rather than finding one.
- **8.6 Sparse Feature Circuits** and **8.7 Attribution Graphs** move from components to features.
- **8.8** benchmarks all the methods on models whose circuits are known exactly, and is the best place to finish.

## Description of material

[1.4.1] finds the IOI circuit by hand. This chapter is about doing that automatically and about knowing when to trust the answer. You'll implement the main families of circuit-discovery methods from scratch, compare them against exact patching, test hypotheses with causal scrubbing, build circuits out of SAE and transcoder features, and finally evaluate every method on compiled transformers where the true circuit is known."""

render_chapter_home(__file__, INTRO)
