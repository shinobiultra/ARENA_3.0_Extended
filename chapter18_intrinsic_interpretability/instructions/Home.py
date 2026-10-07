import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = r"""## TLDR (if you're in a hurry!)

**Prerequisites:** 11.3 (Causal Abstraction and DAS) for the interchange-intervention part of 18.1.

The two sections are independent.

## Description of material

Everything else in this course interprets models after training. The alternative is to build models that are interpretable by construction: concept bottlenecks, Backpack language models, networks trained so that a chosen subspace implements a chosen variable, sparse and modular networks, Kolmogorov-Arnold networks and sparse trees. You'll build each one, and measure what it costs in accuracy and what it buys in interpretability."""

render_chapter_home(__file__, INTRO)
