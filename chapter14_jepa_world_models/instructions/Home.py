import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = r"""## TLDR (if you're in a hurry!)

**Prerequisites:** [1.5.3] OthelloGPT for the world-model sections; [0.2] CNNs & ResNets for the vision sections.

- *JEPA track*: **14.1 I-JEPA → 14.2 V-JEPA → 14.3 Action-Conditioned World Models**, then 14.4 for the wider JEPA family.
- *World-model interpretability track*: **14.5 Maze Transformers → 14.6 Sudoku → 14.7 Recurrent World Models**.
- The two tracks are independent.

## Description of material

Joint-embedding predictive architectures learn by predicting representations rather than pixels or tokens, and are a leading candidate for learning world models. This chapter builds them from scratch, checks for representation collapse, and asks whether their latents encode physical state (positions, velocities, occluded objects) more directly than generative models do. The second track takes the OthelloGPT question ("does the model have a world model?") to mazes, constraint puzzles and model-based RL agents."""

render_chapter_home(__file__, INTRO)
