import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = (
    "## TLDR (if you're in a hurry!)\n"
    '\n'
    '**Prerequisites:** [1.5.3] OthelloGPT for the world-model sections; [0.2] CNNs & ResNets for the vision sections.\n'
    '\n'
    '- *JEPA track*: **14.1 I-JEPA → 14.2 V-JEPA → 14.3 Action-Conditioned World Models**, then 14.4 for the wider JEPA family.\n'
    '- *World-model interpretability track*: **14.5 Maze Transformers → 14.6 Sudoku → 14.7 Recurrent World Models**.\n'
    '- The two tracks are independent.\n'
    '\n'
    '## Description of material\n'
    '\n'
    'Joint-embedding predictive architectures learn by predicting representations rather than pixels or tokens, and are a leading candidate for learning world models. This '
    'chapter builds them from scratch, checks for representation collapse, and asks whether their latents encode physical state (positions, velocities, occluded objects) more '
    'directly than generative models do. The second track takes the OthelloGPT question ("does the model have a world model?") to mazes, constraint puzzles and model-based RL '
    'agents.'
)

render_chapter_home(__file__, INTRO)
