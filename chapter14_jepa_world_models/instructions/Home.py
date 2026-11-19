import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = (
    '## Description of material\n'
    '\n'
    'Joint-embedding predictive architectures learn by predicting representations rather than pixels or tokens, and'
    ' are a leading candidate for learning world models. This chapter builds them from scratch, checks for '
    'representation collapse, and asks whether their latents encode physical state (positions, velocities, occluded'
    ' objects) more directly than generative models do. The second track takes the OthelloGPT question ("does the '
    'model have a world model?") to mazes, constraint puzzles and model-based RL agents.'
)

render_chapter_home(__file__, INTRO)
