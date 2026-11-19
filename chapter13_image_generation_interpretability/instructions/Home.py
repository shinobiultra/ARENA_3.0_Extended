import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = (
    '## Description of material\n'
    '\n'
    'Image generators are a different kind of model to interpret: there\'s no single forward pass, the "answer" '
    'emerges over dozens of denoising steps, and the output is pixels rather than tokens. This chapter builds the '
    'main generator families from scratch and studies them with attention attribution, causal patching across '
    'timesteps, concept directions, sparse autoencoders and adapter analysis.'
)

render_chapter_home(__file__, INTRO)
