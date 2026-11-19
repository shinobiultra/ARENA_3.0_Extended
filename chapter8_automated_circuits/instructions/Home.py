import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = (
    '## Description of material\n'
    '\n'
    "[1.4.1] finds the IOI circuit by hand. This chapter is about doing that automatically and about knowing when to trust the answer. You'll implement the main families of circuit-discovery methods from scratch, compare them against exact patching, test hypotheses with causal scrubbing, build circuits out of SAE and transcoder features, and finally evaluate every method on compiled transformers where the true circuit is known."
)

render_chapter_home(__file__, INTRO)
