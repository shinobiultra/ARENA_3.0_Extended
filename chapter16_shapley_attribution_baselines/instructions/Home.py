import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = (
    '## Description of material\n'
    '\n'
    'Shapley values are the most widely used black-box attribution method, and a natural baseline for mechanistic '
    'interpretability. This chapter builds them up from exact values on games where the right answer is known, '
    'through the fast approximations used in practice and interaction indices, to attributing model behaviour to '
    'tokens, image regions and training examples. Every method is checked against ground truth, and section 16.8 '
    'asks the key question for this course: when does a black-box attribution agree with a causal, mechanistic one?'
)

render_chapter_home(__file__, INTRO)
