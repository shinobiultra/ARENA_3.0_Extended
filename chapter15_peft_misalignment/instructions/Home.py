import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = (
    '## Description of material\n'
    '\n'
    "Most fine-tuning in practice is parameter-efficient. That's an opportunity for interpretability: a LoRA "
    "adapter is a small, low-rank object you can take apart directly. You'll implement LoRA, DoRA and QLoRA from "
    'scratch, compare what they learn with full fine-tuning (intruder dimensions, effective rank, subspace '
    'overlap), and watch a narrow fine-tune turn into broad behaviour change.'
)

render_chapter_home(__file__, INTRO)
