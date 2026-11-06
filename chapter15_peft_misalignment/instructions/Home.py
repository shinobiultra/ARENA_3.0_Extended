import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = (
    "## TLDR (if you're in a hurry!)\n"
    '\n'
    '**Prerequisites:** [1.1] Transformer from Scratch; [4.1] Emergent Misalignment is useful background for 15.4.\n'
    '\n'
    '- **15.1 LoRA and DoRA** first, then **15.2** (how LoRA differs from full fine-tuning) and **15.3 QLoRA**.\n'
    '- **15.4** connects adapters to behaviour drift, and **15.5** applies the same tools to a vision-language model.\n'
    '\n'
    '## Description of material\n'
    '\n'
    "Most fine-tuning in practice is parameter-efficient. That's an opportunity for interpretability: a LoRA adapter is a small, low-rank object you can take apart directly. "
    "You'll implement LoRA, DoRA and QLoRA from scratch, compare what they learn with full fine-tuning (intruder dimensions, effective rank, subspace overlap), and watch a "
    'narrow fine-tune turn into broad behaviour change.'
)

render_chapter_home(__file__, INTRO)
