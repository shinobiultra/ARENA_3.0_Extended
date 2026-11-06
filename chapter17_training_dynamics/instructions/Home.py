import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = (
    "## TLDR (if you're in a hurry!)\n"
    '\n'
    '**Prerequisites:** [1.2] Intro to Mech Interp and [1.5.2] Grokking & Modular Arithmetic.\n'
    '\n'
    '- **17.1 Checkpoint Archaeology and Grokking** and **17.2 Induction Head Emergence** are the core.\n'
    '- 17.3 (fine-tuning diffs and capability emergence) and 17.4 (comparing architectures) build on them.\n'
    '\n'
    '## Description of material\n'
    '\n'
    'Interpretability usually studies a finished model. Developmental interpretability asks how the mechanisms got there: when a circuit forms, what the loss curve looks like '
    "when it does, and whether the same capability is learned the same way by different architectures. You'll train small models yourself, save checkpoints, and also mine the "
    'public training checkpoints of the Pythia suite.'
)

render_chapter_home(__file__, INTRO)
