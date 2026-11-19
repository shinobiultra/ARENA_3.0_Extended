import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = (
    '## Description of material\n'
    '\n'
    "This chapter applies interpretability to safety-relevant behaviour: refusal, honesty, faithfulness of reasoning, drift during fine-tuning, monitoring, editing and unlearning. All experiments use small open models and benign proxy tasks, and report aggregate behaviour rather than harmful content. A recurring theme is the gap between a method that *works on the benchmark* and a method that does what it claims: steering that damages capabilities, monitors that can be evaded, edits that don't generalise, and unlearning that only suppresses."
)

render_chapter_home(__file__, INTRO)
