import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = r"""## TLDR (if you're in a hurry!)

**Prerequisites:** none beyond [0.0]-[0.3]; [1.4.1] for 16.8.

- *Feature attribution*: **16.1 → 16.2 → 16.3**, then 16.4 (global importance), 16.5 (interactions) and 16.6 (sparse interaction discovery).
- *Attribution for LLMs and VLMs*: 16.7, then **16.8 SHAP vs Activation Patching**, which is the bridge to the rest of the course.
- *Data attribution*: **16.9 → 16.10 → 16.11** (Data Shapley, influence functions and TracIn, datamodels and dataset cartography).

## Description of material

Shapley values are the most widely used black-box attribution method, and a natural baseline for mechanistic interpretability. This chapter builds them up from exact values on games where the right answer is known, through the fast approximations used in practice and interaction indices, to attributing model behaviour to tokens, image regions and training examples. Every method is checked against ground truth, and section 16.8 asks the key question for this course: when does a black-box attribution agree with a causal, mechanistic one?"""

render_chapter_home(__file__, INTRO)
