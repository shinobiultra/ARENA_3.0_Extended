import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = r"""## TLDR (if you're in a hurry!)

**Prerequisites:** [1.2] Intro to Mech Interp; [0.2] CNNs & ResNets helps for the vision parts.

- **12.1 CLIP and SigLIP** and **12.3 Mini VLM from Scratch** are the core: build contrastive and generative vision-language models yourself first.
- Then **12.4 Visual Token Flow** takes apart real VLMs, and **12.5** studies hallucination and the competition between image and text.
- 12.2 (CLIP SAEs and typographic attacks), 12.6 (multimodal sparse features) and 12.7 (DINO and SAM) are deeper dives.

## Description of material

Vision-language models add a whole new set of questions: where does visual information enter the language model, when does the model stop looking at the image, and when it hallucinates an object, is it a perception failure or a language prior overriding what it saw? You'll build CLIP, SigLIP and a small captioning/VQA model from scratch on synthetic scenes where the ground truth is known, then apply the same tools to real models, always with background-patch, shuffled-image and text-only controls."""

render_chapter_home(__file__, INTRO)
