import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = (
    '## Description of material\n'
    '\n'
    "Vision-language models add a whole new set of questions: where does visual information enter the language model, when does the model stop looking at the image, and when it hallucinates an object, is it a perception failure or a language prior overriding what it saw? You'll build CLIP, SigLIP and a small captioning/VQA model from scratch on synthetic scenes where the ground truth is known, then apply the same tools to real models, always with background-patch, shuffled-image and text-only controls."
)

render_chapter_home(__file__, INTRO)
