import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = (
    "## TLDR (if you're in a hurry!)\n"
    '\n'
    '**Prerequisites:** [1.2] Intro to Mech Interp; [0.2] CNNs & ResNets helps for the vision parts.\n'
    '\n'
    '- **12.1 CLIP and SigLIP** and **12.3 Mini VLM from Scratch** are the core: build contrastive and generative vision-language models yourself first.\n'
    '- Open **12.3 Mini VLM from Scratch**, then select step **5 — Is a Real VLM Doing the Same?** for the existing Qwen3-VL-2B forced-choice scoring and decoder-state patching exercise, with matched image and token controls.\n'
    '- **12.4 Visual Token Flow in Real VLMs** remains unbuilt at its broader adapter/Patchscopes scope. **12.5** studies hallucination and the competition between image and text.\n'
    '- 12.2 (CLIP SAEs and typographic attacks), 12.6 (multimodal sparse features) and 12.7 (DINO and SAM) are deeper dives.\n'
    '\n'
    '## Description of material\n'
    '\n'
    'Vision-language models add a whole new set of questions: where does visual information enter the language model, when does the model stop looking at the image, and when it '
    "hallucinates an object, is it a perception failure or a language prior overriding what it saw? You'll build CLIP, SigLIP and a small captioning/VQA model from scratch on "
    'synthetic scenes where the ground truth is known, then apply the same tools to real models, always with background-patch, shuffled-image and text-only controls.'
)

render_chapter_home(__file__, INTRO)
