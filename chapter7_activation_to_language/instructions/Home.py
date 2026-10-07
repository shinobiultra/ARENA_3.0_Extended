import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = r"""## TLDR (if you're in a hurry!)

**Prerequisites:** [1.2] Intro to Mech Interp. [1.3.4] Activation Oracles is a good companion to 7.3.

- **7.1 Logit Lens, Tuned Lens and Patchscopes** is the foundation for the whole chapter: do it first.
- 7.2 (feature verbalizers) and 7.6 (interpretability agents) are about *automating explanations*; 7.3 (activation oracles), 7.4 (natural-language autoencoders) and 7.5 (predictive concept decoders) are about *training models to read activations*. Either path works after 7.1.

## Description of material

Interpretability results are ultimately communicated in words: "this feature fires on French text", "at layer 9 the model has retrieved Berlin". This chapter is about methods that go from activations to language directly. You'll start with the cheapest (decoding the residual stream with the model's own unembedding), move to trained readouts and to letting the model decode its own hidden states, then build systems that write and score explanations automatically, and finally train small models whose job is to answer questions about another model's activations. Throughout, the emphasis is on the controls that tell a real explanation from a plausible-sounding one."""

render_chapter_home(__file__, INTRO)
