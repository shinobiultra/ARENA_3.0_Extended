import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = (
    '## Description of material\n'
    '\n'
    'Interpretability results are ultimately communicated in words: "this feature fires on French text", "at layer 9 the model has retrieved Berlin". This chapter is about methods that go from activations to language directly. You\'ll start with the cheapest (decoding the residual stream with the model\'s own unembedding), move to trained readouts and to letting the model decode its own hidden states, then build systems that write and score explanations automatically, and finally train small models whose job is to answer questions about another model\'s activations. Throughout, the emphasis is on the controls that tell a real explanation from a plausible-sounding one.'
)

render_chapter_home(__file__, INTRO)
