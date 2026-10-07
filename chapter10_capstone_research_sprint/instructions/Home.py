import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = r"""## TLDR (if you're in a hurry!)

Do this after at least one full chapter of the extension. The capstone teaches the research process with a worked example that runs in minutes, then hands you a menu of project ideas drawn from the whole course, plus templates for a pre-registration, a research log and a write-up.

## Description of material

Everything else in the course gives you the methods. This chapter is about using them to answer a question you don't already know the answer to, and doing it in a way you'd trust: choosing a question small enough to finish, deciding in advance what would count as evidence, separating exploratory from confirmatory analysis, running the controls, and writing up a negative result as carefully as a positive one."""

render_chapter_home(__file__, INTRO)
