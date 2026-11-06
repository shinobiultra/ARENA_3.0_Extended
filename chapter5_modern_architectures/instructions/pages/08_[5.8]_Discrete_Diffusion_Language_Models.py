import platform
import sys
from pathlib import Path

import streamlit as st
import streamlit_antd_components as sac

# metadata should be a list of dicts like:
# [
#     {"title": "TMS", "icon": "1-circle-fill", "subtitle": "(99%)"},
#     {"title": "Bonus", "icon": "star", "subtitle": "(1%)"},
#     ...
# ]
metadata = [
    {'title': 'Exact copy grammar', 'icon': '1-circle-fill', 'subtitle': ''},
    {'title': 'Finite text diffusion', 'icon': '2-circle-fill', 'subtitle': ''},
    {'title': 'Matched models and measurements', 'icon': '3-circle-fill', 'subtitle': ''},
    {'title': 'Bonus', 'icon': 'star', 'subtitle': ''},
]
chapter_name = 'chapter5_modern_architectures'
chapter_name_long = 'Chapter 5 - Modern Architectures'
section_title = '[5.8] Discrete Diffusion Language Models'

pages_dir = Path(__file__).parent  # ARENA_3/chapter_name/instructions/pages
instructions_dir = pages_dir.parent  # ARENA_3/chapter_name/instructions
chapter_dir = instructions_dir.parent  # ARENA_3/chapter_name
arena_root_dir = chapter_dir.parent  # ARENA_3
if str(arena_root_dir) not in sys.path:
    sys.path.append(str(arena_root_dir))

from st_chat import display_chat_interface, display_content_with_exercise_chats, initialize_chat
from st_dependencies import generate_toc, styling
from infrastructure.extension.render_media import inline_page_media

markdown_content_file = Path(__file__).with_suffix(".md")

# TODO - uncomment the line below, so page is cached (while developing it's easier to comment this out)
# if st.session_state.get("content_filename", None) != str(markdown_content_file):
st.session_state["content_filename"] = str(markdown_content_file)
st.session_state["content"] = inline_page_media(markdown_content_file.read_text(encoding="utf-8"), markdown_content_file).split(
    "=== NEW CHAPTER ==="
)

IS_LOCAL = platform.processor() != ""
DEBUG = False

styling(chapter_name_long, DEBUG)
st.info(
    "We've moved to [learn.arena.education](https://learn.arena.education)! "
    "The material is the same, but the new site comes with additional features "
    "including a course planner and a more advanced AI assistant.",
    icon="🚀",
)
initialize_chat()

with st.sidebar:
    CHAPTER_SELECT = sac.steps(
        [sac.StepsItem(title="Home", icon="house"), *[sac.StepsItem(**data) for data in metadata]],
        size="small",
        return_index=True,
    )
    chapter_content = st.session_state["content"][int(CHAPTER_SELECT)]

    all_content = "\n".join(st.session_state["content"])
    display_chat_interface(all_content, section_title)

    table_of_contents = generate_toc(chapter_content)
    st.markdown(table_of_contents, unsafe_allow_html=True)

display_content_with_exercise_chats(chapter_content, section_title)
