import sys
from pathlib import Path

instructions_dir = Path(__file__).parent
arena_root_dir = instructions_dir.parent.parent
if str(arena_root_dir) not in sys.path:
    sys.path.append(str(arena_root_dir))

from arena_ext.streamlit_home import render_extension_home

render_extension_home(__file__)
