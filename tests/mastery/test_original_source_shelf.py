"""Original quiz answers stay accessible after an explicit exposure choice."""
from copy import deepcopy
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[2]


def test_original_quiz_answers_require_an_explicit_choice(tmp_path, monkeypatch):
    monkeypatch.setenv('ARENA_MASTERY_DATA_DIR', str(tmp_path / 'private'))
    app = AppTest.from_file(str(ROOT / 'mastery_hub/app.py'), default_timeout=20).run()
    assert not app.exception
    original_links = json.loads((ROOT / 'docs/mastery/source_links.json').read_text())['links']
    answer_urls = {item['url'] for item in original_links if '_answer_slides.pdf' in item['url']}
    assert answer_urls
    visible = '\n'.join(item.value for item in app.sidebar.markdown)
    assert not any(url in visible for url in answer_urls)
    assert all(item['url'] in visible for item in original_links if item['url'] not in answer_urls)

    progress = deepcopy(app.session_state['progress'])
    app.checkbox(key='show_original_quiz_answers').check().run()
    assert not app.exception
    exposed = '\n'.join(item.value for item in app.sidebar.markdown)
    assert all(url in exposed for url in answer_urls)
    assert app.session_state['progress'] == progress

    app.checkbox(key='show_original_quiz_answers').uncheck().run()
    assert not app.exception
    hidden = '\n'.join(item.value for item in app.sidebar.markdown)
    assert not any(url in hidden for url in answer_urls)
    assert app.session_state['progress'] == progress
