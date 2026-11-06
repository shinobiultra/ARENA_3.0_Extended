"""Resolve actual local teaching links on every newly migrated source surface."""
import ast
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ('5.9', '8.6', '8.7', '9.4', '9.5', '13.1', '14.1', '14.5')
LOCAL_SUFFIXES = {'.ipynb', '.png', '.svg', '.jpg', '.jpeg'}


def markdown(path):
    if path.suffix == '.ipynb':
        return '\n'.join(''.join(cell['source']) for cell in json.loads(path.read_text())['cells']
                         if cell['cell_type'] == 'markdown')
    if path.suffix == '.py':
        return '\n'.join(node.value.value for node in ast.parse(path.read_text()).body
                         if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                         and isinstance(node.value.value, str))
    return path.read_text()


@pytest.mark.parametrize('section', SECTIONS)
def test_all_local_notebook_and_image_routes_resolve(section):
    config = yaml.safe_load((ROOT / 'infrastructure/core/config.yaml').read_text())
    mapping = config['conversion_map'][section]
    chapter = ROOT / config['chapter_names'][int(section.split('.')[0])]
    basename = mapping['streamlit_page'].split('_[', 1)[-1].replace(']', '')
    exercise = chapter / 'exercises' / mapping['exercise_dir']
    master = ROOT / 'infrastructure/chapters' / chapter.name / (
        'master_' + section.replace('.', '_') + '.py')
    paths = [master, master.with_suffix('.ipynb'),
             chapter / 'instructions/pages' / (mapping['streamlit_page'] + '.md'),
             exercise / (basename + '_exercises.ipynb'),
             exercise / (basename + '_solutions.ipynb')]
    for path in paths:
        assert path.is_file(), path
        text = markdown(path)
        urls = re.findall(r'''<img\b[^>]*\bsrc=["']([^"']+)["']''', text, re.I)
        urls += re.findall(r'\[[^\]]*\]\(([^)]+)\)', text)
        for url in urls:
            parsed = urlsplit(url)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            if Path(parsed.path).suffix.lower() not in LOCAL_SUFFIXES:
                continue
            target = (path.parent / unquote(parsed.path)).resolve()
            assert target.is_relative_to(ROOT), (path, url, target)
            assert target.is_file(), (path, url, target)
