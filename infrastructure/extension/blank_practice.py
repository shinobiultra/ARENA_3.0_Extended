"""Read-only, deterministic blank Practice derivatives of configured notebooks.

No notebook normalization, execution, method import, filesystem write or network.
The inverse accepts exactly this finite projection and no unrelated mutations.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
import posixpath
import stat
from pathlib import Path, PurePosixPath
import re

_KEY = 'arena_blank_practice'
_CONFIG = ('infrastructure/extension/syllabus.yaml', 'infrastructure/core/config_extension.yaml')

_PROVIDER_ROOT = Path(__file__).absolute().parents[2]
_PROVIDER_PATH = 'mastery_hub/practice_addons/foundation_live_signatures/lesson.py'
_STUDY_CAUTIONS = 'docs/mastery/study_cautions_20261023.md'
_FOUNDATIONS = {
    'chapter0_fundamentals/exercises/part6_fake_interpretability_results_practice/0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_Practice_exercises.ipynb': {
        'section': '0.6',
    },
    'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure_practice/1.6_Local_Frontier_ML_Infrastructure_Practice_exercises.ipynb': {
        'section': '1.6',
    },
}


def _read_regular(root: Path, relative: str) -> bytes:
    """Capture a regular file through a retained nofollow descriptor chain."""
    root = Path(root).absolute()
    rel = PurePosixPath(relative)
    if not relative or rel.is_absolute() or '..' in rel.parts or rel.as_posix() != relative or '\\' in relative or '..' in root.parts:
        raise ValueError('Expected exact repository-relative source identity')
    descriptors = []
    edges = []
    primary_error = None
    identity = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    directory_identity = lambda s: (s.st_dev, s.st_ino, s.st_mode)
    try:
        directory = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        descriptors.append(directory)
        for part in (*root.parts[1:], *rel.parts[:-1]):
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
            descriptors.append(child)
            edges.append((directory, part, child))
            directory = child
        name = rel.parts[-1]
        before = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('Configured input is a symlink or nonregular file')
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=directory)
        descriptors.append(fd)
        if identity(os.fstat(fd)) != identity(before):
            raise ValueError('Configured input identity changed before read')
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError('Configured input size changed during read')
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1) or identity(os.fstat(fd)) != identity(before):
            raise ValueError('Configured input identity or size changed during read')
        if identity(os.stat(name, dir_fd=directory, follow_symlinks=False)) != identity(before):
            raise ValueError('Configured input path changed during read')
        for parent, name, child in edges:
            bound = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if directory_identity(bound) != directory_identity(os.fstat(child)) or not stat.S_ISDIR(bound.st_mode):
                raise ValueError('Configured directory binding changed during read')
        return b''.join(chunks)
    except OSError as exc:
        primary_error = ValueError('Configured input symlink, unreadable path or changed identity')
        raise primary_error from exc
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        close_errors = []
        for fd in reversed(descriptors):
            try:
                os.close(fd)
            except BaseException as exc:
                # A failed close may already have released this FD. Never retry.
                close_errors.append(exc)
        if close_errors:
            errors = ([primary_error] if primary_error is not None else []) + close_errors
            raise BaseExceptionGroup('Configured input descriptor cleanup failed', errors)


def _capture_configuration(root: Path) -> dict[str, bytes]:
    return {rel: _read_regular(Path(root), rel) for rel in _CONFIG}


def _configuration_identity(captured: dict[str, bytes]) -> dict[str, str]:
    return {rel: hashlib.sha256(raw).hexdigest() for rel, raw in captured.items()}


def config_identity(root: Path) -> dict[str, str]:
    return _configuration_identity(_capture_configuration(root))


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON object key')
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError('Nonfinite JSON number')


def _load(raw: bytes) -> dict:
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('Malformed notebook JSON') from exc
    _validate(value)
    return value


@lru_cache(maxsize=2)
def _validator(minor: int):
    # Read the installed official schema directly: nbformat's read/normalize APIs
    # can insert cell IDs. The schema validator does not import Jupyter runtime.
    from jsonschema import Draft4Validator
    spec = importlib.util.find_spec('nbformat')
    if spec is None or spec.origin is None:
        raise ValueError('Notebook schema package unavailable')
    schema = Path(spec.origin).parent / 'v4' / f'nbformat.v4.{minor}.schema.json'
    return Draft4Validator(json.loads(schema.read_text(encoding='utf-8')))


def _validate(value):
    if type(value) is not dict or type(value.get('nbformat')) is not int or value['nbformat'] != 4:
        raise ValueError('Unsupported notebook schema')
    minor = value.get('nbformat_minor')
    if type(minor) is not int or minor not in (2, 5):
        raise ValueError('Unsupported notebook minor schema')
    error = next(_validator(minor).iter_errors(value), None)
    if error is not None:
        raise ValueError(f'Invalid notebook schema at {list(error.path)}: {error.message}')
    ids = []
    for cell in value['cells']:
        if 'id' in cell:
            ident = cell['id']
            if type(ident) is not str or re.fullmatch('[a-zA-Z0-9_-]{1,64}', ident) is None:
                raise ValueError('Invalid cell ID')
            ids.append(ident)
        if cell['cell_type'] == 'code' and cell['execution_count'] is not None and type(cell['execution_count']) is not int:
            raise ValueError('Execution count must be an integer or null')
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate cell ID')


def _typed(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _provenance(raw: bytes, relative: str, authority: dict[str, str]) -> dict:
    rel = PurePosixPath(relative)
    if not relative or rel.is_absolute() or '..' in rel.parts or rel.as_posix() != relative or '\\' in relative or not relative.endswith('_exercises.ipynb'):
        raise ValueError('Invalid configured notebook identity')
    if type(authority) is not dict or set(authority) != set(_CONFIG) or any(type(x) is not str or re.fullmatch('[0-9a-f]{64}', x) is None for x in authority.values()):
        raise ValueError('Invalid configuration identity')
    return {'version': 1, 'kind': 'PUBLIC Practice', 'source_path': relative,
            'source_sha256': hashlib.sha256(raw).hexdigest(), 'configuration_sha256': dict(authority),
            'cleared_fields': ['code.outputs', 'code.execution_count'], 'execution_claim': 'none'}


def project(raw: bytes, relative: str, authority: dict[str, str]) -> bytes:
    source = _load(raw)
    if _KEY in source['metadata']:
        raise ValueError('Blank Practice provenance metadata collision')
    provenance = _provenance(raw, relative, authority)
    result = deepcopy(source)
    note = {'cell_type': 'markdown', 'metadata': {}, 'source': [
        '# PUBLIC Practice — blank code outputs\n',
        '\nThis is not blind evaluation. Source, help, worked answers and expected answers remain public. '
        'Only code outputs and execution counts have been cleared. Historical metadata is retained. '
        'This copy makes no fresh execution, scientific or learner acceptance claim.\n',
        f'\nSave this notebook beside the original `{relative}` in your checkout so relative setup and assets keep working.\n',
        f'\nSource SHA256: `{provenance["source_sha256"]}`. The original exercise and worked reference downloads remain unchanged.\n',
    ]}
    if source['nbformat_minor'] >= 5:
        note['id'] = 'blank-practice-' + provenance['source_sha256'][:24]
        if any(c.get('id') == note['id'] for c in source['cells']):
            raise ValueError('Blank Practice provenance cell ID collision')
    result['metadata'][_KEY] = provenance
    for cell in result['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    result['cells'].insert(0, note)
    _validate(result)
    return (json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False) + '\n').encode('utf-8')


def restore(projected: bytes, raw: bytes, relative: str, authority: dict[str, str]) -> dict:
    """Restore only enumerated code slots after exact typed projection comparison."""
    actual = _load(projected)
    expected = _load(project(raw, relative, authority))
    if _typed(actual) != _typed(expected):
        raise ValueError('Derivative differs outside the exact blank Practice projection')
    source = _load(raw)
    restored = deepcopy(actual)
    del restored['metadata'][_KEY]
    restored['cells'].pop(0)
    for index, before in enumerate(source['cells']):
        if before['cell_type'] == 'code':
            for field in ('outputs', 'execution_count'):
                restored['cells'][index][field] = deepcopy(before[field])
    if _typed(restored) != _typed(source):
        raise ValueError('Finite inverse failed to restore the exact typed source')
    return restored


def _foundation_cells(relative: str) -> tuple[list[dict], dict]:
    """Read the live-signature cells from the provider module (source strings only)."""
    pin = _FOUNDATIONS[relative]
    provider = _read_regular(_PROVIDER_ROOT, _PROVIDER_PATH)
    namespace = {'__name__': '_arena_foundation_provider', '__file__': _PROVIDER_PATH}
    # The provider imports only stdlib textwrap. It returns source strings; no
    # generated plotting source or taught callback runs here.
    exec(compile(provider, _PROVIDER_PATH, 'exec', dont_inherit=True), namespace)
    cells = namespace['additions'](pin['section'])
    if type(cells) is not list or len(cells) != 6:
        raise ValueError('Foundation provider must return six cells')
    return cells, {'section': pin['section'], 'provider_path': _PROVIDER_PATH, 'cell_count': 6}


def project_download(raw: bytes, relative: str, authority: dict[str, str]) -> bytes:
    """Explicit V2 download feature; the generic V1 projection stays unchanged."""
    result = _load(project(raw, relative, authority))
    provenance = result['metadata'][_KEY]
    provenance['download_feature_version'] = 2
    provenance['study_cautions_path'] = _STUDY_CAUTIONS
    cautions_link = posixpath.relpath(_STUDY_CAUTIONS, posixpath.dirname(relative))
    result['cells'][0]['source'] = [
        '# PUBLIC Practice — start here\n',
        f'\nSave this blank copy beside `{PurePosixPath(relative).name}` in its original exercise directory. '
        'Then work through the cells and their immediate tests.\n',
        '\nHints and solutions are in the notebook as usual. Code outputs and execution counts are cleared.\n',
        f'\nRead the [study cautions and route notes]({cautions_link}) before choosing an exercise route.\n',
    ]
    if relative in _FOUNDATIONS:
        cells, live_provenance = _foundation_cells(relative)
        if result['nbformat_minor'] < 5:
            # Cell IDs exist only from nbformat 4.5; older generated notebooks omit them.
            cells = [{key: value for key, value in cell.items() if key != 'id'} for cell in cells]
        result['cells'].extend(cells)
        provenance['foundation_live'] = live_provenance
    _validate(result)
    return (json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False) + '\n').encode('utf-8')


def restore_download(projected: bytes, raw: bytes, relative: str, authority: dict[str, str]) -> dict:
    """Remove exactly the V2 note/provenance and six known appended cells."""
    actual = _load(projected)
    expected = _load(project_download(raw, relative, authority))
    if _typed(actual) != _typed(expected):
        raise ValueError('Derivative differs outside the exact V2 download projection')
    baseline = _load(project(raw, relative, authority))
    restored = deepcopy(actual)
    if relative in _FOUNDATIONS:
        del restored['cells'][-6:]
        del restored['metadata'][_KEY]['foundation_live']
    for field in ('download_feature_version', 'study_cautions_path'):
        del restored['metadata'][_KEY][field]
    restored['cells'][0] = deepcopy(baseline['cells'][0])
    return restore((json.dumps(restored, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'),
                   raw, relative, authority)


def download(root: Path, section: str, inventory) -> tuple[bytes, str]:
    """Called only with the reader's configured inventory, never a requested path."""
    root = Path(root)
    if type(section) is not str or re.fullmatch(r'[0-9]+\.[0-9]+', section) is None:
        raise ValueError('Unknown configured lesson identity')
    captured = _capture_configuration(root)
    authority = _configuration_identity(captured)
    matches = [row for row in inventory(root, configuration_bytes=captured) if row['number'] == section and row['available']]
    if len(matches) != 1:
        raise ValueError('Unknown or unavailable configured lesson')
    relative = matches[0]['exercises'].relative_to(root).as_posix()
    raw = _read_regular(root, relative)
    path = PurePosixPath(relative)
    if config_identity(root) != authority:
        raise ValueError('Configuration changed while resolving source')
    return project_download(raw, relative, authority), path.name.removesuffix('_exercises.ipynb') + '_Blank_Practice_exercises.ipynb'
