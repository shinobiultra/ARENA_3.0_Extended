"""Pure preservation recipes: exact source, immutable evidence, no taught imports."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from scripts.evidence_bundles import file_record, seal_review_bundle


def digest(value):
    return hashlib.sha256(value).hexdigest()


def fingerprints(book):
    return [{"cell_type": c["cell_type"], "source_sha256": digest("".join(c["source"]).encode())}
            for c in book["cells"]]


@pytest.fixture
def recipe(tmp_path):
    identity = {"section": "8.6", "chapter": "chapter8_automated_circuits",
                "exercise_dir": "part6_sparse_feature_circuits",
                "page": "06_[8.6]_Sparse_Feature_Circuits_and_SHIFT"}
    master = tmp_path / "copied_masters/master_8_6.py"
    master.parent.mkdir()
    master.write_bytes(b"# Immutable fixture master\n")
    old = {"cells": [
        {"cell_type": "markdown", "source": ["> **Colab: old**"], "metadata": {"old": True},
         "execution_count": None, "outputs": []},
        {"cell_type": "code", "source": ["value = 1  \n"], "metadata": {"historical": True},
         "id": "method", "execution_count": 3,
         "outputs": [{"output_type": "stream", "name": "stdout", "text": ["historical fixture\n"]}]},
    ], "metadata": {"historical": True}, "nbformat": 4, "nbformat_minor": 5}
    raw = copy.deepcopy(old)
    raw["cells"][0]["source"] = ["> **Colab: generated** start the guided workbook #section-8-6"]
    raw["cells"][1]["source"] = ["value = 1"]
    module = "value = 1\n"
    bindings = {}
    for label, book in (("before", old), ("raw", raw)):
        manifest = tmp_path / (label + ".json")
        manifest.write_text(json.dumps({"fixture": label}))
        archive = tmp_path / (label + ".tar.gz")
        with tarfile.open(archive, "w:gz") as handle:
            for name, data in {"ex.ipynb": json.dumps(book).encode(),
                               "sol.ipynb": json.dumps(book).encode(),
                               "solutions.py": module.encode()}.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                handle.addfile(info, io.BytesIO(data))
        seal_review_bundle(archive, provenance={
            "kind": "historical_data_packaging", "measurement_claim": "Fabricated source-only fixture",
            "source_data": {"absolute_path": str(manifest), "repository_path": manifest.name,
                            **file_record(manifest)},
        })
        bindings[label] = {"path": archive.name, "sha256": digest(archive.read_bytes()),
                           "contract_sha256": digest(Path(str(archive) + ".review-bundle.json").read_bytes()),
                           "manifest_path": manifest.name, "manifest_sha256": digest(manifest.read_bytes())}
    contracts = tmp_path / "contracts.json"
    contract = {"schema": "arena.extension-source-preservation.v1", "sections": {"8.6": {
        "identity": identity, "master_sha256": digest(master.read_bytes()), "archives": bindings,
        "books": {kind: {"member": member, "before_sha256": digest(json.dumps(old).encode()),
                          "raw_cells": fingerprints(raw), "id_additions": [{"cell": 0, "id": "preserved-fixture"}],
                          "compatibility_removals": [
                              {"cell": 0, "field": "execution_count", "value": None},
                              {"cell": 0, "field": "outputs", "value": []}]}
                  for kind, member in (("exercises", "ex.ipynb"), ("solutions", "sol.ipynb"))},
        "modules": {"solutions": {"member": "solutions.py", "sha256": digest(module.encode())}},
    }}}
    contracts.write_text(json.dumps(contract))
    return {"root": tmp_path, "contracts_path": contracts, "master_path": master,
            "identity": identity, "raw_notebooks": {"exercises": json.dumps(raw), "solutions": json.dumps(raw)},
            "raw_modules": {"solutions": module}}, old, raw


def restore(args):
    from infrastructure.extension.source_preservation import restore_notebooks
    return restore_notebooks(**args)


def test_whole_historical_objects_preserved_except_exact_generated_navigation(recipe):
    args, old, raw = recipe
    original_args = copy.deepcopy(args)
    result = restore(args)
    expected = copy.deepcopy(old)
    expected["cells"][0]["source"] = raw["cells"][0]["source"]
    expected["cells"][0].pop("execution_count")
    expected["cells"][0].pop("outputs")
    expected["cells"][0]["id"] = "preserved-fixture"
    assert json.loads(result["exercises"]) == expected
    assert json.loads(result["solutions"]) == expected
    assert args == original_args


@pytest.mark.parametrize("change", ["master", "identity", "source_whitespace", "method", "module",
                                    "archive", "sidecar", "manifest", "duplicate_navigation"])
def test_registered_mismatch_raises_without_output_writes(recipe, change):
    args, _, raw = recipe
    if change == "master":
        args["master_path"].write_text("# changed copied master\n")
    elif change == "identity":
        args["identity"]["exercise_dir"] = "wrong"
    elif change in {"source_whitespace", "method", "duplicate_navigation"}:
        altered = copy.deepcopy(raw)
        if change == "duplicate_navigation":
            altered["cells"].append(copy.deepcopy(altered["cells"][0]))
        else:
            altered["cells"][1]["source"] = ["value = 1 " if change == "source_whitespace" else "value = 2"]
        args["raw_notebooks"]["exercises"] = json.dumps(altered)
    elif change == "module":
        args["raw_modules"]["solutions"] += "# changed\n"
    else:
        suffix = {"archive": "before.tar.gz", "sidecar": "before.tar.gz.review-bundle.json",
                  "manifest": "before.json"}[change]
        path = args["root"] / suffix
        path.write_bytes(path.read_bytes() + b" ")
    sentinel = args["root"] / "published.ipynb"
    sentinel.write_text("preserve learner work")
    with pytest.raises(ValueError):
        restore(args)
    assert sentinel.read_text() == "preserve learner work"


def test_unregistered_section_uses_normal_converter(recipe):
    args, _, _ = recipe
    args["identity"]["section"] = "5.1"
    args["master_path"] = args["master_path"].with_name("master_5_1.py")
    assert restore(args) is None


def test_registered_master_cannot_fall_back_through_unknown_section(recipe):
    args, _, _ = recipe
    args["identity"]["section"] = "8.99"
    with pytest.raises(ValueError, match="registered master section identity"):
        restore(args)


@pytest.mark.parametrize("field,value", [("outputs", ["unexpected evidence"]),
                                        ("execution_count", 1), ("unexpected", True)])
def test_noncode_evidence_or_unknown_field_is_never_discarded(recipe, field, value):
    args, _, raw = recipe
    altered = copy.deepcopy(raw)
    altered["cells"][0][field] = value
    args["raw_notebooks"]["exercises"] = json.dumps(altered)
    with pytest.raises(ValueError):
        restore(args)


def test_recipe_cannot_overwrite_existing_code_id(recipe):
    args, _, _ = recipe
    manifest = json.loads(args['contracts_path'].read_text())
    manifest['sections']['8.6']['books']['exercises']['id_additions'].append(
        {'cell': 1, 'id': 'overwritten'})
    args['contracts_path'].write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        restore(args)


def test_missing_explicit_id_operation_is_rejected(recipe):
    args, _, _ = recipe
    manifest = json.loads(args['contracts_path'].read_text())
    manifest['sections']['8.6']['books']['exercises']['id_additions'] = []
    args['contracts_path'].write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        restore(args)


@pytest.mark.parametrize('mutation', ['source', 'output', 'count', 'top_metadata', 'cell_metadata',
                                     'bool_to_int', 'int_to_float'])
def test_dirty_existing_book_rejected_before_publishing(recipe, mutation):
    args, old, _ = recipe
    existing = copy.deepcopy(old)
    if mutation == 'source': existing['cells'][1]['source'] = ['value = 999']
    elif mutation == 'output': existing['cells'][1]['outputs'][0]['text'] = ['unreviewed answer']
    elif mutation == 'count': existing['cells'][1]['execution_count'] = 999
    elif mutation == 'top_metadata': existing['metadata']['widgets'] = {'stale': True}
    elif mutation == 'bool_to_int': existing['metadata']['historical'] = 1
    elif mutation == 'int_to_float': existing['cells'][1]['execution_count'] = 3.0
    else: existing['cells'][1]['metadata']['learner_work'] = True
    args['existing_notebooks'] = {'exercises': None, 'solutions': json.dumps(existing)}
    sentinel = args['root'] / 'published.ipynb'
    sentinel.write_text('learner work survives')
    with pytest.raises(ValueError, match='published notebook has unreviewed changes'):
        planned = restore(args)
        sentinel.write_text(planned['exercises'])
    assert sentinel.read_text() == 'learner work survives'


def test_exact_three_existing_states_accepted(recipe):
    args, old, raw = recipe
    planned = restore(args)
    navigation_only = copy.deepcopy(old)
    navigation_only['cells'][0]['source'] = raw['cells'][0]['source']
    for state in [old, navigation_only, json.loads(planned['solutions'])]:
        args['existing_notebooks'] = {'exercises': json.dumps(state), 'solutions': json.dumps(state)}
        assert restore(args) == planned


@pytest.mark.parametrize('registered', [True, False])
def test_cli_preserves_registered_objects_and_retains_unregistered_postprocessing(tmp_path, monkeypatch, registered):
    # Execute only the real CLI orchestration AST with controlled source-only tools.
    import argparse
    import ast
    import fnmatch
    import types
    import sys
    from infrastructure.extension.build import normalize_generated_notebook
    source = Path('infrastructure/extension/build.py').read_text()
    main = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    masters = tmp_path / 'masters'; masters.mkdir(); (masters / 'master_8_6.py').write_text('# fixture')
    book = tmp_path / 'chapter/exercises/part/8.6_Fixture_exercises.ipynb'
    book.parent.mkdir(parents=True)
    payload = json.dumps({'cells': [{'cell_type': 'markdown', 'source': ['a\n', 'b\n'], 'id': 'prose', 'metadata': {}},
                                   {'cell_type': 'code', 'source': ['x = 1  \n'], 'id': 'original',
                                    'execution_count': 4, 'metadata': {'stored_widget': 'kept'},
                                    'outputs': [{'output_type': 'stream', 'name': 'stdout', 'text': ['kept\n']}]}],
                          'metadata': {'widgets': {'original': True}}, 'nbformat': 4, 'nbformat_minor': 5})
    solutions = book.with_name('8.6_Fixture_solutions.ipynb')
    calls = []
    fake = types.SimpleNamespace(exercises_dir=book.parent, colab_name='8.6_Fixture', source_preservation_applied=registered)
    def generate(*args):
        book.write_text(payload); solutions.write_text(payload)
        for path in [book, solutions]:
            normalize_generated_notebook(path)
        return fake
    def record(name):
        def run(*args): calls.append(name); return True
        return run
    config = {'conversion_map': {'8.6': {'streamlit_page': '06_[8.6]_Fixture', 'exercise_dir': 'part'}},
              'chapter_names': {8: 'chapter'}}
    namespace = {'argparse': argparse, 'fnmatch': fnmatch, 'ROOT': tmp_path, 'MASTERS': masters,
                 'load_config': lambda: config, 'nbformat': types.SimpleNamespace(read=lambda *a, **kw: {}),
                 'generate': generate, 'carry_over_exercise_provenance': record('provenance'),
                 'carry_over_outputs': record('outputs'), 'inline_media': record('inline'),
                 'rewrite_page_media': record('page'), 'summarise': lambda master: 'fixture', '__doc__': 'fixture'}
    exec(compile(ast.Module(body=[main], type_ignores=[]), '<controlled CLI>', 'exec'), namespace)
    monkeypatch.setattr(sys, 'argv', ['build', '8.6'])
    namespace['main']()
    assert calls == (['page'] if registered else ['provenance', 'outputs', 'inline', 'page'])
    if registered:
        assert json.loads(book.read_text()) == json.loads(payload)
        assert json.loads(solutions.read_text()) == json.loads(payload)


def test_inherited_writer_cannot_overwrite_dirty_published_book(recipe, monkeypatch):
    from infrastructure.extension import build
    from infrastructure.extension import source_preservation
    args, old, _ = recipe
    ipynb = args['master_path'].with_suffix('.ipynb')
    ipynb.write_text('{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}')
    identity = args['identity']
    master = build.ExtensionMasterFile(
        master_path=ipynb, chapter_dir=args['root'] / identity['chapter'],
        chapter_name_long='Fixture', exercise_dir_name=identity['exercise_dir'],
        streamlit_page_name=identity['page'])
    master.exercises_dir.mkdir(parents=True); master.instructions_dir.mkdir(parents=True)
    dirty = copy.deepcopy(old); dirty['cells'][1]['outputs'][0]['text'] = ['learner result']
    destination = master.exercises_dir / f'{master.colab_name}_solutions.ipynb'
    destination.write_text(json.dumps(dirty)); before = destination.read_bytes()
    sentinel = master.exercises_dir / 'solutions.py'; sentinel.write_text('# learner implementation')
    def raw_conversion(self, verbose=False):
        self.colab_ex_cells = args['raw_notebooks']['exercises']
        self.colab_soln_cells = args['raw_notebooks']['solutions']
        self.solutions_py_files = args['raw_modules']
        self.streamlit_md_file = ['fixture']; self.streamlit_py_file = '# fixture'
    monkeypatch.setattr(build.MasterFileData, 'create_arena_files', raw_conversion)
    original_restore = source_preservation.restore_notebooks
    def bound_restore(**kwargs):
        kwargs['root'] = args['root']; kwargs['contracts_path'] = args['contracts_path']
        return original_restore(**kwargs)
    monkeypatch.setattr(source_preservation, 'restore_notebooks', bound_restore)
    with pytest.raises(ValueError, match='published notebook has unreviewed changes'):
        master.generate_files(overwrite=True, verbose=False)
    assert destination.read_bytes() == before
    assert sentinel.read_text() == '# learner implementation'
    assert not (master.exercises_dir / f'{master.colab_name}_exercises.ipynb').exists()


def test_all_registered_restored_objects_survive_real_normalizer(tmp_path):
    from infrastructure.extension.build import normalize_generated_notebook
    from infrastructure.extension.source_preservation import restore_notebooks
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'infrastructure/extension/source_preservation_contracts.json').read_text())
    for section, recipe in manifest['sections'].items():
        with tarfile.open(root / recipe['archives']['raw']['path']) as archive:
            books = {kind: archive.extractfile(pin['member']).read().decode() for kind, pin in recipe['books'].items()}
            modules = {kind: archive.extractfile(pin['member']).read().decode() for kind, pin in recipe['modules'].items()}
        master = next((root / 'infrastructure/chapters').rglob('master_' + section.replace('.', '_') + '.py'))
        planned = restore_notebooks(master_path=master, identity=recipe['identity'], raw_notebooks=books,
                                    raw_modules=modules, root=root)
        for kind, payload in planned.items():
            path = tmp_path / (section + kind + '.ipynb'); path.write_text(payload)
            normalize_generated_notebook(path)
            assert json.loads(path.read_text()) == json.loads(payload)
