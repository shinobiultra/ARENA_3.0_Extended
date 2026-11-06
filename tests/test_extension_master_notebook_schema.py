"""Source-only saved-master compatibility: strict schema, no evidence changes."""
from copy import deepcopy
import json
from pathlib import Path

from nbformat.validator import get_validator
import pytest

from infrastructure.extension import build
from conversion.converters import _cells_to_notebook_data
from conversion.master_file import MasterFileData


def typed_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def errors(book):
    # Deliberately bypass nbformat.read/validate and their repair behavior.
    return list(get_validator(4, 5).iter_errors(book))


def master_instance(tmp_path, cls=build.ExtensionMasterFile):
    target = tmp_path / "master_5_99.ipynb"
    target.write_text("sentinel learner work", encoding="utf-8")
    target.with_suffix(".py").write_text(
        "# ! CELL TYPE: markdown\n# ! FILTERS: []\n# ! TAGS: []\n\nr'''\n# Fixture\n'''\n\n"
        "# ! CELL TYPE: code\n# ! FILTERS: []\n# ! TAGS: []\n\nvalue = 1  # preserved literal\n",
        encoding="utf-8",
    )
    return cls(target, tmp_path / "chapter", "Chapter 5", "part99", "99_[5.99]_Fixture")


def fixture_book():
    return {
        "nbformat": 4, "nbformat_minor": 2,
        "metadata": {"language_info": {"name": "python"}, "typed": [True, 1, 1.0, None],
                     "widgets": {"state": "unmodified"}},
        "cells": [
            {"cell_type": "markdown", "source": ["![kept](attachment:x.png)  \n", "évidence"],
             "metadata": {"tags": ["kept"], "typed": [False, 0, 0.0]},
             "attachments": {"x.png": {"image/png": "aGVsbG8="}},
             "execution_count": None, "outputs": []},
            {"cell_type": "code", "source": ["answer = 42  \n", "answer"], "id": "keep-code",
             "metadata": {"nested": {"value": [True, 1, 1.0]}}, "execution_count": 7,
             "outputs": [
                 {"output_type": "stream", "name": "stdout", "text": ["kept  \n"]},
                 {"output_type": "execute_result", "execution_count": 7,
                  "data": {"text/plain": ["42"], "application/json": {"typed": [True, 1, 1.0]}},
                  "metadata": {"kept": True}},
                 {"output_type": "display_data", "data": {"text/html": "<b>kept</b>"},
                  "metadata": {"isolated": True}},
                 {"output_type": "error", "ename": "Example", "evalue": "historical",
                  "traceback": ["preserved traceback"]},
             ]},
            {"cell_type": "raw", "source": "unprocessed source", "id": "keep-raw",
             "metadata": {"format": "text/plain"}},
        ],
    }


def canonical(book, basename="master_5_99.py"):
    from infrastructure.extension.master_notebook_schema import canonical_master_notebook
    return canonical_master_notebook(json.dumps(book, ensure_ascii=False), master_basename=basename)


def test_actual_small_master_saved_schema_and_determinism(tmp_path):
    master = master_instance(tmp_path)
    upstream_raw = json.loads(_cells_to_notebook_data(master.cells))
    assert errors(upstream_raw), "upstream fixture must reproduce the actual defect"
    master.master_py_to_ipynb(overwrite=True)
    first = master.master_path.read_bytes()
    result = json.loads(first)
    assert not errors(result)
    master.master_py_to_ipynb(overwrite=True)
    assert master.master_path.read_bytes() == first
    inverse = deepcopy(result)
    inverse["nbformat_minor"] = upstream_raw["nbformat_minor"]
    for old, new in zip(upstream_raw["cells"], inverse["cells"], strict=True):
        new.pop("id")
        if old["cell_type"] == "markdown":
            new["execution_count"] = None
            new["outputs"] = []
    assert typed_json(inverse) == typed_json(upstream_raw)
    assert len(first.splitlines()) > 10


def test_pure_cells_fixture_saved_schema(tmp_path):
    class FixtureMaster(build.ExtensionMasterFile):
        @property
        def cells(self):
            return [{"cell_type": "markdown", "source": ["Evidence survives"],
                     "metadata": {"typed": [True, 1, 1.0]}, "execution_count": None, "outputs": []},
                    {"cell_type": "code", "source": ["answer = 42"], "metadata": {},
                     "execution_count": 7,
                     "outputs": [{"output_type": "stream", "name": "stdout", "text": "kept\n"}]}]
    master = master_instance(tmp_path, FixtureMaster)
    master.master_py_to_ipynb(overwrite=True)
    assert not errors(json.loads(master.master_path.read_text()))


def test_complete_typed_inverse_preserves_all_content_and_supplied_ids():
    before = fixture_book()
    frozen = typed_json(before)
    text = canonical(before)
    result = json.loads(text)
    assert not errors(result)
    assert result["cells"][1]["id"] == "keep-code"
    assert result["cells"][2]["id"] == "keep-raw"
    inverse = deepcopy(result)
    inverse["nbformat_minor"] = before["nbformat_minor"]
    inverse["cells"][0].pop("id")
    inverse["cells"][0]["execution_count"] = None
    inverse["cells"][0]["outputs"] = []
    assert typed_json(inverse) == frozen
    assert typed_json(before) == frozen
    assert canonical(before) == text
    assert canonical(result) == text


def test_generated_ids_distinguish_repeated_cells_and_master_basenames():
    book = fixture_book()
    book["cells"] = [deepcopy(book["cells"][0]), deepcopy(book["cells"][0])]
    result = json.loads(canonical(book))
    ids = [c["id"] for c in result["cells"]]
    assert len(ids) == len(set(ids)) == 2
    assert [c["id"] for c in json.loads(canonical(book, "master_5_98.py"))["cells"]] != ids


@pytest.mark.parametrize("cell", [
    {"execution_count": 1}, {"execution_count": False}, {"execution_count": 0.0},
    {"outputs": None}, {"outputs": {}}, {"outputs": ["evidence"]}, {"unexpected": None},
    {"id": ""}, {"id": "has space"}, {"id": "x" * 65}, {"id": 7},
    {"metadata": []}, {"source": ["fine", 1]}, {"attachments": {"x": []}},
])
def test_rejects_invalid_markdown_without_dropping_any_field(cell):
    book = fixture_book()
    book["cells"][0].update(cell)
    with pytest.raises(ValueError):
        canonical(book)


@pytest.mark.parametrize("mutation", ["duplicate_id", "generated_collision", "code_field", "code_count",
                                      "output_count", "code_metadata", "output_metadata", "top_metadata",
                                      "top_field", "version_bool", "minor_float", "future_minor", "cell_type"])
def test_rejects_invalid_or_ambiguous_payload(mutation):
    book = fixture_book()
    if mutation == "duplicate_id": book["cells"][0]["id"] = "keep-code"
    elif mutation == "generated_collision":
        generated = json.loads(canonical(book))["cells"][0]["id"]
        book["cells"][1]["id"] = generated
    elif mutation == "code_field": book["cells"][1]["unknown"] = "do not drop"
    elif mutation == "code_count": book["cells"][1]["execution_count"] = 7.0
    elif mutation == "output_count": book["cells"][1]["outputs"][1]["execution_count"] = 7.0
    elif mutation == "code_metadata": book["cells"][1]["metadata"] = []
    elif mutation == "output_metadata": book["cells"][1]["outputs"][1]["metadata"] = []
    elif mutation == "top_metadata": book["metadata"] = []
    elif mutation == "top_field": book["unrecognized"] = {"important": True}
    elif mutation == "version_bool": book["nbformat"] = True
    elif mutation == "minor_float": book["nbformat_minor"] = 2.0
    elif mutation == "future_minor": book["nbformat_minor"] = 6
    else: book["cells"][0]["cell_type"] = "unsupported"
    with pytest.raises(ValueError):
        canonical(book)


@pytest.mark.parametrize("raw", ['{"cells": [], "cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 2}',
                                  '{"cells": [], "metadata": {"x": NaN}, "nbformat": 4, "nbformat_minor": 2}'])
def test_duplicate_keys_and_non_json_numbers_are_rejected(raw):
    from infrastructure.extension.master_notebook_schema import canonical_master_notebook
    with pytest.raises(ValueError):
        canonical_master_notebook(raw, master_basename="master_5_99.py")


@pytest.mark.parametrize("ignore_errors", [False, True])
def test_invalid_raw_payload_raises_before_writing_existing_master(tmp_path, monkeypatch, ignore_errors):
    master = master_instance(tmp_path)
    book = fixture_book()
    book["cells"][0]["outputs"] = [{"preserve": "unexpected evidence"}]
    import conversion.converters as converters
    import conversion.master_file as original_module
    def malformed(_cells): return json.dumps(book)
    monkeypatch.setattr(converters, "_cells_to_notebook_data", malformed)
    monkeypatch.setattr(original_module, "_cells_to_notebook_data", malformed)
    with pytest.raises(ValueError):
        master.master_py_to_ipynb(overwrite=True, ignore_errors=ignore_errors)
    assert master.master_path.read_bytes() == b"sentinel learner work"


def test_no_overwrite_rejects_before_reading_master_source(tmp_path):
    master = master_instance(tmp_path)
    master.master_py_path.unlink()
    with pytest.raises(FileExistsError):
        master.master_py_to_ipynb(overwrite=False)
    assert master.master_path.read_bytes() == b"sentinel learner work"


def test_original_converter_behavior_unchanged(tmp_path):
    master = master_instance(tmp_path, MasterFileData)
    master.master_py_to_ipynb(overwrite=True)
    book = json.loads(master.master_path.read_text())
    assert book["nbformat_minor"] == 2
    assert errors(book)
    assert book["cells"][0]["outputs"] == []
    assert "id" not in book["cells"][0]


def small_source(section):
    def cell(kind, text):
        if kind == "markdown": text = "r'''\n" + text + "\n'''"
        return f"# ! CELL TYPE: {kind}\n# ! FILTERS: []\n# ! TAGS: []\n\n{text}\n\n"
    return "".join([
        cell("markdown", '[{"title": "Source-only fixture", "icon": "1-circle-fill", "subtitle": ""}]'),
        cell("markdown", f"# [{section}] Source-only fixture"),
        cell("markdown", "# Introduction"),
        cell("markdown", "# 1️⃣ Source-only fixture"),
        cell("code", 'raise AssertionError("source-only build must never execute taught code")'),
    ])


def isolated_generation(tmp_path, monkeypatch, section):
    # Candidate harness uses the actual repository core, then the same fixture
    # root mutation as the portable tests after integration.
    core = build.CORE
    masters = tmp_path / "infrastructure/chapters"
    masters.mkdir(parents=True)
    source = masters / f"master_{section.replace('.', '_')}.py"
    source.write_text(small_source(section), encoding="utf-8")
    contracts = tmp_path / "infrastructure/extension/source_preservation_contracts.json"
    contracts.parent.mkdir()
    contracts.write_text('{"schema": "arena.extension-source-preservation.v1", "sections": {}}')
    chapter = int(section.split(".")[0])
    config = {"conversion_map": {section: {"exercise_dir": "part_fixture",
                                             "streamlit_page": f"99_[{section}]_Fixture"}},
              "chapter_names": {chapter: "chapter_fixture"},
              "chapter_names_long": {chapter: "Chapter fixture"}}
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "MASTERS", masters)
    monkeypatch.setattr(build, "CORE", core)
    monkeypatch.setattr(build, "load_config", lambda: config)
    return source, config


# These are fabricated schema fixtures, so they must not claim registered custody.
# Preserve the old case identities explicitly while assigning unregistered IDs.
@pytest.mark.parametrize("section", [
    pytest.param("5.99", id="5.99"),
    pytest.param("5.98", id="former-0.6-fixture-as-unregistered-5.98"),
    pytest.param("5.97", id="former-1.6-fixture-as-unregistered-5.97"),
])
def test_normal_generate_then_cli_source_only_master_is_valid(tmp_path, monkeypatch, section):
    import sys
    source, config = isolated_generation(tmp_path, monkeypatch, section)
    generated = build.generate(section, config)
    first = generated.master_path.read_bytes()
    assert not errors(json.loads(first))
    assert source.read_text() == small_source(section)
    assert not generated.source_preservation_applied
    monkeypatch.setattr(sys, "argv", ["build.py", section])
    build.main()
    assert generated.master_path.read_bytes() == first
    assert not errors(json.loads(generated.master_path.read_text()))
    assert 'raise AssertionError("source-only build must never execute taught code")' in (
        generated.exercises_dir / "solutions.py").read_text()


@pytest.mark.parametrize("section", ["8.6", "8.7", "9.4", "9.5", "13.1", "14.1", "14.5", "5.9"])
def test_registered_master_raw_outputs_are_byte_identical_and_restoration_unchanged(tmp_path, monkeypatch, section):
    import builtins
    source = build.find_master(section)
    config = build.load_config()
    mapping = config["conversion_map"][section]
    chapter = int(section.split(".")[0])
    target = tmp_path / source.with_suffix(".ipynb").name
    target.write_text("sentinel")
    target.with_suffix(".py").write_bytes(source.read_bytes())
    args = {"master_path": target, "chapter_dir": tmp_path / config["chapter_names"][chapter],
            "chapter_name_long": config["chapter_names_long"][chapter],
            "exercise_dir_name": mapping["exercise_dir"], "streamlit_page_name": mapping["streamlit_page"]}
    real_import = builtins.__import__
    real_read_text, real_read_bytes = Path.read_text, Path.read_bytes
    repo = build.ROOT.resolve()
    def source_only_import(name, *a, **kw):
        if name.split(".")[0] in {"torch", "numpy", "transformers", "arena_ext", "solutions"}:
            raise AssertionError("taught/scientific import during source-only build: " + name)
        return real_import(name, *a, **kw)
    def guard(path):
        if path.resolve().is_relative_to(repo) and "exercises" in path.parts:
            raise AssertionError("current published lesson read: " + str(path))
    def read_text(path, *a, **kw):
        guard(path)
        return real_read_text(path, *a, **kw)
    def read_bytes(path, *a, **kw):
        guard(path)
        return real_read_bytes(path, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", source_only_import)
    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    baseline = build.ExtensionMasterFile(**args)
    MasterFileData.master_py_to_ipynb(baseline, overwrite=True)
    old_raw = json.loads(target.read_text())
    MasterFileData.create_arena_files(baseline)
    attrs = ["colab_ex_cells", "colab_soln_cells", "solutions_py_files", "streamlit_md_file", "streamlit_py_file"]
    old_outputs = {name: deepcopy(getattr(baseline, name)) for name in attrs}
    candidate = build.ExtensionMasterFile(**args)
    candidate.master_py_to_ipynb(overwrite=True)
    canonical_raw = json.loads(target.read_text())
    assert not errors(canonical_raw)
    inverse = deepcopy(canonical_raw)
    inverse["nbformat_minor"] = 2
    for old, new in zip(old_raw["cells"], inverse["cells"], strict=True):
        new.pop("id")
        if old["cell_type"] == "markdown":
            new["execution_count"], new["outputs"] = None, []
    assert typed_json(inverse) == typed_json(old_raw)
    MasterFileData.create_arena_files(candidate)
    assert {name: getattr(candidate, name) for name in attrs} == old_outputs
    # The same real preservation recipes still accept all raw module/book bytes.
    restored = build.ExtensionMasterFile(**args)
    restored.create_arena_files()
    assert restored.source_preservation_applied
    assert restored.solutions_py_files == old_outputs["solutions_py_files"]
    assert not errors(json.loads(restored.colab_ex_cells))
    assert not errors(json.loads(restored.colab_soln_cells))


def test_successful_conversion_performs_one_already_valid_write(tmp_path, monkeypatch):
    master = master_instance(tmp_path)
    original = Path.write_text
    writes = []
    def checked_write(path, text, *args, **kwargs):
        if path == master.master_path:
            assert not errors(json.loads(text)), "invalid payload reached the filesystem"
            writes.append(text)
        return original(path, text, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", checked_write)
    master.master_py_to_ipynb(overwrite=True)
    assert len(writes) == 1
    assert master.master_path.read_text() == writes[0]


def surrogate_book(location):
    book = fixture_book()
    # All IDs are supplied so source encoding cannot fail earlier in ID hashing.
    book["cells"][0]["id"] = "keep-prose"
    if location == "metadata_value":
        book["metadata"]["unsafe"] = "\ud800"
    elif location == "metadata_key":
        book["metadata"]["\ud800"] = "preserve this value"
    elif location == "source_with_supplied_id":
        book["cells"][0]["source"] = ["\udfff"]
    else:
        raise AssertionError("unknown fixture location")
    return book


@pytest.mark.parametrize("location", ["metadata_value", "metadata_key", "source_with_supplied_id"])
def test_pure_helper_rejects_entire_unencodable_result_before_return(location):
    with pytest.raises(UnicodeEncodeError):
        canonical(surrogate_book(location))


@pytest.mark.parametrize("location", ["metadata_value", "metadata_key", "source_with_supplied_id"])
@pytest.mark.parametrize("ignore_errors", [False, True])
def test_unencodable_payload_preserves_actual_sentinel_without_any_write(tmp_path, monkeypatch, location, ignore_errors):
    master = master_instance(tmp_path)
    book = surrogate_book(location)
    import conversion.converters as converters
    original_write = Path.write_text
    attempted_writes = []
    def record_write(path, text, *args, **kwargs):
        if path == master.master_path:
            attempted_writes.append(path)
        return original_write(path, text, *args, **kwargs)
    # Keep the actual master parser, adapter, and writer; replace only the raw
    # converter result to introduce a schema-valid Unicode encoding failure.
    monkeypatch.setattr(converters, "_cells_to_notebook_data", lambda cells: json.dumps(book))
    monkeypatch.setattr(Path, "write_text", record_write)
    with pytest.raises(UnicodeEncodeError):
        master.master_py_to_ipynb(overwrite=True, ignore_errors=ignore_errors)
    assert master.master_path.read_bytes() == b"sentinel learner work"
    assert attempted_writes == []


def test_valid_supplementary_unicode_preserved_and_deterministic(tmp_path, monkeypatch):
    master = master_instance(tmp_path)
    book = fixture_book()
    book["cells"][0]["id"] = "keep-prose"
    text = "🧪𝄞𐐷"
    book["metadata"][text] = {"value": text}
    book["cells"][0]["source"] = [text]
    import conversion.converters as converters
    monkeypatch.setattr(converters, "_cells_to_notebook_data", lambda cells: json.dumps(book))
    master.master_py_to_ipynb(overwrite=True)
    first = master.master_path.read_bytes()
    assert text.encode("utf-8") in first
    parsed = json.loads(first)
    assert parsed["metadata"][text] == {"value": text}
    assert parsed["cells"][0]["source"] == [text]
    assert not errors(parsed)
    inverse = deepcopy(parsed)
    inverse["nbformat_minor"] = 2
    inverse["cells"][0]["execution_count"] = None
    inverse["cells"][0]["outputs"] = []
    assert typed_json(inverse) == typed_json(book)
    master.master_py_to_ipynb(overwrite=True)
    assert master.master_path.read_bytes() == first
