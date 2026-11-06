"""Source/JSON engineering only; never import or execute either taught lesson."""
from pathlib import Path
from copy import deepcopy
import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
from urllib.parse import unquote, urlsplit

import pytest
from nbformat.validator import get_validator

ROOT = Path(__file__).resolve().parents[2]
UNIT = ROOT / "mastery_hub/units/pilot_practice"


def generator():
    spec = importlib.util.spec_from_file_location("tested_pilot_projection", UNIT / "generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def protocol():
    return json.loads((UNIT / "protocol.json").read_text())


def files(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def clean_unit(tmp_path):
    target = tmp_path / "unit"
    target.mkdir()
    for name in ["protocol.json", "original_sources.tar.gz", "original_source_manifest.json"]:
        shutil.copyfile(UNIT / name, target / name)
    return target


@pytest.mark.parametrize("section", ["7.1", "9.1"])
def test_whole_original_typed_object_inverse_and_schema(section):
    g = generator()
    row = protocol()["sections"][section]
    original = g.read_json((ROOT / row["case"]["exercise_path"]).read_bytes())
    projected = g.read_json((UNIT / row["projection"]["output_name"]).read_bytes())
    mapping = g.read_json((UNIT / "projection_map.json").read_bytes())["sections"][section]
    assert g.typed(g.invert(projected, mapping)) == g.typed(original)
    assert projected["metadata"] == original["metadata"]
    assert (projected["nbformat"], projected["nbformat_minor"]) == (4, 2)
    assert not list(get_validator(4, 2).iter_errors(projected))
    for cell in mapping["cells"]:
        a = original["cells"][cell["source_index"]]
        b = projected["cells"][cell["projected_index"]]
        restored = deepcopy(b)
        if a["cell_type"] == "code":
            assert b["outputs"] == [] and b["execution_count"] is None
            restored["outputs"] = a["outputs"]
            restored["execution_count"] = a["execution_count"]
        assert g.typed(restored) == g.typed(a)
        assert b.get("id") == a.get("id")
        assert b["source"] == a["source"]


@pytest.mark.parametrize("section", ["7.1", "9.1"])
def test_optional_opening_later_commitment_and_control_reflection(section):
    g = generator()
    row = protocol()["sections"][section]
    case = row["case"]
    book = json.loads((UNIT / row["projection"]["output_name"]).read_text())
    text = "".join(book["cells"][0]["source"])
    assert book["cells"][0]["metadata"]["pilot_practice"]["role"] == "manual-forecast"
    assert "| Confidence, 0–1 | ______ |" in text
    assert "no software-enforced freeze" in text and "no claim of unseen mastery" in text
    assert "expected-output dropdowns" in text and "record that exposure" in text
    assert "Complete this manual forecast before continuing" not in text
    assert "SHA-256 of the exact student-implemented method cells" not in text
    assert "Optional informal opening forecast" in text
    added = {cell.get("metadata", {}).get("pilot_practice", {}).get("role"): cell
             for cell in book["cells"] if "pilot_practice" in cell.get("metadata", {})}
    later = "".join(added["method-commitment"]["source"])
    assert case["question"] in later
    for control in case["controls"]:
        assert f"| {control} | ______ |" in later
    for value in case["config_fields"].values():
        assert value in later
    inverse = json.loads((UNIT / "projection_map.json").read_text())["sections"][section]
    original = json.loads((ROOT / case["exercise_path"]).read_text())
    for insertion in row["projection"]["insert_after"]:
        original_index = insertion["source_index"]
        item = inverse["cells"][original_index]
        extra = book["cells"][item["projected_index"] + 1]
        assert extra["metadata"]["pilot_practice"]["role"] == insertion["role"]
        assert book["cells"][item["projected_index"]]["source"] == original["cells"][original_index]["source"]
        if insertion["role"] == "controls-reflection":
            reflection = "".join(extra["source"])
            assert case["observation_prompt"] in reflection and case["transfer"] in reflection
            assert "| Observed primary metric/identities, with units and denominator | ______ |" in reflection
            assert "| Transfer confidence, reasoning and prior exposure | ______ |" in reflection


def test_grid_causal_qualification_is_adjacent_and_historical_claim_verbatim():
    p = protocol()["sections"]["7.1"]
    old = json.loads((ROOT / p["case"]["exercise_path"]).read_text())
    new = json.loads((UNIT / p["projection"]["output_name"]).read_text())
    m = json.loads((UNIT / "projection_map.json").read_text())["sections"]["7.1"]
    anchor = next(a for a in p["projection"]["insert_after"] if a["role"] == "grid-qualification")
    i = m["cells"][anchor["source_index"]]["projected_index"]
    assert new["cells"][i] == old["cells"][anchor["source_index"]]
    qualification = "".join(new["cells"][i + 1]["source"])
    assert "donor/target representation" in qualification
    assert "does not establish necessity of" in qualification
    assert "blocks 1–5 for native factual recall" in qualification
    assert "later preserved historical necessity wording" in "".join(new["cells"][0]["source"])


@pytest.mark.parametrize("section", ["7.1", "9.1"])
@pytest.mark.parametrize("mutation", ["source", "metadata", "id", "outputs", "count", "top_metadata", "cell_order"])
def test_inverse_rejects_any_learner_cell_or_whole_object_tamper(section, mutation):
    g = generator()
    p = protocol()["sections"][section]
    book = json.loads((UNIT / p["projection"]["output_name"]).read_text())
    mapping = json.loads((UNIT / "projection_map.json").read_text())["sections"][section]
    cell = next(c for c in book["cells"] if c["cell_type"] == "code")
    if mutation == "source":
        cell["source"].append("# learner edit\n")
    elif mutation == "metadata":
        cell["metadata"]["trusted"] = False
    elif mutation == "id":
        cell["id"] = "learner-id"
    elif mutation == "outputs":
        cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": ["own result"]}]
    elif mutation == "count":
        cell["execution_count"] = 1
    elif mutation == "top_metadata":
        book["metadata"]["learner_record"] = "saved forecast"
    else:
        book["cells"][2], book["cells"][3] = book["cells"][3], book["cells"][2]
    with pytest.raises(ValueError):
        g.invert(book, mapping)


def test_source_archive_contains_every_full_original_byte():
    g = generator()
    p = protocol()
    with tarfile.open(UNIT / "original_sources.tar.gz") as archive:
        members = archive.getmembers()
        assert len(members) == len(p["source_files"]) == 40
        for member in members:
            data = archive.extractfile(member).read()
            assert g.record(data) == p["source_files"][member.name]
            assert data == (ROOT / member.name).read_bytes()
    from scripts.evidence_bundles import validate_review_bundle
    contract = validate_review_bundle(UNIT / "original_sources.tar.gz")
    source = contract["provenance"]["source_data"]
    assert source["repository_path"] == "mastery_hub/units/pilot_practice/original_source_manifest.json"
    assert {k: source[k] for k in ["sha256", "size_bytes"]} == g.record((UNIT / "original_source_manifest.json").read_bytes())


def test_generation_check_missing_no_write_and_exact_noop(tmp_path):
    g = generator()
    unit = clean_unit(tmp_path)
    before = files(unit)
    assert not g.synchronize(ROOT, unit, check=True)
    assert files(unit) == before
    assert g.synchronize(ROOT, unit)
    snapshot = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in unit.iterdir() if p.is_file()}
    assert g.synchronize(ROOT, unit, check=True) and g.synchronize(ROOT, unit)
    assert snapshot == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in unit.iterdir() if p.is_file()}


@pytest.mark.parametrize("mutation", ["source", "metadata", "outputs", "count", "formatting", "record"])
def test_edited_late_copy_refused_before_any_other_file_creation(tmp_path, mutation):
    g = generator()
    unit = clean_unit(tmp_path)
    plan = g.planned(ROOT, unit)
    late = unit / "9_1_exercises.ipynb"
    book = json.loads(plan[late.name])
    code = next(c for c in book["cells"] if c["cell_type"] == "code")
    if mutation == "source":
        code["source"].append("# my answer\n")
    elif mutation == "metadata":
        code["metadata"]["trusted"] = False
    elif mutation == "outputs":
        code["outputs"] = [{"output_type": "stream", "name": "stdout", "text": "my run"}]
    elif mutation == "count":
        code["execution_count"] = 5
    elif mutation == "record":
        book["cells"][0]["source"].append("My forecast: 5\n")
    late.write_bytes(json.dumps(book, indent=3 if mutation == "formatting" else 2).encode())
    before = files(unit)
    with pytest.raises(ValueError, match="learner-edited"):
        g.synchronize(ROOT, unit)
    assert files(unit) == before
    assert not (unit / "7_1_exercises.ipynb").exists()
    assert not (unit / "projection_map.json").exists()


def test_exclusive_publication_refuses_racing_symlink(tmp_path, monkeypatch):
    g = generator()
    unit = clean_unit(tmp_path)
    outside = tmp_path / "my-existing-work"
    outside.write_text("keep this")
    original = g.os.open
    def raced(path, flags, mode):
        Path(path).symlink_to(outside)
        return original(path, flags, mode)
    monkeypatch.setattr(g.os, "open", raced)
    with pytest.raises(FileExistsError):
        g.synchronize(ROOT, unit)
    assert outside.read_text() == "keep this"
    assert not (unit / "9_1_exercises.ipynb").exists()


@pytest.mark.parametrize("mutation", ["archive", "case", "source"])
def test_changed_input_refuses_before_writes(tmp_path, mutation):
    g = generator()
    unit = clean_unit(tmp_path)
    source_root = ROOT
    if mutation == "archive":
        (unit / "original_sources.tar.gz").write_bytes(b"changed archive")
    elif mutation == "case":
        value = json.loads((unit / "protocol.json").read_text())
        value["sections"]["7.1"]["case"]["question"] += " changed"
        (unit / "protocol.json").write_text(json.dumps(value))
    else:
        source_root = tmp_path / "source"
        for rel in protocol()["source_files"]:
            target = source_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, target)
        target = source_root / "mastery_hub/extension_checkpoints.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "mastery_hub/extension_checkpoints.json", target)
        first = source_root / protocol()["sections"]["7.1"]["case"]["exercise_path"]
        first.write_bytes(first.read_bytes() + b" ")
    before = files(unit)
    with pytest.raises(ValueError, match="changed"):
        g.synchronize(source_root, unit)
    assert files(unit) == before


def test_preserved_methods_are_source_only_and_links_resolve():
    g = generator()
    tree = ast.parse((UNIT / "generate.py").read_text())
    imported = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imported += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert set(imported) <= {"__future__", "argparse", "copy", "hashlib", "json", "os", "pathlib", "tarfile"}
    for row in protocol()["sections"].values():
        first = json.loads((UNIT / row["projection"]["output_name"]).read_text())["cells"][0]
        for rel in [row["projection"][key] for key in ["original_exercises_link", "original_solutions_link", "original_page_link"]]:
            assert (UNIT / unquote(urlsplit(rel).path)).resolve().is_file()
        assert "no software-enforced freeze" in "".join(first["source"])
    assert g.synchronize(ROOT, UNIT, check=True)


@pytest.mark.parametrize("section", ["7.1", "9.1"])
def test_opening_guide_needs_no_projection_or_lesson_execution(tmp_path, section):
    before = files(UNIT)
    result = subprocess.run(
        [sys.executable, str(UNIT / "generate.py"), "--guide", section,
         "--source-root", str(tmp_path / "no-source-checkout")],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
    assert files(UNIT) == before


def test_new_opening_text_cannot_overwrite_existing_pinned_projection(monkeypatch):
    g = generator()
    before = files(UNIT)
    forecast = g.forecast
    monkeypatch.setattr(g, "forecast", lambda *args: forecast(*args) + "\nNew opening guidance\n")
    with pytest.raises(ValueError, match="learner-edited"):
        g.synchronize(ROOT, UNIT, check=True)
    assert files(UNIT) == before


@pytest.mark.parametrize("section", ["7.1", "9.1"])
def test_session_stop_and_commitment_precede_their_later_execution(section):
    g = generator()
    row = protocol()["sections"][section]
    book = json.loads((UNIT / row["projection"]["output_name"]).read_text())
    inverse = json.loads((UNIT / "projection_map.json").read_text())["sections"][section]
    added = {item["cell"]["metadata"]["pilot_practice"]["role"]: item["projected_index"]
             for item in inverse["added_markdown_cells"]}
    original = json.loads((ROOT / row["case"]["exercise_path"]).read_text())
    texts = ["".join(cell["source"]) for cell in original["cells"]]
    if section == "7.1":
        stop_boundary = next(i for i, text in enumerate(texts) if text.startswith("### Exercise - KL divergence"))
        check = next(i for i, text in enumerate(texts) if text.startswith("tests.test_patchscope_controls("))
        run = next(i for i, text in enumerate(texts) if text.startswith("controls = patchscope_controls("))
    else:
        stop_boundary = next(i for i, text in enumerate(texts) if text.startswith("# 2️⃣ Ablation"))
        check = next(i for i, text in enumerate(texts) if "tests.test_sweep_directions(" in text and original["cells"][i]["cell_type"] == "code")
        run = next(i for i, text in enumerate(texts) if text.startswith("sweep = sweep_directions("))
    position = {item["source_index"]: item["projected_index"] for item in inverse["cells"]}
    assert 0 < added["session-stop"] < position[stop_boundary]
    assert position[check] < added["method-commitment"] < position[run]
    assert book["cells"][added["session-stop"]]["cell_type"] == "markdown"
    assert book["cells"][added["method-commitment"]]["cell_type"] == "markdown"
