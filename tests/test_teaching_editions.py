"""Exact prose-only projection contracts, including the real foundation parent."""
import hashlib
import json
from pathlib import Path

import pytest

from infrastructure.extension import blank_practice
from infrastructure.extension import teaching_editions as editions

ROOT = Path(__file__).resolve().parents[1]


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def notebook():
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"historical": {"run": "old", "flag": True}},
        "cells": [
            {"cell_type": "markdown", "id": "question", "metadata": {"custom": "retained"},
             "attachments": {"a.png": {"image/png": "AAAA"}},
             "source": ["# A question\n", "Original explanation.\n"]},
            {"cell_type": "code", "id": "core", "metadata": {"historical_execution": "old"},
             "source": ["# Original explanation.\n", "raise RuntimeError('never execute')\n"],
             "execution_count": 7,
             "outputs": [{"output_type": "stream", "name": "stdout", "text": ["saved result\n"]}]},
            {"cell_type": "markdown", "id": "empty", "metadata": {"keep": True}, "source": ["\n"]},
            {"cell_type": "markdown", "id": "play", "metadata": {}, "source": "Try it yourself."},
        ],
    }


@pytest.fixture
def lane(tmp_path):
    for relative in blank_practice._CONFIG:
        config = tmp_path / relative
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("fixture: true\n")
    paths = {"page": "chapter5/instructions/pages/01_[5.1]_Fixture.md",
             "exercises": "chapter5/exercises/part1_fixture/5.1_Fixture_exercises.ipynb",
             "reference": "chapter5/exercises/part1_fixture/5.1_Fixture_solutions.ipynb"}
    raw = {"page": b"# A question\nOriginal explanation.\n=== NEW CHAPTER ===\nTry it yourself.\n",
           "exercises": encoded(notebook()), "reference": encoded(notebook())}
    for kind, relative in paths.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw[kind])
    manifest = {
        "schema_version": 1, "section": "5.1",
        "sources": {kind: {"path": relative, "sha256": hashlib.sha256(raw[kind]).hexdigest()}
                    for kind, relative in paths.items()},
        "replacements": [{"old": "Original explanation.",
                          "new": "Predict the sign first; compare the saved result with the shuffled control.",
                          "counts": {"page": 1, "exercises": 1, "reference": 1}}],
        "rationale": ["F-1: explain the prediction and control beside the core exercise"],
    }
    manifest_path = tmp_path / "mastery_hub/teaching_editions/5_1.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(encoded(manifest))
    row = {"number": "5.1", "available": True, **{kind: tmp_path / path for kind, path in paths.items()}}
    return tmp_path, row, raw, manifest, manifest_path


def load(lane):
    root, row, _, _, _ = lane
    return editions.load_edition(root, row)


def test_deterministic_readonly_adjacent_projection_and_exact_inverse(lane):
    root, _, raw, _, _ = lane
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    spec = load(lane)
    page = editions.apply_page(spec, raw["page"].decode())
    assert "Predict the sign first" in page
    assert page.count("=== NEW CHAPTER ===") == 1
    for kind in ("exercises", "reference"):
        projected = editions.project_notebook(spec, raw[kind], kind)
        assert projected == editions.project_notebook(spec, raw[kind], kind)
        original = json.loads(raw[kind])
        derived = json.loads(projected)
        assert derived["cells"][1:] == original["cells"][1:]
        assert derived["cells"][0]["id"] == "question"
        assert derived["cells"][0]["metadata"] == original["cells"][0]["metadata"]
        assert derived["cells"][0]["attachments"] == original["cells"][0]["attachments"]
        assert "Predict the sign first" in "".join(derived["cells"][0]["source"])
        assert derived["metadata"]["historical"] == original["metadata"]["historical"]
        provenance = derived["metadata"]["arena_teaching_edition"]
        assert provenance["execution_claim"] == "none"
        assert provenance["source_sha256"] == hashlib.sha256(raw[kind]).hexdigest()
        assert provenance["manifest_sha256"] == hashlib.sha256(lane[4].read_bytes()).hexdigest()
        assert editions.restore_notebook(spec, projected, raw[kind], kind) == raw[kind]
    assert before == {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("change", [
    lambda n: n["cells"][0].__setitem__("source", "tampered prose"),
    lambda n: n["cells"][0]["metadata"].__setitem__("other", True),
    lambda n: n["cells"][0].__setitem__("attachments", {}),
    lambda n: n["cells"][0].__setitem__("id", "other-id"),
    lambda n: n["cells"][1].__setitem__("source", "pass"),
    lambda n: n["cells"][1].__setitem__("outputs", []),
    lambda n: n["cells"][1].__setitem__("execution_count", None),
    lambda n: n["cells"][1]["metadata"].__setitem__("historical_execution", "new"),
    lambda n: n["cells"].reverse(),
    lambda n: n["cells"].pop(),
    lambda n: n["metadata"]["historical"].__setitem__("flag", 1),
    lambda n: n["metadata"]["arena_teaching_edition"].__setitem__("source_sha256", "0" * 64),
])
def test_inverse_rejects_every_undeclared_object_change(lane, change):
    spec, raw = load(lane), lane[2]["reference"]
    value = json.loads(editions.project_notebook(spec, raw, "reference"))
    change(value)
    with pytest.raises(ValueError):
        editions.restore_notebook(spec, encoded(value), raw, "reference")


@pytest.mark.parametrize("kind", ["page", "exercises", "reference"])
def test_load_rejects_stale_source_pin(lane, kind):
    lane[1][kind].write_bytes(lane[2][kind] + b" ")
    with pytest.raises(ValueError, match=f"Stale.*{kind}"):
        load(lane)


@pytest.mark.parametrize("kind", ["exercises", "reference"])
def test_project_rechecks_canonical_bytes(lane, kind):
    with pytest.raises(ValueError, match="Stale"):
        editions.project_notebook(load(lane), lane[2][kind] + b" ", kind)


@pytest.mark.parametrize("old,count", [("anchor missing", 1), ("Original explanation.", 2)])
def test_wrong_preimage_or_occurrence_count_fails(lane, old, count):
    lane[3]["replacements"][0].update(old=old, counts=dict.fromkeys(("page", "exercises", "reference"), count))
    lane[4].write_bytes(encoded(lane[3]))
    spec = load(lane)
    with pytest.raises(ValueError, match="preimage count"):
        editions.apply_page(spec, lane[2]["page"].decode())
    with pytest.raises(ValueError, match="preimage count"):
        editions.project_notebook(spec, lane[2]["exercises"], "exercises")


def test_zero_means_explicitly_inapplicable_and_string_source_preserved(lane):
    lane[3]["replacements"][0]["counts"]["reference"] = 0
    lane[3]["replacements"].append({"old": "Try it yourself.", "new": "Try a changed-condition prediction.",
                                   "counts": {"page": 1, "exercises": 1, "reference": 0}})
    lane[4].write_bytes(encoded(lane[3]))
    spec = load(lane)
    n = json.loads(editions.project_notebook(spec, lane[2]["exercises"], "exercises"))
    assert n["cells"][-1]["source"] == "Try a changed-condition prediction."
    n = json.loads(editions.project_notebook(spec, lane[2]["reference"], "reference"))
    assert n["cells"] == json.loads(lane[2]["reference"])["cells"]


def test_empty_markdown_removed_only_when_explicit_and_inverse_recovers_it(lane):
    lane[3]["drop_empty_markdown_cells"] = True
    lane[4].write_bytes(encoded(lane[3]))
    spec, raw = load(lane), lane[2]["reference"]
    projected = editions.project_notebook(spec, raw, "reference")
    assert [c["id"] for c in json.loads(projected)["cells"]] == ["question", "core", "play"]
    assert editions.restore_notebook(spec, projected, raw, "reference") == raw


@pytest.mark.parametrize("change", [
    lambda m: m.__setitem__("unknown", "bad"),
    lambda m: m.__setitem__("schema_version", True),
    lambda m: m.__setitem__("section", "5.2"),
    lambda m: m.__setitem__("rationale", "not a list"),
    lambda m: m.__setitem__("drop_empty_markdown_cells", 1),
    lambda m: m["sources"]["page"].__setitem__("sha256", "z" * 64),
    lambda m: m["sources"]["page"].__setitem__("path", "../outside.md"),
    lambda m: m["sources"]["page"].__setitem__("path", "/outside.md"),
    lambda m: m["sources"]["page"].__setitem__("path", "chapter5/./other.md"),
    lambda m: m["sources"]["page"].__setitem__("path", "chapter5\\other.md"),
    lambda m: m["sources"]["page"].__setitem__("path", "chapter5/other.md"),
    lambda m: m["sources"]["page"].__setitem__("unbound", True),
    lambda m: m["replacements"][0].__setitem__("old", ""),
    lambda m: m["replacements"][0].__setitem__("new", "=== NEW CHAPTER ==="),
    lambda m: m["replacements"][0].__setitem__("regex", True),
    lambda m: m["replacements"][0]["counts"].__setitem__("page", True),
    lambda m: m["replacements"][0]["counts"].__setitem__("page", -1),
    lambda m: m["replacements"][0]["counts"].pop("page"),
    lambda m: m["replacements"][0].__setitem__("counts", dict.fromkeys(("page", "exercises", "reference"), 0)),
])
def test_manifest_is_strict(lane, change):
    change(lane[3])
    lane[4].write_bytes(encoded(lane[3]))
    with pytest.raises(ValueError):
        load(lane)


def test_duplicate_json_keys_and_nonfinite_values_rejected(lane):
    for raw in (b'{"schema_version":1,"schema_version":1}', b'{"schema_version":NaN}'):
        lane[4].write_bytes(raw)
        with pytest.raises(ValueError):
            load(lane)


@pytest.mark.parametrize("target", ["page", "manifest", "source_directory", "root"])
def test_symlinks_rejected_in_every_path_component(lane, tmp_path, target):
    root, row, _, _, manifest = lane
    if target == "root":
        alias = root.parent / (root.name + "-alias")
        alias.symlink_to(root, target_is_directory=True)
        row = {**row, **{kind: alias / row[kind].relative_to(root) for kind in ("page", "exercises", "reference")}}
        with pytest.raises(ValueError, match="symlink"):
            editions.load_edition(alias, row)
        return
    path = manifest if target == "manifest" else (row["page"].parent if target == "source_directory" else row["page"])
    backup = path.with_name(path.name + ".real")
    path.rename(backup)
    path.symlink_to(backup, target_is_directory=backup.is_dir())
    with pytest.raises(ValueError, match="symlink"):
        load(lane)


def test_absent_manifest_is_only_optional_input(lane):
    lane[4].unlink()
    assert load(lane) is None
    lane[4].write_bytes(encoded(lane[3]))
    lane[1]["reference"].unlink()
    with pytest.raises(ValueError, match="Missing"):
        load(lane)


def test_nonexistent_root_is_not_an_absent_optional_manifest(tmp_path):
    with pytest.raises(ValueError, match="Missing"):
        editions.load_edition(tmp_path / "absent-checkout", {"number": "5.1"})


@pytest.mark.parametrize("change", [
    lambda n: n["cells"][1].__setitem__("id", "question"),
    lambda n: n["cells"][1].__setitem__("execution_count", True),
    lambda n: n["cells"][1].__setitem__("execution_count", 1.0),
    lambda n: n["cells"][1].__setitem__("undeclared", "invalid schema"),
    lambda n: n.__setitem__("nbformat_minor", 99),
])
def test_pinned_notebooks_must_still_pass_strict_schema(lane, change):
    value = notebook()
    change(value)
    raw = encoded(value)
    lane[1]["reference"].write_bytes(raw)
    lane[3]["sources"]["reference"]["sha256"] = hashlib.sha256(raw).hexdigest()
    lane[4].write_bytes(encoded(lane[3]))
    with pytest.raises(ValueError):
        editions.project_notebook(load(lane), raw, "reference")


def test_captured_manifest_mutation_and_notebook_provenance_collision_rejected(lane):
    spec = load(lane)
    spec["replacements"][0]["new"] = "changed captured authority"
    with pytest.raises(ValueError, match="manifest changed"):
        editions.project_notebook(spec, lane[2]["reference"], "reference")
    n = notebook()
    n["metadata"]["arena_teaching_edition"] = {}
    raw = encoded(n)
    lane[1]["reference"].write_bytes(raw)
    lane[3]["sources"]["reference"]["sha256"] = hashlib.sha256(raw).hexdigest()
    lane[4].write_bytes(encoded(lane[3]))
    with pytest.raises(ValueError, match="collision"):
        editions.project_notebook(load(lane), raw, "reference")


def test_blank_parent_is_validated_and_original_inverse_still_works(lane):
    spec, raw = load(lane), lane[2]["exercises"]
    authority = blank_practice.config_identity(lane[0])
    base = blank_practice.project_download(raw, lane[3]["sources"]["exercises"]["path"], authority)
    derived = editions.project_notebook(spec, raw, "exercises", base)
    assert json.loads(derived)["metadata"]["arena_blank_practice"] == json.loads(base)["metadata"]["arena_blank_practice"]
    assert json.loads(derived)["cells"][2] == json.loads(base)["cells"][2]
    assert editions.restore_notebook(spec, derived, raw, "exercises", base) == base
    restored = blank_practice.restore_download(base, raw, lane[3]["sources"]["exercises"]["path"], authority)
    assert restored == json.loads(raw)
    changed = json.loads(base)
    changed["cells"][2]["source"] = "pass"
    with pytest.raises(ValueError):
        editions.project_notebook(spec, raw, "exercises", encoded(changed))
    with pytest.raises(ValueError, match="Only exercises"):
        editions.project_notebook(spec, lane[2]["reference"], "reference", base)


def test_composed_parent_configuration_is_independent_authority(lane):
    spec, raw = load(lane), lane[2]["exercises"]
    relative = lane[3]["sources"]["exercises"]["path"]
    wrong = {path: "f" * 64 for path in blank_practice._CONFIG}
    base = blank_practice.project_download(raw, relative, wrong)
    with pytest.raises(ValueError, match="configuration differs"):
        editions.project_notebook(spec, raw, "exercises", base)
    base = blank_practice.project_download(raw, relative, blank_practice.config_identity(lane[0]))
    (lane[0] / blank_practice._CONFIG[0]).write_text("changed: true\n")
    with pytest.raises(ValueError, match="Configuration changed"):
        editions.project_notebook(spec, raw, "exercises", base)


@pytest.mark.parametrize("section", ["0.6", "1.6"])
def test_real_foundation_six_cells_compose_without_rebinding_pins(tmp_path, section):
    from mastery_hub.read import inventory

    row = next(row for row in inventory(ROOT) if row["number"] == section)
    for relative_config in blank_practice._CONFIG:
        target = tmp_path / relative_config
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative_config).read_bytes())
    raw = row["exercises"].read_bytes()
    base, _ = blank_practice.download(ROOT, section, inventory)
    base_object = json.loads(base)
    relative = row["exercises"].relative_to(ROOT).as_posix()
    # Capture only the three exact canonical parents into the fixture checkout.
    sources = {}
    for kind in ("page", "exercises", "reference"):
        path = row[kind].relative_to(ROOT).as_posix()
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(row[kind].read_bytes())
        sources[kind] = {"path": path, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    first = next(c for c in json.loads(raw)["cells"] if c["cell_type"] == "markdown" and "".join(c["source"]).strip())
    anchor = "".join(first["source"])
    document = {"schema_version": 1, "section": section, "sources": sources,
                "replacements": [{"old": anchor, "new": anchor + "\nMake a prediction, then inspect the matched control.\n",
                                  "counts": {"page": 0, "exercises": 1, "reference": 0}}]}
    path = tmp_path / f"mastery_hub/teaching_editions/{section.replace('.', '_')}.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(encoded(document))
    fixture_row = {**row, **{kind: tmp_path / source["path"] for kind, source in sources.items()}}
    spec = editions.load_edition(tmp_path, fixture_row)
    derived = editions.project_notebook(spec, raw, "exercises", base)
    after = json.loads(derived)
    assert after["cells"][-6:] == base_object["cells"][-6:]
    assert [c for c in after["cells"] if c["cell_type"] == "code"] == [c for c in base_object["cells"] if c["cell_type"] == "code"]
    assert after["metadata"]["arena_blank_practice"]["foundation_live"] == base_object["metadata"]["arena_blank_practice"]["foundation_live"]
    restored_base = editions.restore_notebook(spec, derived, raw, "exercises", base)
    assert restored_base == base
    assert blank_practice.restore_download(restored_base, raw, relative, blank_practice.config_identity(ROOT)) == json.loads(raw)


def guarded_lane(lane, *, old=None, new=None):
    old = old or (
        "def core(x):\n    return x + 1\n\n"
        "print(core(2))\n"
        "native_result = optional_native()\nprint(native_result)\n"
    )
    new = new or (
        "RUN_OPTIONAL_NATIVE = False\n"
        "def core(x):\n    return x + 1\n\n"
        "print(core(2))\n"
        "if RUN_OPTIONAL_NATIVE:\n    native_result = optional_native()\n    print(native_result)\n"
    )
    for kind in ("exercises", "reference"):
        value = notebook()
        value["cells"][1]["source"] = old
        raw = encoded(value)
        lane[2][kind] = raw
        lane[1][kind].write_bytes(raw)
        lane[3]["sources"][kind]["sha256"] = hashlib.sha256(raw).hexdigest()
    lane[3]["practice_code_replacements"] = [{"old": old, "new": new, "reason": "Native work is an explicit opt-in after the CPU exercise."}]
    lane[4].write_bytes(encoded(lane[3]))
    return old, new


def test_finite_practice_guard_keeps_cpu_core_default_and_reference_exact(lane, capsys):
    old, new = guarded_lane(lane)
    spec, raw = load(lane), lane[2]["exercises"]
    relative = lane[3]["sources"]["exercises"]["path"]
    base = blank_practice.project_download(raw, relative, blank_practice.config_identity(lane[0]))
    projected = editions.project_notebook(spec, raw, "exercises", base)
    value = json.loads(projected)
    before = json.loads(base)
    assert value["cells"][2] == {**before["cells"][2], "source": new}
    assert value["cells"][2]["outputs"] == []
    assert value["cells"][2]["execution_count"] is None
    info = value["metadata"]["arena_teaching_edition"]
    assert info["practice_plumbing_revised"] is True
    assert info["practice_code_replacements"][0]["reason"] == lane[3]["practice_code_replacements"][0]["reason"]
    assert info["execution_claim"] == "none"
    assert editions.restore_notebook(spec, projected, raw, "exercises", base) == base
    assert blank_practice.restore_download(base, raw, relative, blank_practice.config_identity(lane[0])) == json.loads(raw)
    for kind in ("exercises", "reference"):
        canonical_edition = json.loads(editions.project_notebook(spec, lane[2][kind], kind))
        assert canonical_edition["cells"][1] == json.loads(lane[2][kind])["cells"][1]
        assert "practice_plumbing_revised" not in canonical_edition["metadata"]["arena_teaching_edition"]
    def forbidden_native():
        raise AssertionError("The optional native branch must stay disabled")
    namespace = {"optional_native": forbidden_native}
    exec(compile(new, "<tiny-practice-fixture>", "exec"), namespace)
    assert namespace["core"](5) == 6
    assert capsys.readouterr().out == "3\n"


@pytest.mark.parametrize("new", [
    "RUN_OPTIONAL_NATIVE = False\ndef core(x):\n    return x + 2\nprint(core(2))\nif RUN_OPTIONAL_NATIVE:\n    native_result = optional_native()\n    print(native_result)\n",
    "RUN_OPTIONAL_NATIVE = False\ndef core(x):\n    return x + 1\nprint(core(2))\nif RUN_OPTIONAL_NATIVE:\n    native_result = another_native()\n    print(native_result)\n",
    "RUN_OPTIONAL_NATIVE = False\nimport subprocess\ndef core(x):\n    return x + 1\nprint(core(2))\nif RUN_OPTIONAL_NATIVE:\n    native_result = optional_native()\n    print(native_result)\n",
    "RUN_OPTIONAL_NATIVE = True\ndef core(x):\n    return x + 1\nprint(core(2))\nif RUN_OPTIONAL_NATIVE:\n    native_result = optional_native()\n    print(native_result)\n",
    "RUN_OPTIONAL_NATIVE = False\ndef core(x):\n    return x + 1\nprint(core(2))\nif RUN_OPTIONAL_NATIVE:\n    native_result = optional_native()\n    print(native_result)\nelse:\n    pass\n",
    "RUN_OPTIONAL_NATIVE = False\ndef core(x):\n    return x + 1\nif RUN_OPTIONAL_NATIVE:\n    native_result = optional_native()\n    print(native_result)\nprint(core(2))\n",
])
def test_practice_transform_rejects_core_calls_imports_flags_else_and_reordering(lane, new):
    guarded_lane(lane, new=new)
    with pytest.raises(ValueError, match="Practice plumbing"):
        load(lane)


def test_guard_requires_disabled_flag_before_optional_code(lane):
    _, new = guarded_lane(lane)
    lane[3]["practice_code_replacements"][0]["new"] = new.removeprefix("RUN_OPTIONAL_NATIVE = False\n")
    lane[4].write_bytes(encoded(lane[3]))
    spec, raw = load(lane), lane[2]["exercises"]
    base = blank_practice.project_download(raw, lane[3]["sources"]["exercises"]["path"], blank_practice.config_identity(lane[0]))
    with pytest.raises(ValueError, match="earlier RUN_OPTIONAL_NATIVE=False"):
        editions.project_notebook(spec, raw, "exercises", base)


@pytest.mark.parametrize("fault", ["stale", "duplicate", "missing_reason", "unknown_key", "duplicate_entry"])
def test_practice_preimage_and_manifest_guards(lane, fault):
    guarded_lane(lane)
    item = lane[3]["practice_code_replacements"][0]
    if fault == "stale":
        item["old"] += "# stale preimage\n"
    elif fault == "duplicate":
        value = json.loads(lane[2]["exercises"])
        extra = dict(value["cells"][1], id="duplicate-source")
        value["cells"].append(extra)
        raw = encoded(value)
        lane[1]["exercises"].write_bytes(raw)
        lane[3]["sources"]["exercises"]["sha256"] = hashlib.sha256(raw).hexdigest()
    elif fault == "missing_reason":
        item["reason"] = " "
    elif fault == "unknown_key":
        item["regex"] = True
    else:
        lane[3]["practice_code_replacements"].append(dict(item))
    lane[4].write_bytes(encoded(lane[3]))
    with pytest.raises(ValueError):
        load(lane)


def test_guarded_derivative_tamper_still_rejects_and_blank_parent_is_required(lane):
    guarded_lane(lane)
    spec, raw = load(lane), lane[2]["exercises"]
    base = blank_practice.project_download(raw, lane[3]["sources"]["exercises"]["path"], blank_practice.config_identity(lane[0]))
    projected = json.loads(editions.project_notebook(spec, raw, "exercises", base))
    projected["cells"][2]["source"] = projected["cells"][2]["source"].replace("False", "True", 1)
    with pytest.raises(ValueError, match="exact finite projection"):
        editions.restore_notebook(spec, encoded(projected), raw, "exercises", base)
    parent = json.loads(base)
    parent["cells"][2]["outputs"] = [{"output_type": "stream", "name": "stdout", "text": "invented"}]
    with pytest.raises(ValueError):
        editions.project_notebook(spec, raw, "exercises", encoded(parent))


@pytest.mark.parametrize("section,key,old_default,new_default,allowed", [
    ("5.8", "ARENA_DIFFUSION_NUMERICAL_ONLY", "0", "1", True),
    ("5.1", "ARENA_DIFFUSION_NUMERICAL_ONLY", "0", "1", False),
    ("5.8", "ARENA_NUMERICAL_ONLY", "0", "1", False),
    ("5.8", "ARENA_DIFFUSION_NUMERICAL_ONLY", "1", "0", False),
])
def test_numerical_default_whitelist_is_exact(section, key, old_default, new_default, allowed):
    old = f"NUMERICAL_ONLY = os.environ.get({key!r}, {old_default!r}) == '1'\n"
    new = f"NUMERICAL_ONLY = os.environ.get({key!r}, {new_default!r}) == '1'\n"
    if allowed:
        editions._validate_practice_change(section, old, new)
    else:
        with pytest.raises(ValueError, match="finite setup whitelist"):
            editions._validate_practice_change(section, old, new)


def test_existing_text_study_can_only_change_boolean_default_to_false():
    editions._validate_practice_change("5.8", "RUN_TEXT_STUDY = True\n", "RUN_TEXT_STUDY = False\n")
    with pytest.raises(ValueError):
        editions._validate_practice_change("5.8", "RUN_TEXT_STUDY = False\n", "RUN_TEXT_STUDY = True\n")


def folding_lane(lane, source="def tree_curriculum_namespace():\n    return {'helper': True}\n"):
    for kind in ("exercises", "reference"):
        value = notebook()
        value["cells"][1]["source"] = source.splitlines(keepends=True)
        value["cells"][1]["metadata"]["jupyter"] = {"existing": "retained", "source_hidden": False}
        raw = encoded(value)
        lane[2][kind] = raw
        lane[1][kind].write_bytes(raw)
        lane[3]["sources"][kind]["sha256"] = hashlib.sha256(raw).hexdigest()
    lane[1]["number"] = lane[3]["section"] = "16.3"
    lane[3]["presentation_collapses"] = [{"source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                                          "reason": "Fold setup plumbing before the first student method."}]
    lane[4].unlink()
    path = lane[4].with_name("16_3.json")
    path.write_bytes(encoded(lane[3]))
    return path


@pytest.mark.parametrize("name", sorted(editions._FOLD_HELPERS))
def test_only_authorized_helper_presentation_metadata_changes_and_inverse_exact(lane, name):
    folding_lane(lane, f"def {name}():\n    return {{'helper': True}}\n")
    spec = load(lane)
    for kind in ("exercises", "reference"):
        raw = lane[2][kind]
        original = json.loads(raw)
        projected = editions.project_notebook(spec, raw, kind)
        after = json.loads(projected)
        expected = original["cells"][1]
        expected["metadata"]["jupyter"]["source_hidden"] = True
        assert after["cells"][1] == expected
        assert after["metadata"]["arena_teaching_edition"]["presentation_folding"] is True
        assert after["metadata"]["arena_teaching_edition"]["presentation_collapses"][0]["helper"] == name
        assert editions.restore_notebook(spec, projected, raw, kind) == raw
    raw = lane[2]["exercises"]
    base = blank_practice.project_download(raw, lane[3]["sources"]["exercises"]["path"], blank_practice.config_identity(lane[0]))
    projected = editions.project_notebook(spec, raw, "exercises", base)
    assert json.loads(projected)["cells"][2]["metadata"]["jupyter"]["source_hidden"] is True
    assert editions.restore_notebook(spec, projected, raw, "exercises", base) == base


@pytest.mark.parametrize("source", [
    "def student_core(x):\n    return x + 1\n",
    "def tree_curriculum_namespace():\n    raise NotImplementedError()\n",
    "def tree_curriculum_namespace():\n    return {}\nprint('extra execution')\n",
    "async def tree_curriculum_namespace():\n    return {}\n",
])
def test_presentation_cannot_fold_core_stub_or_additional_execution(lane, source):
    folding_lane(lane, source)
    with pytest.raises(ValueError, match="six complete 16.3 setup helpers"):
        editions.project_notebook(load(lane), lane[2]["reference"], "reference")


def test_presentation_rejects_other_sections_stale_hashes_and_tampering(lane):
    path = folding_lane(lane)
    lane[3]["section"] = lane[1]["number"] = "5.1"
    lane[4].write_bytes(encoded(lane[3]))
    with pytest.raises(ValueError, match="restricted to section 16.3"):
        load(lane)
    lane[3]["section"] = lane[1]["number"] = "16.3"
    spec = load(lane)
    raw = lane[2]["reference"]
    after = json.loads(editions.project_notebook(spec, raw, "reference"))
    after["cells"][1]["metadata"]["jupyter"]["existing"] = "tampered"
    with pytest.raises(ValueError, match="exact finite projection"):
        editions.restore_notebook(spec, encoded(after), raw, "reference")
    lane[3]["presentation_collapses"][0]["source_sha256"] = "0" * 64
    path.write_bytes(encoded(lane[3]))
    with pytest.raises(ValueError, match="one exact helper source"):
        editions.project_notebook(load(lane), raw, "reference")


def test_presentation_requires_unique_exact_cell_and_preserves_existing_metadata(lane):
    path = folding_lane(lane)
    value = json.loads(lane[2]["reference"])
    value["cells"].append(dict(value["cells"][1], id="duplicate-helper-source"))
    raw = encoded(value)
    lane[1]["reference"].write_bytes(raw)
    lane[3]["sources"]["reference"]["sha256"] = hashlib.sha256(raw).hexdigest()
    path.write_bytes(encoded(lane[3]))
    with pytest.raises(ValueError, match="found 2"):
        editions.project_notebook(load(lane), raw, "reference")
