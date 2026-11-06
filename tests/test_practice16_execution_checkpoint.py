"""Pure DEV adversarial probes plus file-only accepted actual evidence checks; no execution."""

import ast
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").is_file())
HERE = Path(__file__).resolve().parents[1]


@pytest.fixture
def checkpoint(monkeypatch):
    name = "scripts.practice16_execution_checkpoint"
    spec = importlib.util.spec_from_file_location(name, HERE / "scripts/practice16_execution_checkpoint.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def encode(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode()


def record(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


@pytest.fixture
def dev_capsule(tmp_path, monkeypatch, checkpoint):
    """Fabricated DEV sentinels, deliberately neither 107 cells nor 36 observations."""
    c = checkpoint
    scope = "DEV_ONLY_NOT_RUNTIME_OR_PUBLICATION"
    target = "DEV_current/DEV_two_code_objects.json"
    identity = {"section": "DEV_ONLY_06", "chapter": "DEV_chapter", "exercise_dir": "DEV_objects", "page": "DEV_page"}
    before = {"scope": scope, "cells": [
        {"id": "DEV_alpha", "cell_type": "code", "source": ["DEV_LITERAL_ALPHA\n"], "metadata": {"DEV_keep": True}, "execution_count": None, "outputs": []},
        {"id": "DEV_text", "cell_type": "markdown", "source": ["DEV prose only"], "metadata": {}},
        {"id": "DEV_beta", "cell_type": "code", "source": ["DEV_LITERAL_BETA\n"], "metadata": {}, "execution_count": None, "outputs": []},
    ], "metadata": {"DEV_keep": 1}}
    after = copy.deepcopy(before)
    for index, count in [(0, 701), (2, 709)]:
        after["cells"][index]["execution_count"] = count
        after["cells"][index]["outputs"] = [{"output_type": "stream", "name": "stdout", "text": "DEV_SENTINEL_" + str(index)}]
    def event(kind, request, content):
        return {"msg_type": kind, "parent_header": {"msg_id": request}, "content": content}
    rows = []
    for index in (0, 2):
        cell = after["cells"][index]
        request = "DEV_request_" + str(index)
        rows.append({"index": index, "id": cell["id"], "source": cell["source"], "request_id": request,
            "kernel_id": "DEV_NO_KERNEL", "after_cell": copy.deepcopy(cell),
            "iopub": [event("status", request, {"execution_state": "busy"}),
                event("execute_input", request, {"code": "".join(cell["source"]), "execution_count": cell["execution_count"]}),
                event("stream", request, {"name": "stdout", "text": "DEV_SENTINEL_" + str(index)}),
                event("status", request, {"execution_state": "idle"})],
            "shell_reply": event("execute_reply", request, {"status": "ok", "execution_count": cell["execution_count"]})})
    for row in rows:
        raw = copy.deepcopy(row["after_cell"])
        raw["source"] = "".join(raw["source"])
        row["raw_after_cell"] = raw
        row["projection"] = {"rule": "exact_observed_source_join_and_text_split_v1",
            "canonical_source": copy.deepcopy(row["source"]), "raw_source": raw["source"], "output_fields": []}
    observations = {"schema": "DEV_observations", "scope": scope, "attempt_id": "DEV_NOT_AN_ATTEMPT", "rows": rows}
    receipt = {"schema": "DEV_receipt", "scope": scope, "attempt_id": "DEV_NOT_AN_ATTEMPT", "status": "completed",
        "claim": {"attempt_id": "DEV_NOT_AN_ATTEMPT", "claim_id": "DEV_NO_CLAIM", "exclusive": True, "consumed": True},
        "cleanup": {"kernel_id": "DEV_NO_KERNEL", "status": "reaped", "normal": True},
        "ordered_requests": [row["request_id"] for row in rows]}
    objects = {"before": before, "after": after, "observations": observations, "receipt": receipt}
    dependency = tmp_path / "DEV_inputs/DEV_source.txt"
    dependency.parent.mkdir()
    dependency.write_text("DEV source dependency; never executed\n")
    dependencies = {"DEV_inputs/DEV_source.txt": record(dependency.read_bytes())}
    signatures = [(index, before["cells"][index]["id"], hashlib.sha256(c._typed(before["cells"][index]["source"]).encode()).hexdigest()) for index in (0, 2)]
    for name, value in {
        "_TARGET": target, "_IDENTITY": identity, "_SCOPE": scope,
        "_SHAPE": (3, 2), "_BEFORE_RECORD": record(encode(before)),
        "_SIGNATURES": signatures, "_DEPENDENCIES": dependencies, "_PROVENANCE": {},
        "_SCHEMAS": {"contract": "DEV_contract", "sidecar": "DEV_sidecar", "observations": "DEV_observations", "receipt": "DEV_receipt"},
    }.items():
        monkeypatch.setattr(c, name, value, raising=False)

    def seal(extra_member=None, member_fault=None):
        payloads = {c._MEMBERS[role]: encode(value) for role, value in objects.items()}
        if extra_member:
            payloads[extra_member] = b"DEV unexpected member"
        files = {name: record(raw) for name, raw in payloads.items()}
        bundle = tmp_path / "DEV_evidence/bundle.tar.gz"
        bundle.parent.mkdir(exist_ok=True)
        with tarfile.open(bundle, "w:gz") as archive:
            for name, raw in payloads.items():
                member = tarfile.TarInfo(name)
                member.size = len(raw)
                if member_fault == "symlink" and name == "before.json":
                    member.type, member.linkname, member.size = tarfile.SYMTYPE, "../DEV_outside.json", 0
                    archive.addfile(member)
                else:
                    archive.addfile(member, io.BytesIO(raw))
                if member_fault == "duplicate" and name == "before.json":
                    archive.addfile(member, io.BytesIO(raw))
        manifest = {"scope": scope, "files": files}
        manifest_path = bundle.with_name("manifest.json")
        manifest_path.write_bytes(encode(manifest))
        sidecar = {"schema": "DEV_sidecar", "scope": scope, "archive": record(bundle.read_bytes()), "manifest": record(manifest_path.read_bytes()), "files": files}
        sidecar_path = bundle.with_name("sidecar.json")
        sidecar_path.write_bytes(encode(sidecar))
        descriptor = {"archive_path": "DEV_evidence/bundle.tar.gz", "archive": record(bundle.read_bytes()),
            "manifest_path": "DEV_evidence/manifest.json", "manifest": record(manifest_path.read_bytes()),
            "sidecar_path": "DEV_evidence/sidecar.json", "sidecar": record(sidecar_path.read_bytes()), "files": files}
        contract = {"schema": "DEV_contract", "scope": scope, "target": target, "identity": identity,
            "before": record(encode(objects["before"])), "after": record(encode(objects["after"])),
            "bundle": descriptor, "attempt_id": "DEV_NOT_AN_ATTEMPT", "source_pins": dependencies,
            "code_entries": [{"index": index, "before_cell": objects["before"]["cells"][index], "after_cell": objects["after"]["cells"][index]} for index in (0, 2)]}
        contract_path = tmp_path / "DEV_evidence/contract.json"
        contract_path.write_bytes(encode(contract))
        monkeypatch.setattr(c, "_ADOPTION", {"contract_path": "DEV_evidence/contract.json", "record": record(contract_path.read_bytes())})
        physical = tmp_path / target
        physical.parent.mkdir(exist_ok=True)
        physical.write_bytes(encode(objects["after"]))
        return contract_path, physical
    contract_path, physical = seal()
    return SimpleNamespace(c=c, root=tmp_path, objects=objects, seal=seal, physical=physical,
        contract=contract_path, identity=identity, dependency=dependency)


def test_unbound_guard_still_fails_before_reading_any_target(tmp_path, checkpoint, monkeypatch):
    monkeypatch.setattr(checkpoint, "_ADOPTION", None)
    monkeypatch.setattr(checkpoint, "_read", lambda *args: pytest.fail("pending gate read a target"))
    with pytest.raises(RuntimeError, match="pending"):
        checkpoint.preflight_practice16_publication(root=tmp_path, identity=checkpoint._IDENTITY, master_path=Path("master_1_6.py"))


def test_distinct_dev_physical_after_has_exact_whole_inverse(dev_capsule):
    d = dev_capsule
    checked = d.c._guard(d.root)
    assert checked is not None
    assert checked["before_bytes"] == encode(d.objects["before"])
    assert checked["after_bytes"] == d.physical.read_bytes()
    assert d.c.recover_practice16_checkpoint_bytes(d.root, d.c._TARGET, d.physical.read_bytes()) == encode(d.objects["before"])


@pytest.mark.parametrize("fault", ["output", "count", "id", "source", "source_representation", "metadata", "timestamp", "top_field", "markdown", "order", "cell_count"])
def test_physical_after_drift_is_never_hidden_by_checkpoint_view(dev_capsule, fault):
    d = dev_capsule
    changed = copy.deepcopy(d.objects["after"])
    if fault == "output": changed["cells"][0]["outputs"][0]["text"] += "edited"
    if fault == "count": changed["cells"][0]["execution_count"] += 1
    if fault == "id": changed["cells"][0]["id"] += "edited"
    if fault == "source": changed["cells"][0]["source"] = ["DEV changed"]
    if fault == "source_representation": changed["cells"][0]["source"] = "".join(changed["cells"][0]["source"])
    if fault == "metadata": changed["cells"][0]["metadata"]["unknown"] = True
    if fault == "timestamp": changed["cells"][0]["metadata"]["execution"] = {"timestamp": "DEV changed"}
    if fault == "top_field": changed["metadata"]["DEV_keep"] = True
    if fault == "markdown": changed["cells"][1]["source"] = ["DEV edit"]
    if fault == "order": changed["cells"].reverse()
    if fault == "cell_count": changed["cells"].pop()
    d.physical.write_bytes(encode(changed))
    with pytest.raises(ValueError):
        d.c.recover_practice16_checkpoint_bytes(d.root, d.c._TARGET, d.physical.read_bytes())


@pytest.mark.parametrize("fault", ["missing_row", "request_parent", "shell_status", "shell_count", "input_source", "raw_output", "after_cell", "claim", "cleanup", "request_order", "unknown_receipt", "bool_count", "source_representation", "unadopted_metadata"])
def test_hashes_and_success_flag_alone_do_not_accept_bad_dev_raw_objects(dev_capsule, fault):
    d = dev_capsule
    obs, receipt, after = d.objects["observations"], d.objects["receipt"], d.objects["after"]
    row = obs["rows"][0]
    if fault == "missing_row": obs["rows"].pop()
    if fault == "request_parent": row["shell_reply"]["parent_header"]["msg_id"] = "DEV other request"
    if fault == "shell_status": row["shell_reply"]["content"]["status"] = "error"
    if fault == "shell_count": row["shell_reply"]["content"]["execution_count"] += 1
    if fault == "input_source": row["iopub"][1]["content"]["code"] = "DEV other source"
    if fault == "raw_output": row["iopub"][2]["content"]["text"] = "DEV other output"
    if fault == "after_cell": row["after_cell"]["outputs"] = []
    if fault == "claim": receipt["claim"]["consumed"] = False
    if fault == "cleanup": receipt["cleanup"]["status"] = "unknown"
    if fault == "request_order": receipt["ordered_requests"].reverse()
    if fault == "unknown_receipt": receipt["ignored"] = "DEV unknown"
    if fault == "bool_count": after["cells"][0]["execution_count"] = True
    if fault == "source_representation": after["cells"][0]["source"] = "".join(after["cells"][0]["source"])
    if fault == "unadopted_metadata": after["cells"][0]["metadata"]["execution"] = {"timestamp": "DEV not adopted"}
    d.seal()
    with pytest.raises(ValueError):
        d.c._guard(d.root)


@pytest.mark.parametrize("fault", ["dependency", "archive", "sidecar", "manifest", "extra_member", "missing_contract", "old_physical", "old_supplied", "unknown_path", "symlink_target", "symlink_parent"])
def test_finite_capsule_dependencies_and_exact_physical_state_fail_closed(dev_capsule, fault):
    d = dev_capsule
    if fault == "dependency": d.dependency.write_text("DEV changed dependency")
    if fault in {"archive", "sidecar", "manifest"}:
        path = d.root / {"archive": "DEV_evidence/bundle.tar.gz", "sidecar": "DEV_evidence/sidecar.json", "manifest": "DEV_evidence/manifest.json"}[fault]
        path.write_bytes(path.read_bytes() + b"DEV changed")
    if fault == "extra_member": d.seal("DEV_unexpected.json")
    if fault == "missing_contract": d.contract.unlink()
    if fault == "old_physical": d.physical.write_bytes(encode(d.objects["before"]))
    if fault == "symlink_target":
        destination = d.root / "DEV_other.json"
        d.physical.rename(destination)
        d.physical.symlink_to(destination)
    if fault == "symlink_parent":
        parent = d.physical.parent
        other = parent.with_name("DEV_other_parent")
        parent.rename(other)
        parent.symlink_to(other, target_is_directory=True)
    if fault == "old_supplied":
        with pytest.raises(ValueError): d.c.recover_practice16_checkpoint_bytes(d.root, d.c._TARGET, encode(d.objects["before"]))
    elif fault == "unknown_path":
        with pytest.raises(ValueError): d.c.recover_practice16_checkpoint_bytes(d.root, "DEV_other.json", d.physical.read_bytes())
    else:
        with pytest.raises((ValueError, OSError)): d.c._guard(d.root)


@pytest.mark.parametrize("raw", [b'{"duplicate":1,"duplicate":2}', b'{"nonfinite":NaN}', b'{"overflow":1e999}'])
def test_strict_contract_json_rejects_duplicate_and_nonfinite_values(dev_capsule, raw, monkeypatch):
    d = dev_capsule
    d.contract.write_bytes(raw)
    monkeypatch.setattr(d.c, "_ADOPTION", {"contract_path": "DEV_evidence/contract.json", "record": record(raw)})
    with pytest.raises(ValueError): d.c._guard(d.root)


def test_ordinary_generate_rejects_before_even_master_stub_write(tmp_path, checkpoint, monkeypatch):
    monkeypatch.setattr(checkpoint, "_ADOPTION", None)
    build = HERE / "infrastructure/extension/build.py"
    node = next(node for node in ast.parse(build.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "generate")
    master_py = tmp_path / "master_1_6.py"
    master_py.write_text("DEV unexecuted source")
    writes = []
    class Master:
        def __init__(self, **kwargs):
            writes.append("constructed")
            self.exercises_dir, self.instructions_dir = tmp_path / "DEV_exercises", tmp_path / "DEV_pages"
            self.streamlit_page_name, self.colab_name = "DEV_page", "DEV_book"
        def master_py_to_ipynb(self, **kwargs): writes.append("master")
        def generate_files(self, **kwargs): writes.append("products")
    namespace = {"Path": Path, "ExtensionMasterFile": Master, "ROOT": tmp_path, "FIRST_EXTENSION_CHAPTER": 5,
        "find_master": lambda section: master_py, "prepare_instruction_page": lambda *args: None,
        "normalize_generated_notebook": lambda *args: None}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<candidate generate source boundary>", "exec"), namespace)
    config = {"conversion_map": {"1.6": {"exercise_dir": checkpoint._IDENTITY["exercise_dir"], "streamlit_page": checkpoint._IDENTITY["page"]}},
        "chapter_names": {1: checkpoint._IDENTITY["chapter"]}, "chapter_names_long": {1: "DEV title"}}
    with pytest.raises(RuntimeError, match="pending"):
        namespace["generate"]("1.6", config)
    assert writes == [] and not master_py.with_suffix(".ipynb").exists()


def test_q008_assert_asts_and_physical_execution_gate_are_unchanged():
    source = (ROOT / "tests/test_q008_practice_projection.py").read_text()
    actual = [ast.dump(node) for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Assert)]
    assert len(actual) == 31
    # Ordered AST digest pinned from the preserved original; no ignored artifact dependency.
    assert hashlib.sha256(json.dumps(actual, separators=(",", ":")).encode()).hexdigest() == '6cd0e9599bc4475ee07f962217736eef436c6338c44ed88059e431e7ce673fac'
    unchanged = {'infrastructure/extension/source_preservation.py': '0b28c52b73a3ede9c669e561b6dff0a3bf4d594ff45dee67ec96b59a36a511dc', 'infrastructure/extension/source_preservation_contracts.json': '1f05a959af920e750b739989357451719a90241f1220ace9f3f5fe72bfc58c4b', 'tests/test_extension_sections.py': '7ca6da2ddb17a38e2d78782b9dc31617cf1c76dc512284f479a27186fdcdb1b2', 'tests/test_extension_verification_assets.py': 'cd74e1d0a52def687467985cd0380c7a3a62c1d60c4b1d6083c8aed2b96b9c90'}
    for path, digest in unchanged.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest



@pytest.mark.parametrize("fault", ["duplicate", "symlink"])
def test_duplicate_and_nonregular_archive_members_fail_even_when_resigned_dev_only(dev_capsule, fault):
    d = dev_capsule
    d.seal(member_fault=fault)
    with pytest.raises(ValueError):
        d.c._guard(d.root)


def install_old_restore(monkeypatch, restore):
    monkeypatch.setitem(sys.modules, "infrastructure.extension.source_preservation", SimpleNamespace(restore_notebooks=restore))


def restore_arguments(d):
    return {"root": d.root, "identity": d.identity, "master_path": d.root / "master_1_6.py",
        "raw_notebooks": {"exercises": "DEV raw blank", "solutions": "DEV raw source"},
        "raw_modules": {"solutions": "DEV unexecuted module"},
        "existing_notebooks": {"exercises": "DEV current blank", "solutions": d.physical.read_bytes().decode()}}


def test_restore_passes_exact_historical_checkpoint_to_unchanged_old_boundary_then_reapplies_whole_actual_cells(dev_capsule, monkeypatch):
    d = dev_capsule
    arguments = restore_arguments(d)
    seen = []
    def old_restore(**kwargs):
        seen.append(kwargs)
        return {"exercises": "DEV old-recipe blank unchanged", "solutions": encode(d.objects["before"]).decode()}
    install_old_restore(monkeypatch, old_restore)
    result = d.c.restore_practice16_notebooks(**arguments)
    assert seen[0]["existing_notebooks"]["solutions"].encode() == encode(d.objects["before"])
    for key in ("root", "identity", "master_path", "raw_notebooks", "raw_modules"):
        assert seen[0][key] is arguments[key]
    assert result["exercises"] == "DEV old-recipe blank unchanged"
    assert result["solutions"].encode() == d.physical.read_bytes() == encode(d.objects["after"])


@pytest.mark.parametrize("fault", ["old_recipe_rejects", "wrong_restored_whole_object", "physical_changes_during_restore", "old_supplied_book"])
def test_old_recipe_failure_or_intervening_drift_never_returns_forward_payload(dev_capsule, monkeypatch, fault):
    d = dev_capsule
    arguments = restore_arguments(d)
    called = []
    def old_restore(**kwargs):
        called.append(True)
        if fault == "old_recipe_rejects":
            raise ValueError("DEV old recipe raw-module mismatch")
        before = copy.deepcopy(d.objects["before"])
        if fault == "wrong_restored_whole_object": before["metadata"]["DEV_keep"] = True
        if fault == "physical_changes_during_restore": d.physical.write_text("DEV concurrent edit")
        return {"exercises": "DEV blank", "solutions": encode(before).decode()}
    install_old_restore(monkeypatch, old_restore)
    if fault == "old_supplied_book": arguments["existing_notebooks"]["solutions"] = encode(d.objects["before"]).decode()
    with pytest.raises(ValueError): d.c.restore_practice16_notebooks(**arguments)
    assert called == ([] if fault == "old_supplied_book" else [True])


def test_other_section_restore_resolves_original_at_call_time_without_touching_pending_overlay(checkpoint, monkeypatch):
    arguments = {"identity": {"section": "DEV_other_section"}, "root": "DEV root", "master_path": "DEV_other_master.py"}
    first, second = object(), object()
    seen = []
    def original(**kwargs):
        seen.append(kwargs)
        return first
    install_old_restore(monkeypatch, original)
    assert checkpoint.restore_practice16_notebooks(**arguments) is first
    install_old_restore(monkeypatch, lambda **kwargs: second)
    assert checkpoint.restore_practice16_notebooks(**arguments) is second
    assert seen == [arguments]


def test_only_exact_local_json_target_gets_historical_view(dev_capsule):
    d = dev_capsule
    assert d.c.load_practice16_checkpoint_json(d.root, d.physical) == d.objects["before"]
    other = d.root / "DEV_unrelated.json"
    other.write_text('{"unchanged_normal_loader":true}')
    assert d.c.load_practice16_checkpoint_json(d.root, other) == {"unchanged_normal_loader": True}
    assert json.loads(d.physical.read_bytes()) == d.objects["after"]


def test_direct_master_writer_also_stops_before_import_or_write(tmp_path, checkpoint, monkeypatch):
    monkeypatch.setattr(checkpoint, "_ADOPTION", None)
    tree = ast.parse((HERE / "infrastructure/extension/build.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ExtensionMasterFile")
    node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "master_py_to_ipynb")
    namespace = {"ROOT": tmp_path}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<candidate direct master boundary>", "exec"), namespace)
    target = tmp_path / "master_1_6.ipynb"
    target.write_bytes(b"DEV original master sentinel")
    marker = SimpleNamespace(master_py_path=tmp_path / "master_1_6.py", master_path=target,
        colab_name="1.6_DEV", chapter_dir=tmp_path / checkpoint._IDENTITY["chapter"],
        exercise_dir_name=checkpoint._IDENTITY["exercise_dir"], streamlit_page_name=checkpoint._IDENTITY["page"])
    with pytest.raises(RuntimeError, match="pending"):
        namespace["master_py_to_ipynb"](marker, overwrite=True)
    assert target.read_bytes() == b"DEV original master sentinel"


def test_existing_normalizer_preserves_every_provided_code_object(dev_capsule):
    d = dev_capsule
    node = next(node for node in ast.parse((HERE / "infrastructure/extension/build.py").read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "normalize_generated_notebook")
    notebook = copy.deepcopy(d.objects["after"])
    notebook["cells"][1].update(execution_count=None, outputs=[])
    captured = []
    fake_nbformat = SimpleNamespace(read=lambda *args, **kwargs: SimpleNamespace(cells=notebook["cells"]), validate=lambda obj: None,
        write=lambda obj, path: captured.append(copy.deepcopy(obj.cells)))
    # Attribute access is nbformat's I/O boundary; no notebook/Jupyter package is imported.
    class Cell(dict):
        @property
        def cell_type(self): return self["cell_type"]
    notebook["cells"] = [Cell(cell) for cell in notebook["cells"]]
    namespace = {"Path": Path, "nbformat": fake_nbformat}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<unchanged normalizer source>", "exec"), namespace)
    namespace["normalize_generated_notebook"](d.physical)
    assert [cell for cell in captured[0] if cell["cell_type"] == "code"] == [cell for cell in d.objects["after"]["cells"] if cell["cell_type"] == "code"]
    assert "execution_count" not in captured[0][1] and "outputs" not in captured[0][1]


def test_helper_import_and_validation_have_no_jupyter_scientific_or_write_dependency(monkeypatch):
    import builtins
    original = builtins.__import__
    def checked_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"nbformat", "nbclient", "jupyter_client", "ipykernel", "torch", "numpy", "matplotlib"}
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", checked_import)
    path = HERE / "scripts/practice16_execution_checkpoint.py"
    spec = importlib.util.spec_from_file_location("DEV_inert_practice16_import", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert type(module._ADOPTION) is dict and type(module._PROVENANCE) is dict and len(module._PROVENANCE) > 97
    writes = {"write_text", "write_bytes", "mkdir", "unlink", "rename", "replace", "generate_files", "execute", "start_kernel"}
    tree = ast.parse(path.read_text())
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in writes]


def test_unknown_registered_identity_fails_before_a_pending_contract_read(tmp_path, checkpoint, monkeypatch):
    monkeypatch.setattr(checkpoint, "_read", lambda *args: pytest.fail("wrong identity reached contract read"))
    with pytest.raises(ValueError, match="identity"):
        checkpoint.preflight_practice16_publication(root=tmp_path, identity={**checkpoint._IDENTITY, "page": "DEV wrong"}, master_path=Path("master_1_6.py"))
    with pytest.raises(ValueError, match="identity"):
        checkpoint.preflight_practice16_publication(root=tmp_path, identity={"section": "7.1"}, master_path=Path("master_1_6.py"))
    assert checkpoint.preflight_practice16_publication(root=tmp_path, identity={"section": "7.1"}, master_path=Path("master_7_1.py")) is None


def test_dev_exact_source_join_and_native_text_split_are_explicitly_reversible(dev_capsule):
    d = dev_capsule
    row = d.objects["observations"]["rows"][0]
    raw = row["raw_after_cell"]["outputs"][0]["text"]
    published = raw.splitlines(True)
    row["after_cell"]["outputs"][0]["text"] = published
    d.objects["after"]["cells"][0]["outputs"][0]["text"] = published
    row["projection"]["output_fields"] = [{"output_index": 0, "path": ["text"], "raw": raw, "published": published}]
    d.seal()
    checked = d.c._guard(d.root)
    assert checked["after"]["cells"][0]["source"] == ["DEV_LITERAL_ALPHA\n"]
    assert checked["before"]["cells"][0]["outputs"] == []
    assert "".join(checked["after"]["cells"][0]["outputs"][0]["text"]) == raw


@pytest.mark.parametrize("fault", ["source_join", "raw_metadata", "unwitnessed_text", "wrong_text", "trailing_newline", "nontext_path", "duplicate_path", "unknown_projection"])
def test_dev_projection_rejects_unadopted_typed_changes_even_with_resigned_bytes(dev_capsule, fault):
    d = dev_capsule
    row = d.objects["observations"]["rows"][0]
    raw = row["raw_after_cell"]["outputs"][0]["text"]
    published = raw.splitlines(True)
    row["after_cell"]["outputs"][0]["text"] = published
    d.objects["after"]["cells"][0]["outputs"][0]["text"] = published
    fields = [{"output_index": 0, "path": ["text"], "raw": raw, "published": published}]
    row["projection"]["output_fields"] = fields
    if fault == "source_join": row["raw_after_cell"]["source"] += "DEV altered"
    if fault == "raw_metadata": row["raw_after_cell"]["metadata"]["unknown"] = True
    if fault == "unwitnessed_text": fields.clear()
    if fault == "wrong_text": fields[0]["raw"] += "DEV altered"
    if fault == "trailing_newline": fields[0]["published"] = [raw + "\n"]
    if fault == "nontext_path": fields[0]["path"] = ["name"]
    if fault == "duplicate_path": fields.append(copy.deepcopy(fields[0]))
    if fault == "unknown_projection": row["projection"]["ignore"] = True
    d.seal()
    with pytest.raises(ValueError): d.c._guard(d.root)


@pytest.fixture
def actual_replica(tmp_path, checkpoint):
    import shutil
    c = checkpoint
    relatives = set(c._DEPENDENCIES) | set(c._PROVENANCE) | {c._ADOPTION["contract_path"], c._TARGET}
    contract = json.loads((HERE / c._ADOPTION["contract_path"]).read_bytes())
    relatives |= {contract["bundle"][key] for key in ("archive_path", "manifest_path", "sidecar_path")}
    for relative in relatives:
        source = HERE / relative
        if not source.is_file(): source = ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return SimpleNamespace(root=tmp_path, c=c, physical=tmp_path / c._TARGET)


def test_accepted_actual_physical_slots_and_full_typed_historical_inverse(actual_replica):
    d = actual_replica
    checked = d.c._guard(d.root)
    assert checked["after_bytes"] == d.physical.read_bytes()
    assert record(checked["before_bytes"]) == d.c._BEFORE_RECORD
    assert record(checked["after_bytes"]) != d.c._BEFORE_RECORD
    indices = [row[0] for row in d.c._SIGNATURES]
    assert len(checked["after"]["cells"]) == 116 and len(indices) == 42
    assert [checked["after"]["cells"][index]["execution_count"] for index in indices] == list(range(1, 43))
    assert {index for index in indices if checked["after"]["cells"][index]["outputs"]} == {25,38,44,50,56,103,105,107,109,111,113,115}
    for index in indices:
        old, new = checked["before"]["cells"][index], checked["after"]["cells"][index]
        assert type(new["source"]) is list
        assert d.c._same({key: value for key,value in old.items() if key not in ("execution_count", "outputs")}, {key:value for key,value in new.items() if key not in ("execution_count", "outputs")})
    assert d.c.recover_practice16_checkpoint_bytes(d.root, d.c._TARGET, d.physical.read_bytes()) == checked["before_bytes"]
    assert d.physical.read_bytes() == checked["after_bytes"]


@pytest.mark.parametrize("role", ["raw_transport.jsonl", "actual_review", "projection_witness", "consumed_claim", "controller_exit", "result.json", "completion.json"])
def test_actual_evidence_byte_drift_is_rejected_without_changing_any_physical_output(actual_replica, role):
    d = actual_replica
    unchanged = d.physical.read_bytes()
    path = d.root / d.c._ACTUAL_ROLES[role]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError): d.c._guard(d.root)
    assert d.physical.read_bytes() == unchanged


def test_actual_stream_and_mime_text_witnesses_preserve_every_trailing_newline(actual_replica):
    d = actual_replica
    checked = d.c._guard(d.root)
    witness = json.loads((d.root / d.c._ACTUAL_ROLES["projection_witness"]).read_text())
    assert len(witness["cells"]) == 42
    fields = [field for cell in witness["cells"] for field in cell["changed_output_fields"]]
    assert len(fields) == 14
    assert sum(field["path"] == ["text"] for field in fields) == 12
    assert sum(field["path"] == ["data", "text/plain"] for field in fields) == 2
    for field in fields:
        assert type(field["raw"]) is str and type(field["published"]) is list
        assert field["raw"].splitlines(True) == field["published"]
        assert "".join(field["published"]) == field["raw"]
    assert record(checked["before_bytes"]) == d.c._BEFORE_RECORD


def actual_restore_arguments(d, master):
    recipe = json.loads((d.root / "infrastructure/extension/source_preservation_contracts.json").read_text())["sections"]["1.6"]
    with tarfile.open(d.root / recipe["archives"]["raw"]["path"]) as archive:
        books = {kind: archive.extractfile(pin["member"]).read().decode() for kind,pin in recipe["books"].items()}
        modules = {kind: archive.extractfile(pin["member"]).read().decode() for kind,pin in recipe["modules"].items()}
    return {"root": d.root, "identity": d.c._IDENTITY, "master_path": master, "raw_notebooks": books, "raw_modules": modules,
        "existing_notebooks": {"exercises": None, "solutions": None}}


def test_valid_copied_master_absent_destination_still_uses_unchanged_real_recipe(actual_replica):
    import shutil
    d = actual_replica
    canonical = d.root / "infrastructure/chapters/chapter1_transformer_interp/master_1_6.py"
    master = d.root / "private_empty_conversion/master_1_6.py"
    master.parent.mkdir()
    shutil.copy2(canonical, master)
    after = d.physical.read_bytes()
    result = d.c.restore_practice16_notebooks(**actual_restore_arguments(d, master))
    assert result["solutions"].encode() == after and d.physical.read_bytes() == after
    assert not (master.parent / "outputs").exists()


@pytest.mark.parametrize("fault", ["missing_canonical", "canonical_none", "stale_supplied", "nonmatching_master"])
def test_copied_master_branch_never_hides_missing_canonical_or_unvalidated_input(actual_replica, fault):
    import shutil
    d = actual_replica
    canonical = d.root / "infrastructure/chapters/chapter1_transformer_interp/master_1_6.py"
    master = d.root / "private_empty_conversion/master_1_6.py"
    master.parent.mkdir(); shutil.copy2(canonical, master)
    args = actual_restore_arguments(d, master)
    if fault == "missing_canonical": d.physical.unlink()
    if fault == "canonical_none": args["master_path"] = canonical
    if fault == "stale_supplied": args["existing_notebooks"]["solutions"] = d.c._guard(d.root)["before_bytes"].decode()
    if fault == "nonmatching_master": master.write_text("# altered copied source\n")
    with pytest.raises((ValueError, OSError)):
        d.c.restore_practice16_notebooks(**args)
