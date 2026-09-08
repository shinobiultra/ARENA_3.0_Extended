"""Stage exact 16.18/16.19 evidence for review; install only a byte-approved candidate.

This module is outside the native experiment's frozen imports. It never loads a
model, fits an estimator, imports native study code, or rewrites learner content.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import evidence_bundles as bundles

CHAPTER = "chapter16_shapley_attribution_baselines"
SECTIONS = {
    "16.18": ("part18_spex", "16.18_Find_the_Interactions_Without_Trying_Every_Subset",
              "18_[16.18]_Find_the_Interactions_Without_Trying_Every_Subset.md", "1618_"),
    "16.19": ("part19_proxyspex", "16.19_Explain_the_Proxy_Check_the_Model",
              "19_[16.19]_Explain_the_Proxy_Check_the_Model.md", "1619_"),
}
FIXTURE = f"{CHAPTER}/exercises/part18_spex/expected_outputs/native_query_fixture.json.gz"
SECTION_SCHEMA = "arena.sparse-interaction.section-publication.v1"
CANDIDATE_SCHEMA = "arena.sparse-interaction.publication-candidate.v1"
APPROVAL_STATUS = "APPROVE_SPARSE_INTERACTION_PUBLICATION"
REQUIRED_ROLES = {"exercises", "solutions", "page", "figure", "verification_report", "data_instructions",
                  "native_summary", "artifacts_lock", "verification_schema"}
GENERATED_RECEIPTS = {"learner_execution": "learner_execution.json", "native_execution": "native_execution.json",
                      "fresh_execution": "fresh_native_execution.json", "reproduction": "fresh_reproduction.json",
                      "comparison_summary": "native_comparison_summary.json"}
ARCHIVE_ROLES = {"review_archive": "execution_review.tar.gz", "review_archive_contract": "execution_review.tar.gz.review-bundle.json"}
MAX_RECEIPT_BYTES = 64 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Missing or linked JSON evidence: {path}")
    require(path.stat().st_size <= MAX_RECEIPT_BYTES, f"JSON evidence exceeds bound: {path}")
    return json.loads(path.read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def safe_relative(value):
    require(isinstance(value, str) and value, "Missing relative artifact path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and "\\" not in value and ":" not in value
            and all(part not in {"", ".", ".."} for part in value.split("/"))
            and not any(ord(char) < 32 for char in value), f"Unsafe artifact path: {value}")
    return Path(*path.parts)


def physical_record(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Missing or linked artifact: {path}")
    return {"absolute_path": str(path.resolve()), **bundles.file_record(path)}


def verify_record(record):
    require(isinstance(record, dict) and Path(record.get("absolute_path", "")).is_absolute(), "Missing absolute evidence locator")
    actual = physical_record(record["absolute_path"])
    require(all(actual[key] == record.get(key) for key in actual), f"Evidence bytes changed: {actual['absolute_path']}")
    return Path(actual["absolute_path"])


def _decode_metadata(value):
    """Decode only the study's structured JSON; NPZ arrays stay untouched."""
    if isinstance(value, list):
        return [_decode_metadata(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"__mapping__"}:
            return {_decode_metadata(key): _decode_metadata(item) for key, item in value["__mapping__"]}
        if set(value) == {"__tuple__"}:
            return tuple(_decode_metadata(item) for item in value["__tuple__"])
        require(set(value) == {"__array__"}, "Unknown native structured encoding")
    return value


def _study_record(path, expected=None):
    hashes = read_json(path / "hashes.json")
    require(set(hashes) == {"arrays.npz", "record.json"}, "Native record inventory changed")
    require(expected is None or hashes == expected, "Native summary link changed")
    for name, checksum in hashes.items():
        require(bundles.sha256(path / name) == checksum, f"Native record changed: {path / name}")
    return _decode_metadata(read_json(path / "record.json"))


def validate_native_run(directory):
    directory = Path(directory).resolve()
    receipt = read_json(directory / "execution.json")
    require(receipt.get("status") == "COMPLETE" and receipt.get("returncode") == 0,
            "Native execution must have completed successfully")
    require(receipt.get("execution_kind") == "cold_source_native"
            and receipt.get("child_received_learner_payloads") is False, "A cold source-only native run is required")
    require(not any(receipt.get(key) for key in ("failures", "resource_violations", "changed_sources")), "Native execution has failures or drift")
    require(receipt.get("source_inputs") and receipt["source_inputs"] == receipt.get("source_inputs_after"), "Native source inventory changed")
    for name, record in receipt["source_inputs"].items():
        require(bundles.file_record(Path(name)) == record, f"Frozen input changed: {name}")
    study = directory / "study"
    summary = _study_record(study / "summary")
    require(summary.get("status") == "COMPLETE", "Native scientific summary is incomplete")
    boundary = _study_record(study / "prediction_boundary", summary["prediction_boundary_hashes"])
    require(boundary["condition_records"] == summary["conditions"] and boundary["references"] == summary["references"],
            "Predictions changed across the recorded audit boundary")
    for entry in [*summary["conditions"], *summary["audit_records"]]:
        _study_record(study / safe_relative(entry["path"]), entry["hashes"])
    _study_record(study / "oracle_values", summary["oracle_value_hashes"])
    database = summary["oracle_inventory"]["database"]
    require(Path(database["path"]).resolve() == (study / "oracle/oracle.sqlite3").resolve(), "Foreign oracle database")
    require(bundles.file_record(Path(database["path"])) == {key: database[key] for key in ("sha256", "size_bytes")},
            "Native oracle database changed")
    return receipt, summary


def validate_native_pair(native_run, fresh_run):
    native_run, fresh_run = Path(native_run).resolve(), Path(fresh_run).resolve()
    require(native_run != fresh_run, "Reproduction requires two distinct cold runs")
    original, summary = validate_native_run(native_run)
    fresh, fresh_summary = validate_native_run(fresh_run)
    reproduction = read_json(fresh_run / "reproduction.json")
    require(reproduction.get("status") == "PASS" and not reproduction.get("failures"), "Cold reproduction must pass")
    require(reproduction.get("native_child_received_learner_payloads") is False
            and reproduction.get("cached_learner_models_or_values_used_for_native_execution") is False,
            "Reproduction used cached learner payloads")
    require(reproduction["learner_directory"] == str(native_run) and reproduction["fresh_directory"] == str(fresh_run),
            "Reproduction refers to different runs")
    for folder, key in ((native_run, "learner_execution_sha256"), (fresh_run, "fresh_execution_sha256")):
        require(bundles.sha256(folder / "execution.json") == reproduction[key], "Reproduced execution receipt changed")
    require(original["source_inputs"] == fresh["source_inputs"] and original["method_modules"] == fresh["method_modules"],
            "Cold runs used different sources or methods")
    process = reproduction["comparison_execution"]
    process_path = Path(process["path"])
    require(process_path.resolve() == (fresh_run / "comparison_process/execution.json").resolve(), "Foreign comparison receipt")
    require(process["sha256"] == bundles.sha256(process_path), "Comparison process receipt changed")
    actual_process = read_json(process_path)
    require(actual_process["status"] == "COMPLETE" and actual_process["returncode"] == 0
            and not actual_process["resource_violations"], "Comparison process did not complete")
    details = fresh_run / "comparison_process/comparison.json"
    comparison = read_json(fresh_run / "comparison_process/comparison_summary.json")
    require(bundles.sha256(details) == comparison["comparison_sha256"] == reproduction["comparison_sha256"], "Comparison detail bytes changed")
    require(comparison["status"] == "PASS" and comparison["mismatch_count"] == 0, "Scientific reproduction mismatch")
    require(summary["observed_inventory"] == fresh_summary["observed_inventory"], "Native measured inventory changed")
    return {"native_execution": physical_record(native_run / "execution.json"),
            "fresh_execution": physical_record(fresh_run / "execution.json"),
            "reproduction": physical_record(fresh_run / "reproduction.json"),
            "comparison_summary": physical_record(fresh_run / "comparison_process/comparison_summary.json"),
            "observed_inventory": summary["observed_inventory"],
            "reference_outcomes": [{key: row.get(key) for key in ("stage_id", "status", "process_status", "scientific_status")}
                                   for row in summary["references"]],
            "bitwise_exact": comparison["bitwise_exact"], "numerical_difference_leaves": comparison["numerical_difference_leaves"]}


def artifact_target(section, role, name=None):
    directory, stem, page, prefix = SECTIONS[section]
    base = f"{CHAPTER}/exercises/{directory}"
    if role in {"exercises", "solutions"}:
        return f"{base}/{stem}_{role}.ipynb"
    if role == "page":
        return f"{CHAPTER}/instructions/pages/{page}"
    if role == "verification_report":
        return f"{base}/verification_report.json"
    if role == "verification_schema":
        return f"{base}/verification_report.schema.json"
    if role == "artifacts_lock":
        return f"{base}/artifacts.lock.yml"
    if role == "native_summary":
        return f"{base}/expected_outputs/native_summary.json"
    if role == "learner_data":
        require(section == "16.18", "The portable fixture is shared from 16.18")
        return FIXTURE
    if role == "data_contract":
        require(section == "16.18", "The portable data contract belongs to 16.18")
        return FIXTURE + ".data-contract.json"
    if role in GENERATED_RECEIPTS:
        return f"{base}/expected_outputs/{GENERATED_RECEIPTS[role]}"
    if role in ARCHIVE_ROLES:
        require(section == "16.18", "One shared review archive belongs to 16.18")
        return f"{base}/expected_outputs/{ARCHIVE_ROLES[role]}"
    require(isinstance(name, str) and safe_relative(name).name == name, "Artifact requires a simple filename")
    if role == "figure":
        require(name.startswith(prefix) and name.endswith(".png"), "Figure lacks its section namespace")
        return f"{CHAPTER}/instructions/assets/{name}"
    require(role in {"data_instructions", "supporting_evidence"}, f"Unknown publication role: {role}")
    return f"{base}/expected_outputs/{name}"


def _section_inputs(path, native):
    section = read_json(path)
    require(section.get("schema") == SECTION_SCHEMA and section.get("section") in SECTIONS, "Unsupported section publication manifest")
    require(section.get("native_execution_sha256") == native["native_execution"]["sha256"]
            and section.get("fresh_reproduction_sha256") == native["reproduction"]["sha256"], "Lesson belongs to another native run")
    execution_path = verify_record(section["learner_execution"])
    execution = read_json(execution_path)
    require(execution.get("status") in {"passed", "FULL_LEARNER_PASS"} and execution.get("sources_unchanged") is True,
            "Complete own-answer learner execution is required")
    require(execution.get("executed_code_cells", 0) > 0
            and execution["executed_code_cells"] == execution.get("total_code_cells"), "Learner code cells are incomplete")
    for key in ("native_execution_sha256", "fresh_reproduction_sha256"):
        require(execution.get(key) == section[key], "Learner execution uses different native evidence")
    require(bundles.sha256(Path(execution["executed_notebook"])) == execution["notebook_sha256"], "Executed own-answer notebook changed")
    replay = execution.get("portable_replay", {})
    require(replay.get("status") == "PASS" and replay.get("no_full_study_cache_required") is True,
            "Portable student replay must pass without the full external study")
    require(section.get("data_access", {}).get("fixture_target") == FIXTURE
            and section["data_access"].get("fixture_sha256") == replay.get("fixture_sha256"), "Portable fixture binding changed")
    files, roles = {}, set()
    for item in section["artifacts"]:
        require(item["role"] not in {*GENERATED_RECEIPTS, *ARCHIVE_ROLES, "data_contract"}, "Publisher-generated role supplied as lesson content")
        source = verify_record(item)
        target = artifact_target(section["section"], item["role"], item.get("name"))
        require(target not in files, "Duplicate canonical publication target")
        require(source.stat().st_size <= bundles.MAX_MEMBER_BYTES, "Publication artifact exceeds the review member bound")
        bundles.inspect_member(target, source.read_bytes())
        files[target] = {**item, "absolute_path": str(source), "section": section["section"]}
        roles.add(item["role"])
    require(REQUIRED_ROLES <= roles, "Missing canonical learner, visible result, report or data instructions")
    require(files[artifact_target(section["section"], "solutions")]["sha256"] == execution["notebook_sha256"],
            "Published solutions must be the actual completed own-answer notebook")
    recorded_figures = execution.get("figures", {})
    require(isinstance(recorded_figures, dict) and recorded_figures, "Learner execution must bind its actual figures")
    require(Counter(recorded_figures.values()) == Counter(item["sha256"] for item in files.values() if item["role"] == "figure"),
            "Published figure bytes differ from the complete actual learner figure inventory")
    access = section["data_access"]
    instructions = access.get("regeneration_instructions_target")
    require(instructions in files and files[instructions]["role"] == "data_instructions",
            "Portable data access needs explicit packaged regeneration instructions")
    require(access.get("default_requires_full_study") is False
            and access.get("full_study_access") == "verified_external_cache_or_source_regeneration",
            "Do not present external full-study arrays as a portable default input")
    return section, files, physical_record(execution["executed_notebook"])


def _full_inventory(native_run, fresh_run):
    result = {}
    for label, directory in (("native", Path(native_run)), ("fresh", Path(fresh_run))):
        for path in sorted(directory.rglob("*")):
            require(not path.is_symlink(), "Native retained output contains a symlink")
            if path.is_file():
                result[f"{label}/{path.relative_to(directory).as_posix()}"] = physical_record(path)
    return result


def seal_fixture(fixture, native_execution):
    """Called after export, before a portable learner replay needs its sidecar."""
    execution = read_json(native_execution)
    require(execution.get("status") == "COMPLETE", "Cannot seal a fixture from incomplete native execution")
    return bundles.seal_json_data(Path(fixture), provenance={
        "kind": "native_execution_packaging", "source_execution": physical_record(native_execution),
        "measurement_claim": "Exact exported native-query subset; no new model measurement or complete-study substitute.",
    })


def _make_archive(path, files):
    require(len(files) <= bundles.MAX_MEMBERS, "Too many review bundle members")
    require(sum(Path(source).stat().st_size for source in files.values()) <= bundles.MAX_TOTAL_BYTES, "Review bundle members exceed total bound")
    with Path(path).open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name, source in sorted(files.items()):
                require(Path(source).stat().st_size <= bundles.MAX_MEMBER_BYTES, "Review evidence exceeds member bound: " + name)
                data = Path(source).read_bytes()
                bundles.inspect_member(name, data)
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                archive.addfile(info, io.BytesIO(data))


def stage_candidate(native_run, fresh_run, section_manifests, output):
    output = Path(output).resolve()
    require(not output.exists(), "Staging requires a new directory; previous evidence is preserved")
    require(not output.is_relative_to(ROOT), "Stage outside the repository; canonical files are installed separately")
    native = validate_native_pair(native_run, fresh_run)
    sections, artifacts = {}, {}
    for path in section_manifests:
        section, files, executed_notebook = _section_inputs(path, native)
        require(section["section"] not in sections, "Duplicate section manifest")
        require(not (set(files) & set(artifacts)), "Canonical target collision")
        sections[section["section"]] = {"input_manifest": physical_record(path), "data_access": section["data_access"],
                                        "learner_execution": section["learner_execution"], "executed_notebook": executed_notebook}
        artifacts.update(files)
    require(set(sections) == set(SECTIONS) and FIXTURE in artifacts, "Both sections and their shared portable fixture are required")
    fixture_hash = artifacts[FIXTURE]["sha256"]
    require(all(row["data_access"]["fixture_sha256"] == fixture_hash for row in sections.values()), "Sections use different portable fixtures")
    external = _full_inventory(native_run, fresh_run)
    for section, row in sections.items():
        for key in ("input_manifest", "learner_execution", "executed_notebook"):
            external[f"learner_{section}/{key}" + Path(row[key]["absolute_path"]).suffix] = row[key]
    output.mkdir(parents=True)
    record = {"schema": CANDIDATE_SCHEMA, "status": "STAGED_PENDING_INDEPENDENT_REVIEW",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "native": native, "sections": sections,
              "publisher": physical_record(__file__), "artifacts": {}, "external_payloads": external}
    archive_files = {}
    for target, item in artifacts.items():
        destination = output / "tree" / safe_relative(target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(verify_record(item), destination)
        record["artifacts"][target] = {"role": item["role"], "section": item["section"], **bundles.file_record(destination)}
        archive_files["tree/" + target] = destination
    fixture = output / "tree" / FIXTURE
    seal_fixture(fixture, native["native_execution"]["absolute_path"])
    sidecar = Path(str(fixture) + ".data-contract.json")
    sidecar_target = FIXTURE + ".data-contract.json"
    record["artifacts"][sidecar_target] = {"role": "data_contract", "section": "16.18", **bundles.file_record(sidecar)}
    archive_files["tree/" + sidecar_target] = sidecar
    for section, row in sections.items():
        for role in GENERATED_RECEIPTS:
            target = artifact_target(section, role)
            require(target not in record["artifacts"], "Supporting evidence conflicts with a generated receipt")
            destination = output / "tree" / target
            source = row[role] if role == "learner_execution" else native[role]
            shutil.copyfile(verify_record(source), destination)
            record["artifacts"][target] = {"role": role, "section": section, **bundles.file_record(destination)}
            archive_files["tree/" + target] = destination
    for key in ("native_execution", "fresh_execution", "reproduction", "comparison_summary"):
        destination = output / "evidence" / (key + ".json")
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(verify_record(native[key]), destination)
        archive_files["evidence/" + destination.name] = destination
    inventory = output / "evidence/full_native_artifacts.json"
    write_json(inventory, {"schema": "arena.external-native-inventory.v1", "files": external,
                           "boundary": "Full retained runs remain external. The packaged query fixture supports the documented student replay."})
    archive_files["evidence/full_native_artifacts.json"] = inventory
    archive = output / "execution_review.tar.gz"
    _make_archive(archive, archive_files)
    bundles.seal_review_bundle(archive, provenance={
        "kind": "native_execution_packaging", "source_execution": native["native_execution"],
        "measurement_claim": "Packaging completed native execution and reproduction; failed reference outcomes remain source outcomes. Publication is pending review.",
    }, external_payloads=external)
    record["review_archive"] = physical_record(archive)
    record["review_archive_contract"] = physical_record(bundles.contract_path(archive))
    # An archive cannot include itself. These two installation artifacts are
    # independently hash-bound by the candidate and its review approval.
    for role in ARCHIVE_ROLES:
        source = verify_record(record[role])
        target = artifact_target("16.18", role)
        destination = output / "tree" / target
        shutil.copyfile(source, destination)
        record["artifacts"][target] = {"role": role, "section": "16.18", **bundles.file_record(destination)}
    write_json(output / "candidate_manifest.json", record)
    return record


def validate_candidate(directory, *, verify_external=True):
    directory = Path(directory).resolve()
    candidate = read_json(directory / "candidate_manifest.json")
    require(candidate.get("schema") == CANDIDATE_SCHEMA
            and candidate.get("status") == "STAGED_PENDING_INDEPENDENT_REVIEW", "Unsupported publication candidate")
    verify_record(candidate["publisher"])
    for target, item in candidate["artifacts"].items():
        require(item.get("section") in SECTIONS, "Unknown candidate section")
        expected = artifact_target(item["section"], item["role"], Path(target).name)
        require(target == expected, "Unexpected canonical target: " + target)
        source = directory / "tree" / safe_relative(target)
        require(not source.is_symlink() and source.resolve().is_relative_to(directory / "tree"), "Staged file escaped its candidate tree")
        actual = bundles.file_record(source)
        require(all(actual[key] == item[key] for key in actual), "Staged canonical artifact changed: " + target)
    archive = verify_record(candidate["review_archive"])
    require(archive.parent == directory, "Foreign candidate review archive")
    verify_record(candidate["review_archive_contract"])
    contract = bundles.validate_review_bundle(archive)
    require(contract["external_payloads"] == candidate["external_payloads"], "Candidate external inventory differs from sealed archive")
    for target, item in candidate["artifacts"].items():
        if item["role"] in ARCHIVE_ROLES:
            require(all(candidate[item["role"]][key] == item[key] for key in ("sha256", "size_bytes")), "Installed archive binding differs")
            continue
        member = contract["members"].get("tree/" + target, {})
        require(all(member.get(key) == item[key] for key in ("sha256", "size_bytes")), "Candidate and archive artifact differ")
    if verify_external:
        for item in candidate["external_payloads"].values():
            verify_record(item)
        execution = read_json(verify_record(candidate["native"]["native_execution"]))
        for name, record in execution["source_inputs"].items():
            require(bundles.file_record(Path(name)) == record, "Frozen native input changed since staging: " + name)
    return candidate


def validate_approval(directory, approval_path):
    directory = Path(directory).resolve()
    candidate = validate_candidate(directory)
    approval = read_json(approval_path)
    require(approval.get("status") == APPROVAL_STATUS and approval.get("independent_review") is True,
            "Independent scientific and learner publication approval is required")
    require(approval.get("candidate_manifest_sha256") == bundles.sha256(directory / "candidate_manifest.json"),
            "Approval refers to different candidate bytes")
    require(approval.get("sections") == ["16.18", "16.19"], "Approval must cover both sections")
    checks = approval.get("checks", {})
    require(all(checks.get(key) is True for key in ("scientific_outcomes", "visible_figures", "controls_and_failures",
                                                   "portable_student_replay", "source_and_reproduction")), "Publication review is incomplete")
    require(approval.get("review_evidence"), "Approval needs physical review evidence")
    for item in approval["review_evidence"]:
        verify_record(item)
    return candidate


def install_candidate(directory, approval_path, *, repository=ROOT):
    """Install exact reviewed bytes, retaining prior target bytes before replacement."""
    directory, repository = Path(directory).resolve(), Path(repository).resolve()
    candidate = validate_approval(directory, approval_path)
    destination_receipt = directory / "installation.json"
    require(not destination_receipt.exists(), "Candidate installation already recorded; do not overwrite its history")
    paths = {}
    for target, item in candidate["artifacts"].items():
        destination = repository / safe_relative(target)
        require(destination.resolve().is_relative_to(repository) and not destination.is_symlink(), "Canonical destination escapes repository")
        require(not any(parent.is_symlink() for parent in destination.parents if parent.is_relative_to(repository)),
                "Canonical destination has a linked parent directory")
        paths[target] = destination
    record = {"status": "INSTALLING", "candidate_manifest_sha256": bundles.sha256(directory / "candidate_manifest.json"),
              "approval": physical_record(approval_path), "previous_files": {}, "installed_files": {}}
    write_json(destination_receipt, record)
    for target, destination in paths.items():
        if destination.exists():
            backup = directory / "previous_canonical" / target
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(destination, backup)
            record["previous_files"][target] = physical_record(backup)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(directory / "tree" / target, destination)
        record["installed_files"][target] = bundles.file_record(destination)
        require(all(record["installed_files"][target][key] == candidate["artifacts"][target][key]
                    for key in ("sha256", "size_bytes")), "Installed bytes differ from approved candidate")
        write_json(destination_receipt, record)
    record["status"] = "INSTALLED_APPROVED_CANDIDATE"
    write_json(destination_receipt, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("stage")
    for name in ("native-run", "fresh-run", "output"):
        stage.add_argument("--" + name, type=Path, required=True)
    stage.add_argument("--section-manifest", type=Path, action="append", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("candidate", type=Path)
    install = sub.add_parser("install")
    install.add_argument("candidate", type=Path)
    install.add_argument("--approval", type=Path, required=True)
    seal = sub.add_parser("seal-fixture")
    seal.add_argument("fixture", type=Path)
    seal.add_argument("--native-execution", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "stage":
        record = stage_candidate(args.native_run, args.fresh_run, args.section_manifest, args.output)
    elif args.command == "validate":
        record = validate_candidate(args.candidate)
    elif args.command == "install":
        record = install_candidate(args.candidate, args.approval)
    else:
        record = seal_fixture(args.fixture, args.native_execution)
    print(json.dumps({"status": record.get("status", "SEALED_FIXTURE")}, indent=2))


if __name__ == "__main__":
    main()
