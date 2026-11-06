"""Deterministic blank projections of two public lessons; never execute lesson code.

Only missing files are created. Any different existing bytes are learner-owned and
refused, including formatting/metadata/output changes. --check never writes.
--guide prints additive first-session reading guidance without touching projections.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[2]
SCHEMA = "arena.pilot-practice-projection.v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def record(data):
    return {"sha256": digest(data), "size_bytes": len(data)}


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(data):
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def typed(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def safe(root, relative):
    p = PurePosixPath(relative)
    require(isinstance(relative, str) and not p.is_absolute() and ".." not in p.parts
            and "\\" not in relative and p.as_posix() == relative, "Unsafe path")
    target = root / relative
    require(target.resolve().is_relative_to(root.resolve()), "Path escapes root")
    for item in [target, *target.parents]:
        if item == root.parent:
            break
        require(not item.is_symlink(), "Symlink input/output refused: " + str(item))
    return target


def markdown(section, role, text, minor):
    cell = {"cell_type": "markdown", "metadata": {"pilot_practice": {"section": section, "role": role}},
            "source": text.splitlines(keepends=True)}
    if minor >= 5:
        cell["id"] = "pilot-" + section.replace(".", "-") + "-" + role
    return cell


def forecast(case, projection):
    section = case["section"]
    question = ("At which layers does a correct capital become readable, and how do other plausible "
                "capitals and random activations change that interpretation?" if section == "7.1" else
                "Can a mean-difference readout distinguish these prompt groups, and how much separation "
                "remains for random or label-shuffled directions?")
    lines = [f"# {section} — blank public Practice\n",
             f"\n**First-session question:** {question}\n\n",
             "Start with this question. Exact method/config commitment comes after "
             "implementation and checks, before your selected new experiment.\n\n",
             "### Optional informal opening forecast\n\n",
             "| Your record | Blank response |\n| --- | --- |\n",
             "| Prior exposure: seen / unknown / unexposed; where? | ______ |\n",
             "| Informal prediction, or 'I don't know yet' | ______ |\n",
             "| Confidence, 0–1 | ______ |\n| Causal reasoning | ______ |\n",
             "\nLeave method hashes pending until the relevant methods exist and pass their checks. "
             "The exact record appears beside the later selected experiment.\n\n",
             reading_guide(section), "\n\n"]
    lines.extend(["**Exposure remains possible.** Code outputs and counts are blank in this copy, but all original "
                  "source text, expected-output dropdowns, published numbers, original links and metadata remain intact. "
                  "Reading them exposes results; record that exposure. Original provenance metadata describes the "
                  "public reference, not execution of this blank copy.\n\n",
                  f"[Intact original exercises]({projection['original_exercises_link']}) · "
                  f"[Intact original solutions]({projection['original_solutions_link']}) · "
                  f"[Original instruction page]({projection['original_page_link']})\n\n",
                  "Run from this repository checkout using its existing environment. The original setup cells remain "
                  "unchanged. Only run a model/experiment when you intend to do so; opening/generating this copy "
                  "runs none. Use the unit README for commands.\n"])
    if section == "7.1":
        lines.append("\n**Interpretation pointer:** read the new qualification immediately after the original "
                     "Patchscope grid help. It also qualifies the later preserved historical necessity wording "
                     "in the summary; the original paragraphs and numbers are retained verbatim.\n")
    return "".join(lines)


def commitment(case):
    lines = ["## Before the selected new experiment: freeze the completed method and plan\n\n",
             "This is the later exact record, after learning and semantic checks. If you already ran or read "
             "this result, mark it exposed; freezing now does not undo exposure. Preserve the registered "
             "question and protocol below. For a genuinely new changed case, freeze its change before its result.\n\n",
             f"**Question:** {case['question']}\n\n",
             "First finish and test: " + ", ".join(f"`{name}`" for name in case["method_functions"]) + ".\n\n"]
    if case["section"] == "9.1":
        lines.append("`select_direction` is the next exercise below: implement it and run "
                     "`tests.test_select_direction` before returning here and running the full sweep. "
                     "Keep development selection separate from subsequent evaluation; inspected public test "
                     "results are monitored, not untouched TEST.\n\n")
    lines.extend(["| Your record | Blank response |\n| --- | --- |\n",
                  f"| Prediction ({case['prediction_type']}) | ______ |\n",
                  "| Confidence, 0–1 | ______ |\n| Causal reasoning | ______ |\n",
                  "| Prior exposure and any exploratory results already inspected | ______ |\n",
                  "| Exact completed method/input/config/selection fingerprint | ______ |\n\n",
                  "Record SHA-256 values of completed method cells and used helper/dependency source; "
                  "retain the recoverable inputs/config beside this record. Hashes of stubs do not identify "
                  "your completed method. Save before running/opening the selected result. "
                  "Unknown historical details stay unknown.\n\n"])
    for key, value in case["config_fields"].items():
        lines.append(f"- **{key}:** {value}\n")
    lines.extend(["\n| Control | Your predicted outcome and reason |\n| --- | --- |\n"])
    for control in case["controls"]:
        lines.append(f"| {control} | ______ |\n")
    lines.append("\nThis manual record has no software-enforced freeze and no claim of unseen mastery.\n")
    return "".join(lines)


SESSION_ANCHORS = {"7.1": {"stop_after": 40, "commit_after": 102},
                   "9.1": {"stop_after": 54, "commit_after": 73}}


def session_stop(section):
    if section == "7.1":
        return ("## First-session stopping point\n\nYou have reached the decoded-token table and rank "
                "plot with other-capital and norm-matched random controls. Explain the difference between "
                "decoding an answer type and an answer identity. Save your working copy and stop here. "
                "KL is optional later; TinyStories and **2️⃣ The Tuned Lens** are a separate session. "
                "Patchscopes comes after that. A readout is not a necessity test.\n")
    return ("## First-session stopping point\n\nYou have reached the rubric, mean-difference direction "
            "and layerwise AUROC with random and shuffled-direction bands. Explain any remaining control "
            "signal and why reading is not causation. Save your working copy and stop here, before "
            "**2️⃣ Ablation, Addition and Controls** and `sweep_directions`. Interventions, weight edits "
            "and geometry are later sessions.\n")


def reflection(case):
    lines = ["## Pause here: reflect on the controls\n\n",
             "Complete this manual record before moving to the next original cell. Published answers remain public "
             "reference material; separate a result you actually ran from one you only read.\n\n",
             f"**Observation:** {case['observation_prompt']}\n\n",
             "| Your record | Blank response |\n| --- | --- |\n",
             "| What I actually ran or read; exact source/config and exposure | ______ |\n",
             "| Observed primary metric/identities, with units and denominator | ______ |\n",
             "| Difference from my forecast; what surprised me? | ______ |\n",
             "| Alternative explanation and limitation | ______ |\n"]
    for control in case["controls"]:
        lines.append(f"| Observed control: {control} | ______ |\n")
    lines.extend([f"\n**Changed-case transfer:** {case['transfer']}\n\n",
                  f"| Transfer prediction ({case['transfer_prediction_type']}) | ______ |\n",
                  "| Transfer confidence, reasoning and prior exposure | ______ |\n",
                  "| Frozen change, comparison and control plan | ______ |\n\n",
                  "Leave the transfer observation blank until separately run or explicitly read as an exposed reference. "
                  "Neither this record nor a correct prediction is an independent scientific or safety evaluation.\n"])
    return "".join(lines)


QUALIFICATION = """## Interpretation qualification: intervention compatibility is not native necessity

The historical paragraph immediately above remains verbatim. The grid shows where this donor-vector intervention
is readable or useful in this target prompt. Available downstream computation and donor/target representation
compatibility both affect success. It motivates a localization hypothesis, but does not establish necessity of
blocks 1–5 for native factual recall. The original numerical values and figures are unchanged.
"""


def project(original, case, projection):
    result = deepcopy(original)
    mapped = []
    added = []
    code_state = []
    cells = []
    minor = original["nbformat_minor"]
    prefix = markdown(case["section"], "manual-forecast", forecast(case, projection), minor)
    cells.append(prefix)
    added.append({"projected_index": 0, "cell": deepcopy(prefix)})
    anchors = {item["source_index"]: item for item in projection["insert_after"]}
    require(len(anchors) == len(projection["insert_after"]), "Duplicate insertion anchor")
    session = SESSION_ANCHORS[case["section"]]
    require(set(session.values()) <= set(range(len(original["cells"]))), "Missing session anchor")
    for index, original_cell in enumerate(original["cells"]):
        cell = deepcopy(original_cell)
        require("pilot_practice" not in cell.get("metadata", {}), "Source is already a projection")
        if cell["cell_type"] == "code":
            code_state.append({"source_index": index, "execution_count": deepcopy(cell["execution_count"]),
                               "outputs": deepcopy(cell["outputs"])})
            cell["outputs"] = []
            cell["execution_count"] = None
        mapped.append({"source_index": index, "projected_index": len(cells),
                       "original_cell_sha256": digest(typed(original_cell).encode()),
                       "projected_cell_sha256": digest(typed(cell).encode())})
        cells.append(cell)
        if index in anchors:
            anchor = anchors[index]
            require(digest(typed(original_cell).encode()) == anchor["original_cell_sha256"],
                    "Insertion anchor changed")
            role = anchor["role"]
            require(role in {"controls-reflection", "grid-qualification"}, "Unknown addition")
            text = QUALIFICATION if role == "grid-qualification" else reflection(case)
            extra = markdown(case["section"], role, text, minor)
            added.append({"projected_index": len(cells), "cell": deepcopy(extra)})
            cells.append(extra)
        for key, role, text in [("stop_after", "session-stop", session_stop(case["section"])),
                                ("commit_after", "method-commitment", commitment(case))]:
            if index == session[key]:
                extra = markdown(case["section"], role, text, minor)
                added.append({"projected_index": len(cells), "cell": deepcopy(extra)})
                cells.append(extra)
    require(set(anchors) <= set(range(len(original["cells"]))), "Missing insertion anchor")
    result["cells"] = cells
    inverse = {"source_path": case["exercise_path"], "source_notebook_record": projection["source_record"],
               "original_typed_sha256": digest(typed(original).encode()), "cells": mapped,
               "added_markdown_cells": added, "original_code_state": code_state}
    require(typed(invert(result, inverse)) == typed(original), "Full typed-object inverse failed")
    return result, inverse


def invert(projected, inverse):
    """Restore the complete original typed object; no source execution or inference."""
    result = deepcopy(projected)
    for item in sorted(inverse["added_markdown_cells"], key=lambda row: row["projected_index"], reverse=True):
        require(typed(result["cells"][item["projected_index"]]) == typed(item["cell"]), "Added cell changed")
        result["cells"].pop(item["projected_index"])
    require(len(result["cells"]) == len(inverse["cells"]), "Original cell count changed")
    states = {row["source_index"]: row for row in inverse["original_code_state"]}
    for item in inverse["cells"]:
        index = item["source_index"]
        cell = result["cells"][index]
        require(digest(typed(cell).encode()) == item["projected_cell_sha256"], "Preserved source/cell object changed")
        if index in states:
            cell["execution_count"] = deepcopy(states[index]["execution_count"])
            cell["outputs"] = deepcopy(states[index]["outputs"])
        require(digest(typed(cell).encode()) == item["original_cell_sha256"], "Original cell inverse failed")
    require(digest(typed(result).encode()) == inverse["original_typed_sha256"], "Whole notebook inverse failed")
    return result


def inputs(source_root, unit):
    protocol = read_json((unit / "protocol.json").read_bytes())
    require(protocol["schema"] == SCHEMA, "Unsupported projection protocol")
    files = protocol["source_files"]
    for relative, pin in files.items():
        require(record(safe(source_root, relative).read_bytes()) == pin, "Original source changed: " + relative)
    archive = unit / "original_sources.tar.gz"
    require(archive.stat().st_size <= 64 * 1024 * 1024, "Oversized original archive file")
    require(record(archive.read_bytes()) == protocol["original_archive_record"], "Original archive changed")
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        require(len(members) == len(files) <= 4096, "Archive member set changed")
        names = set()
        total = 0
        for member in members:
            require(member.isfile() and not member.issparse() and member.name not in names, "Unsafe archive member")
            safe(source_root, member.name)
            require(member.name in files and member.size <= 16 * 1024 * 1024, "Unplanned/oversized archive member")
            names.add(member.name)
            total += member.size
            require(total <= 128 * 1024 * 1024, "Oversized original archive")
            require(record(handle.extractfile(member).read()) == files[member.name], "Archived source changed")
    registry = read_json((source_root / "mastery_hub/extension_checkpoints.json").read_bytes())
    for section, value in protocol["sections"].items():
        found = [case for case in registry["checkpoints"] if case["section"] == section]
        require(len(found) == 1 and typed(found[0]) == typed(value["case"]), "Registered question/protocol changed")
    return protocol


def planned(source_root, unit):
    protocol = inputs(source_root, unit)
    outputs = {}
    inverse = {"schema": SCHEMA, "scope": "Two additive public Practice copies; no registry/count changes", "sections": {}}
    for section, value in protocol["sections"].items():
        original_bytes = safe(source_root, value["case"]["exercise_path"]).read_bytes()
        require(record(original_bytes) == value["projection"]["source_record"], "Original notebook raced during generation")
        original = read_json(original_bytes)
        book, mapping = project(original, value["case"], value["projection"])
        outputs[value["projection"]["output_name"]] = encoded(book)
        inverse["sections"][section] = mapping
    outputs["projection_map.json"] = encoded(inverse)
    return outputs


def synchronize(source_root, unit, *, check=False):
    """Check every source and target before creating any file; never overwrite."""
    outputs = planned(source_root, unit)
    missing = []
    for name, data in outputs.items():
        path = safe(unit, name)
        if path.exists():
            require(path.is_file() and path.read_bytes() == data,
                    "Refusing different/learner-edited file: " + str(path))
        else:
            missing.append((path, data))
    if check:
        return not missing
    require(unit.is_dir(), "Unit directory must already exist")
    for path, data in missing:
        # Exclusive creation also refuses a target that races after full preflight.
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    return True


def reading_guide(section):
    """Read the common opening and selected route from the unit's human guide."""
    text = (UNIT / "README.md").read_text()
    excerpts = []
    for heading in ["## Start here:", f"## {section}: first session"]:
        start = text.index(heading)
        end = text.find("\n## ", start + 1)
        excerpts.append(text[start:] if end == -1 else text[start:end])
    return "\n\n".join(excerpts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Read-only equality check; missing or edited copies fail")
    mode.add_argument("--guide", choices=["7.1", "9.1"], help="Print the first-session route; no projection writes or lesson execution")
    parser.add_argument("--source-root", type=Path, default=ROOT, help="Read-only source checkout; exact pins must match")
    args = parser.parse_args()
    if args.guide:
        print(reading_guide(args.guide))
        return 0
    try:
        ok = synchronize(args.source_root.resolve(), UNIT, check=args.check)
    except (ValueError, FileExistsError) as error:
        print(str(error))
        return 1
    print("Practice projections match" if ok else "Practice projections missing; no files written")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
