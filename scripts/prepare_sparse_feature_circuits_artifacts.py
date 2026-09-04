"""Prepare official Sparse Feature Circuits artifacts outside the repo history.

This script does not claim replication. It pins and prepares the external
source/data/SAE files needed before the [8.5] GT-2 replication can be run.
Large files are placed under `external/feature-circuits/`, which is gitignored.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url, list_repo_files

ROOT = Path(__file__).resolve().parents[1]

OFFICIAL_REPO_URL = "https://github.com/saprmarks/feature-circuits.git"
OFFICIAL_REPO_COMMIT = "7fbd82b895ae16294f4e6fc7bfc675d1d680d659"
HF_REPO_ID = "saprmarks/pythia-70m-deduped-saes"
HF_COMMIT = "50a434461d36ed78d1b0b901944e6edc829f1dce"
HF_ZIP = "dictionaries_pythia-70m-deduped_10.zip"
HF_ZIP_SIZE_BYTES = 2_369_891_306
HF_ZIP_ETAG = "e8e320f0d068b2edf6b43ae2b641c96aaf0b3e5631cf173576325d4ad75c7979"

REQUIRED_REPO_FILES = {
    "README.md",
    "attribution.py",
    "activation_utils.py",
    "ablation.py",
    "circuit.py",
    "circuit_plotting.py",
    "dictionary_loading_utils.py",
    "loading_utils.py",
    "data/simple_train.json",
    "data/simple_test.json",
    "data/rc_train.json",
    "data/rc_test.json",
    "annotations/pythia-70m-deduped.jsonl",
    "scripts/get_circuit.sh",
    "scripts/evaluate_circuit.sh",
}


def expected_dictionary_paths(num_layers: int = 6) -> tuple[str, ...]:
    paths = ["embed/10_32768/ae.pt"]
    for layer in range(num_layers):
        paths += [
            f"attn_out_layer{layer}/10_32768/ae.pt",
            f"mlp_out_layer{layer}/10_32768/ae.pt",
            f"resid_out_layer{layer}/10_32768/ae.pt",
        ]
    return tuple(paths)


def run_git(args: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def clone_or_update_repo(source_dir: Path, *, download: bool) -> dict[str, Any]:
    if not source_dir.exists():
        if not download:
            return {
                "present": False,
                "head": None,
                "pinned": False,
                "missing_files": sorted(REQUIRED_REPO_FILES),
            }
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        run_git(["clone", "--depth", "1", OFFICIAL_REPO_URL, str(source_dir)])

    head = run_git(["rev-parse", "HEAD"], cwd=source_dir)
    present_files = {
        str(path.relative_to(source_dir))
        for path in source_dir.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }
    return {
        "present": True,
        "head": head,
        "pinned": head == OFFICIAL_REPO_COMMIT,
        "missing_files": sorted(REQUIRED_REPO_FILES - present_files),
    }


def hf_manifest() -> dict[str, Any]:
    files = set(list_repo_files(HF_REPO_ID, repo_type="model", revision=HF_COMMIT))
    metadata = get_hf_file_metadata(hf_hub_url(HF_REPO_ID, HF_ZIP, revision=HF_COMMIT))
    return {
        "files": sorted(files),
        "zip_present": HF_ZIP in files,
        "commit_hash": metadata.commit_hash,
        "etag": metadata.etag,
        "size": metadata.size,
        "metadata_matches_pin": (
            metadata.commit_hash == HF_COMMIT
            and metadata.etag == HF_ZIP_ETAG
            and metadata.size == HF_ZIP_SIZE_BYTES
        ),
    }


def download_zip(download_dir: Path, *, download: bool) -> Path | None:
    path = download_dir / HF_ZIP
    if path.exists():
        return path
    if not download:
        return None
    download_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_ZIP,
        repo_type="model",
        revision=HF_COMMIT,
        local_dir=download_dir,
    )
    return Path(downloaded)


def extract_zip(zip_path: Path | None, extract_dir: Path, *, extract: bool) -> dict[str, Any]:
    if zip_path is None:
        return {"zip_available": False, "extracted": False, "zip_members": 0}
    if not extract:
        return {"zip_available": True, "extracted": False, "zip_members": None}
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        archive.extractall(extract_dir)
    return {"zip_available": True, "extracted": True, "zip_members": len(members)}


def validate_dictionaries(dictionary_dir: Path) -> dict[str, Any]:
    expected = expected_dictionary_paths()
    present = {
        str(path.relative_to(dictionary_dir))
        for path in dictionary_dir.rglob("ae.pt")
        if path.is_file()
    } if dictionary_dir.exists() else set()
    missing = [path for path in expected if path not in present]
    return {
        "dictionary_dir": str(dictionary_dir),
        "expected_dictionary_count": len(expected),
        "present_dictionary_count": len(expected) - len(missing),
        "missing_dictionary_paths": missing,
        "ready": not missing,
    }


def link_official_repo_dictionaries(source_dir: Path, dictionary_dir: Path) -> dict[str, Any]:
    """Expose downloaded dictionaries at the path expected by the official repo."""

    link_path = source_dir / "dictionaries"
    dictionary_root = dictionary_dir.parent
    if not source_dir.exists():
        return {
            "link_path": str(link_path),
            "target": str(dictionary_root),
            "present": False,
            "ready": False,
            "reason": "official repo is not present",
        }
    if not dictionary_root.exists():
        return {
            "link_path": str(link_path),
            "target": str(dictionary_root),
            "present": False,
            "ready": False,
            "reason": "dictionary root is not present",
        }

    if link_path.is_symlink():
        ready = link_path.resolve() == dictionary_root.resolve()
        return {
            "link_path": str(link_path),
            "target": str(dictionary_root),
            "present": True,
            "ready": ready,
            "reason": "existing symlink" if ready else "symlink points elsewhere",
        }
    if link_path.exists():
        nested_ready = validate_dictionaries(link_path / "pythia-70m-deduped")["ready"]
        return {
            "link_path": str(link_path),
            "target": str(dictionary_root),
            "present": True,
            "ready": nested_ready,
            "reason": "existing non-symlink dictionary directory",
        }

    link_path.symlink_to(Path("..") / "dictionaries", target_is_directory=True)
    return {
        "link_path": str(link_path),
        "target": str(dictionary_root),
        "present": True,
        "ready": True,
        "reason": "created symlink",
    }


def patch_official_repo_compatibility(source_dir: Path) -> dict[str, Any]:
    """Patch pinned official scripts for current PyTorch without changing algorithms."""

    ablation_path = source_dir / "ablation.py"
    if not ablation_path.exists():
        return {
            "ablation_path": str(ablation_path),
            "present": False,
            "patched": False,
            "ready": False,
            "reason": "ablation.py is not present",
        }

    text = ablation_path.read_text()
    original = text
    replacements = {
        "from loading_utils import load_examples": "from data_loading_utils import load_examples",
        'circuit = t.load(args.circuit)["nodes"]': (
            'circuit = t.load(args.circuit, weights_only=False)["nodes"]'
        ),
    }
    applied: list[str] = []
    for old, new in replacements.items():
        if old in text:
            text = text.replace(old, new)
            applied.append(old)
    if text != original:
        ablation_path.write_text(text)

    ready_text = ablation_path.read_text()
    ready = all(new in ready_text for new in replacements.values())
    return {
        "ablation_path": str(ablation_path),
        "present": True,
        "patched": text != original,
        "ready": ready,
        "applied_replacements": applied,
        "reason": "patched or already compatible" if ready else "compatibility replacements missing",
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=ROOT / "external" / "feature-circuits",
        help="Ignored local cache root for source, downloads, dictionaries, and report.",
    )
    parser.add_argument("--download", action="store_true", help="Clone source and download zip.")
    parser.add_argument("--extract", action="store_true", help="Extract the downloaded zip.")
    parser.add_argument("--clean", action="store_true", help="Delete artifact root before running.")
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit nonzero unless all local dictionaries are present.",
    )
    args = parser.parse_args()

    artifact_root = args.artifact_root.resolve()
    if args.clean and artifact_root.exists():
        shutil.rmtree(artifact_root)

    source_dir = artifact_root / "feature-circuits"
    download_dir = artifact_root / "downloads"
    dictionary_dir = artifact_root / "dictionaries" / "pythia-70m-deduped"

    source = clone_or_update_repo(source_dir, download=args.download)
    hf = hf_manifest()
    zip_path = download_zip(download_dir, download=args.download)
    extraction = extract_zip(zip_path, artifact_root, extract=args.extract)
    dictionaries = validate_dictionaries(dictionary_dir)
    dictionary_link = link_official_repo_dictionaries(source_dir, dictionary_dir)
    official_repo_compatibility = patch_official_repo_compatibility(source_dir)
    report = {
        "artifact_root": str(artifact_root),
        "official_repo": {
            "url": OFFICIAL_REPO_URL,
            "expected_commit": OFFICIAL_REPO_COMMIT,
            **source,
        },
        "huggingface": {
            "repo_id": HF_REPO_ID,
            "expected_commit": HF_COMMIT,
            "expected_zip": HF_ZIP,
            "expected_size_bytes": HF_ZIP_SIZE_BYTES,
            "expected_etag": HF_ZIP_ETAG,
            "zip_path": str(zip_path) if zip_path is not None else None,
            **hf,
        },
        "extraction": extraction,
        "dictionaries": dictionaries,
        "official_repo_dictionary_link": dictionary_link,
        "official_repo_compatibility": official_repo_compatibility,
        "ready_for_gt2_replication": (
            source["pinned"]
            and not source["missing_files"]
            and hf["metadata_matches_pin"]
            and dictionaries["ready"]
            and dictionary_link["ready"]
            and official_repo_compatibility["ready"]
        ),
    }
    write_json(artifact_root / "artifact_readiness_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_ready and not report["ready_for_gt2_replication"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
