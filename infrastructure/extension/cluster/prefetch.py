#!/usr/bin/env python
"""Download the Hugging Face models and datasets that sections need into HF_HOME (run on a login node).

    HF_HOME=~/projects/arena-extended/hf_cache .venv/bin/python infrastructure/extension/cluster/prefetch.py 8.4 9.1

Scans each section's master file, utils.py and tests.py for `from_pretrained("...")`, `hf_hub_download(repo_id=...)`
and `load_dataset("...")` calls with literal names. TransformerLens names (e.g. "gpt2-small") are mapped to their
official HF repos. Extra repos can be given with --extra.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG = yaml.safe_load((ROOT / "infrastructure" / "core" / "config.yaml").read_text())
MODEL = re.compile(r"""(?:from_pretrained(?:_no_processing)?|hf_hub_download)\(\s*(?:repo_id\s*=\s*)?["']([\w./-]+)["']""")
DATASET = re.compile(r"""load_dataset\(\s*["']([\w./-]+)["']""")
SKIP_PATTERNS = ["*.msgpack", "*.h5", "*.ot", "*.onnx", "*.tflite", "flax_model*", "tf_model*", "rust_model*", "onnx/*", "openvino/*"]


def section_files(number: str) -> list[Path]:
    mapping = CONFIG["conversion_map"][number]
    chapter = CONFIG["chapter_names"][int(number.split(".")[0])]
    files = list((ROOT / "infrastructure" / "chapters").rglob(f"master_{number.replace('.', '_')}.py"))
    section_dir = ROOT / chapter / "exercises" / mapping["exercise_dir"]
    files += [section_dir / name for name in ("utils.py", "tests.py") if (section_dir / name).exists()]
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sections", nargs="+")
    parser.add_argument("--extra", nargs="*", default=[], help="extra model repo ids")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from transformer_lens.loading_from_pretrained import get_official_model_name

    models, datasets = set(args.extra), set()
    for number in args.sections:
        for file in section_files(number):
            text = file.read_text()
            models.update(MODEL.findall(text))
            datasets.update(DATASET.findall(text))

    for name in sorted(models):
        repo = name
        if "/" not in name:
            try:
                repo = get_official_model_name(name)
            except ValueError:
                pass
        if "/" not in repo and repo not in ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "distilgpt2"):
            print(f"skip {name!r} (not a hub repo id)")
            continue
        try:
            snapshot_download(repo, ignore_patterns=SKIP_PATTERNS)
            print(f"model   {repo}")
        except Exception as e:  # noqa: BLE001
            print(f"FAILED  {repo}: {e}")
    for name in sorted(datasets):
        try:
            snapshot_download(name, repo_type="dataset")
            print(f"dataset {name}")
        except Exception as e:  # noqa: BLE001
            print(f"FAILED  dataset {name}: {e}")


if __name__ == "__main__":
    main()
