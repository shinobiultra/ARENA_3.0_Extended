"""Build the generated ARENA config from frozen original + extension overlay."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "infrastructure" / "core"
ORIGINAL_CONFIG = CONFIG_DIR / "config_original.yaml"
EXTENSION_CONFIG = CONFIG_DIR / "config_extension.yaml"
MERGED_CONFIG = CONFIG_DIR / "config.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _section_map(chapter: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(section.get("number")): section for section in chapter.get("sections", [])}


def merge_config(original: dict[str, Any], extension: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic merged config, rejecting edits to original entries."""

    merged = copy.deepcopy(original)

    for number, mapping in extension.get("conversion_map", {}).items():
        number = str(number)
        if number in merged["conversion_map"]:
            raise ValueError(f"extension conversion_map duplicates original section {number}")
        merged["conversion_map"][number] = copy.deepcopy(mapping)

    for key in ("chapter_names", "chapter_names_long"):
        for chapter_index, value in extension.get(key, {}).items():
            if chapter_index in merged[key]:
                raise ValueError(f"extension {key} duplicates original chapter {chapter_index}")
            merged[key][chapter_index] = value

    for chapter_name, extension_chapter in extension.get("chapters", {}).items():
        if chapter_name not in merged["chapters"]:
            merged["chapters"][chapter_name] = copy.deepcopy(extension_chapter)
            continue

        original_chapter = merged["chapters"][chapter_name]
        extension_metadata = {
            key: value for key, value in extension_chapter.items() if key != "sections"
        }
        if extension_metadata:
            raise ValueError(
                f"extension chapter {chapter_name!r} may only add sections to original chapters"
            )

        original_sections = _section_map(original_chapter)
        for section in extension_chapter.get("sections", []):
            number = str(section.get("number"))
            if number in original_sections:
                raise ValueError(
                    f"extension chapter {chapter_name!r} duplicates original section {number}"
                )
            original_chapter.setdefault("sections", []).append(copy.deepcopy(section))

    return merged


def dump_config(config: dict[str, Any]) -> str:
    return yaml.safe_dump(
        config,
        sort_keys=False,
        allow_unicode=True,
        width=1000,
    )


def build_config_text(
    *,
    original_path: Path = ORIGINAL_CONFIG,
    extension_path: Path = EXTENSION_CONFIG,
) -> str:
    return dump_config(merge_config(load_yaml(original_path), load_yaml(extension_path)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="rewrite infrastructure/core/config.yaml")
    parser.add_argument("--check", action="store_true", help="verify config.yaml matches the merge")
    args = parser.parse_args()

    generated = build_config_text()
    if args.write:
        MERGED_CONFIG.write_text(generated)
    if args.check:
        current = load_yaml(MERGED_CONFIG)
        expected = yaml.safe_load(generated)
        if current != expected:
            print("MERGED_CONFIG=FAIL")
            print(f"{MERGED_CONFIG.relative_to(ROOT)} differs from generated split config")
            raise SystemExit(1)
        print("MERGED_CONFIG=PASS")
    if not args.write and not args.check:
        sys.stdout.write(generated)


if __name__ == "__main__":
    main()
