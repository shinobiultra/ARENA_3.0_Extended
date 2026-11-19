#!/usr/bin/env python
"""Generate `infrastructure/core/config_extension.yaml` from `infrastructure/extension/syllabus.yaml`.

Then run `scripts/build_merged_config.py` to merge it with the original ARENA config.

    uv run python infrastructure/extension/sync_config.py
    uv run python scripts/build_merged_config.py --write
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

try:
    from infrastructure.extension.route import original_sections, validate_route
except ModuleNotFoundError:  # run as a script: `python infrastructure/extension/sync_config.py`
    from route import original_sections, validate_route  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[2]
SYLLABUS = ROOT / "infrastructure" / "extension" / "syllabus.yaml"
CONFIG = ROOT / "infrastructure" / "core" / "config_extension.yaml"
REGISTRY = ROOT / "docs" / "roadmap_requirement_registry.yml"

DEFAULT_HEADER = (
    "https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/headers/header-13-2.png"
)
CHAPTER_STYLE = {
    5: ("#2563EB", "cpu", "header-11.png"),
    6: ("#7C3AED", "microscope", "header-13-2.png"),
    7: ("#0891B2", "microscope", "header-14-1.png"),
    8: ("#0F766E", "workflow", "header-13-2.png"),
    9: ("#BE123C", "shield-check", "header-14-1.png"),
    10: ("#CA8A04", "graduation-cap", "header-13-2.png"),
    11: ("#4F46E5", "scatter-chart", "header-13-2.png"),
    12: ("#0891B2", "image", "header-15-3.png"),
    13: ("#DB2777", "image-plus", "header-05.png"),
    14: ("#16A34A", "network", "header-15-3.png"),
    15: ("#7C3AED", "sliders", "header-15-3.png"),
    16: ("#9333EA", "split", "header-13-2.png"),
    17: ("#2563EB", "timeline", "header-15-2.png"),
    18: ("#0D9488", "puzzle", "header-12.png"),
}


COMPANION_STATUSES = ("planned", "written", "finished")
COMPANION_UNIT = re.compile(r"^u0[1-7]$")


def validate_companions(companions: object, root: Path = ROOT) -> None:
    """Raise ValueError if the syllabus `companions:` list is malformed."""
    if not isinstance(companions, list) or not companions:
        raise ValueError("syllabus `companions:` must be a non-empty list")
    seen: set[str] = set()
    for entry in companions:
        unit = entry.get("unit") if isinstance(entry, dict) else None
        if not isinstance(unit, str) or not COMPANION_UNIT.match(unit):
            raise ValueError(f"companion unit must look like u01..u07, got {unit!r}")
        if unit in seen:
            raise ValueError(f"duplicate companion unit {unit}")
        seen.add(unit)
        if not str(entry.get("title") or "").strip():
            raise ValueError(f"companion {unit} needs a title")
        status = entry.get("status")
        if status not in COMPANION_STATUSES:
            raise ValueError(
                f"companion {unit} has status {status!r}; use one of {COMPANION_STATUSES}"
            )
        directory = entry.get("dir")
        if directory is None:
            if status != "planned":
                raise ValueError(f"companion {unit} is {status} and requires a dir")
        elif not (root / directory).is_dir():
            raise ValueError(f"companion {unit} dir {directory} does not exist")


def validate_requirements(assigned: dict[str, str], registry_ids: set[str]) -> None:
    """Raise ValueError if the syllabus maps a requirement id the registry does not have."""
    unknown = sorted(set(assigned) - registry_ids)
    if unknown:
        raise ValueError(f"unknown requirement ids in syllabus: {unknown}")


def slug(title: str) -> str:
    title = title.replace("&", "and").replace(" - ", " ")
    title = re.sub(r"[^A-Za-z0-9\-\s]", "", title)
    return re.sub(r"\s+", "_", title.strip())


def page_name(section: dict, index: int) -> str:
    prefix = section.get("page_prefix", f"{index:02d}")
    return f"{prefix}_[{section['number']}]_{slug(section['title'])}"


def main() -> None:
    syllabus_doc = yaml.safe_load(SYLLABUS.read_text())
    validate_companions(syllabus_doc.get("companions"))
    validate_route(syllabus_doc, original_sections(ROOT))
    syllabus = syllabus_doc["chapters"]
    config = {"conversion_map": {}, "chapter_names": {}, "chapter_names_long": {}, "chapters": {}}
    assigned: dict[str, str] = {}

    for chapter_name, chapter in syllabus.items():
        number = chapter["number"]
        sections_out = []
        for index, section in enumerate(chapter["sections"], start=1):
            page = page_name(section, index)
            config["conversion_map"][section["number"]] = {
                "streamlit_page": page,
                "exercise_dir": section["dir"],
            }
            color, icon, header = CHAPTER_STYLE.get(number, ("#2563EB", "book", "header-13-2.png"))
            header_url = (
                f"https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/headers/{header}"
            )
            description = section.get("description") or section["title"]
            sections_out.append(
                {
                    "id": f"{index}_{section['dir'].split('_', 1)[1]}",
                    "number": section["number"],
                    "title": section["title"],
                    "exercise_dir": section["dir"],
                    "page_file": f"{page}.md",
                    "header_image": header_url
                    if number >= 5
                    else "https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/headers/header-00.png",
                    "streamlit_description": description,
                    "streamlit_description_short": description.split(". ")[0].rstrip("."),
                }
            )
            for req in section.get("requirements", []):
                if req in assigned:
                    raise ValueError(
                        f"{req} is assigned to both {assigned[req]} and {section['number']}"
                    )
                assigned[req] = section["number"]

        if number >= 5:
            color, icon, header = CHAPTER_STYLE.get(number, ("#2563EB", "book", "header-13-2.png"))
            config["chapter_names"][number] = chapter_name
            config["chapter_names_long"][number] = chapter["title"].replace(":", " -")
            config["chapters"][chapter_name] = {
                "title": chapter["title"],
                "short_title": chapter["title"].split(": ", 1)[1],
                "short_description": chapter["short_description"],
                "description": chapter["short_description"],
                "color": color,
                "icon": icon,
                "header_image": f"https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/headers/{header}",
                "sections": sections_out,
            }
        else:
            config["chapters"][chapter_name] = {"sections": sections_out}

    CONFIG.write_text(
        "# Generated by infrastructure/extension/sync_config.py from infrastructure/extension/syllabus.yaml.\n"
        "# Edit the syllabus, not this file.\n"
        + yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=1000),
        encoding="utf-8",
    )
    n_sections = len(config["conversion_map"])
    print(f"Wrote {CONFIG.relative_to(ROOT)} with {n_sections} sections")

    registry_ids = {r["id"] for r in yaml.safe_load(REGISTRY.read_text())["requirements"]}
    validate_requirements(assigned, registry_ids)
    unassigned = sorted(registry_ids - set(assigned))
    print(f"{len(assigned)} roadmap requirements assigned to sections")
    if unassigned:
        print(f"Not assigned to any extension section ({len(unassigned)}): {unassigned}")


if __name__ == "__main__":
    main()
