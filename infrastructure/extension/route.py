"""The study route: one ordered list of 18 stages, read from the syllabus `route:` block.

The study guide, the reader's route sidebar and "Next on the route", and the extension chapter homes all
read the route through this module, so they agree on stage order, core flags, pacing and destinations.
`sync_config` calls `validate_route` (Spec #39, Study route).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SYLLABUS = Path("infrastructure/extension/syllabus.yaml")
ORIGINAL_CONFIG = Path("infrastructure/core/config_original.yaml")
SITE_ROOT = "https://learn.arena.education/"
TIME_LINE = re.compile(
    r"(?i)you should spend (?:up to |around |about |approximately )?~?(\d+)(?:\s*-\s*~?(\d+))? minutes"
)
PARTS = re.compile(r"^([1-9])-([1-9])?$")
MINUTES_PER_DAY = 6 * 60


@dataclass(frozen=True)
class Stop:
    """One place on the route: a section (or some of its parts), a planned lesson or a companion."""

    stage: int  # 1-based
    stage_title: str
    kind: str  # "original" | "extension" | "planned" | "companion"
    number: str  # section number, or the companion unit ("u02")
    title: str
    core: bool
    first_part: int | None = None  # numbered parts 1️⃣…; None means the whole section
    last_part: int | None = None  # None with first_part set means "to the end"
    days: float | None = None
    chapter: str | None = None  # chapter directory, for sections

    @property
    def label(self) -> str:
        if self.kind == "companion":
            return f"Alchemy {self.number} · {self.title}"
        return f"[{self.number}] {self.title}{self.parts_label}"

    @property
    def parts_label(self) -> str:
        if self.first_part is None:
            return ""
        if self.last_part is None:
            return f", parts {_emoji(self.first_part)} onward" if self.first_part > 1 else ""
        if self.first_part == self.last_part:
            return f", part {_emoji(self.first_part)}"
        return f", parts {_emoji(self.first_part)}–{_emoji(self.last_part)}"

    def covers(self, part_index: int, part_count: int) -> bool:
        """Whether a reader part (0 = the section's opening page) belongs to this stop."""
        first, last = self.reader_parts(part_count)
        return first <= part_index <= last

    def reader_parts(self, part_count: int) -> tuple[int, int]:
        """The reader part indices this stop covers; part k is the reader's index k, 0 is the opening."""
        if self.first_part is None:
            return 0, part_count - 1
        first = 0 if self.first_part == 1 else self.first_part
        last = part_count - 1 if self.last_part is None else self.last_part
        return first, min(last, part_count - 1)


def _emoji(number: int) -> str:
    return f"{number}️⃣"


def original_sections(root: Path = ROOT) -> dict[str, str]:
    """Upstream ARENA's sections (number → title) from its unchanged config, including split sections."""
    config = yaml.safe_load((Path(root) / ORIGINAL_CONFIG).read_text(encoding="utf-8"))
    titles = {}
    for chapter in config["chapters"].values():
        for section in chapter.get("sections", []):
            number = section.get("number")
            if number in config["conversion_map"]:
                titles[str(number)] = section["title"]
    return titles


def _extension_sections(doc: dict) -> dict[str, dict]:
    return {
        str(section["number"]): section
        for chapter in doc["chapters"].values()
        for section in chapter["sections"]
    }


def _parse_parts(value: object, where: str) -> tuple[int | None, int | None]:
    if value is None:
        return None, None
    match = PARTS.match(str(value))
    if not match:
        raise ValueError(f"{where}: parts must look like 1-4 or 5-, got {value!r}")
    first, last = int(match[1]), int(match[2]) if match[2] else None
    if last is not None and last < first:
        raise ValueError(f"{where}: parts {value!r} run backwards")
    return first, last


def validate_route(doc: dict, originals: dict[str, str]) -> None:
    """Raise ValueError unless the syllabus `route:` block is consistent.

    Every section number exists (upstream config or extension syllabus) and is on the route; a section
    appears once unless every appearance names its parts; planned labels don't shadow configured
    sections; companions match the `companions:` list one to one; every extension prerequisite exists
    and comes at an earlier stop, and a core stop's prerequisites are core stops.
    """
    stages = doc.get("route")
    if not isinstance(stages, list) or not stages:
        raise ValueError("syllabus `route:` must be a non-empty list of stages")
    extension = _extension_sections(doc)
    known = set(originals) | set(extension)
    companions = {entry["unit"] for entry in doc.get("companions") or []}

    first_stop: dict[str, int] = {}  # section → index of its first stop
    core_at_first: dict[str, bool] = {}
    appearances: dict[str, list[bool]] = {}  # section → [has parts] per appearance
    routed_companions: list[str] = []
    index = 0
    for stage_number, stage in enumerate(stages, start=1):
        if not str(stage.get("stage") or "").strip() or not stage.get("stops"):
            raise ValueError(f"route stage {stage_number} needs a `stage:` title and `stops:`")
        for stop in stage["stops"]:
            where = f"route stage {stage_number} ({stage['stage']})"
            kinds = [key for key in ("section", "planned", "companion") if key in stop]
            if len(kinds) != 1:
                raise ValueError(f"{where}: each stop needs exactly one of section, planned, companion: {stop}")
            if "core" in stop and not isinstance(stop["core"], bool):
                raise ValueError(f"{where}: core must be true or false: {stop}")
            if "section" in stop:
                number = str(stop["section"])
                if number not in known:
                    raise ValueError(f"{where}: unknown section {number}")
                _parse_parts(stop.get("parts"), f"{where} {number}")
                if "parts" in stop and stop.get("days") is None:
                    raise ValueError(f"{where}: section {number} names its parts, so it needs `days:`")
                appearances.setdefault(number, []).append("parts" in stop)
                if number not in first_stop:
                    first_stop[number] = index
                    core_at_first[number] = bool(stop.get("core"))
            elif "planned" in stop:
                number = str(stop["planned"])
                if number in known:
                    raise ValueError(
                        f"{where}: planned lesson {number} is now a configured section; make it a section stop"
                    )
                if not str(stop.get("title") or "").strip():
                    raise ValueError(f"{where}: planned lesson {number} needs a title")
                if stop.get("days") is None:
                    raise ValueError(f"{where}: planned lesson {number} has no time lines yet, so it needs `days:`")
            else:
                unit = str(stop["companion"])
                if unit not in companions:
                    raise ValueError(f"{where}: unknown companion {unit}; add it to `companions:` first")
                if stop.get("core"):
                    raise ValueError(f"{where}: companion {unit} is an optional side trip, never core")
                routed_companions.append(unit)
            index += 1

    for number, has_parts in appearances.items():
        if len(has_parts) > 1 and not all(has_parts):
            raise ValueError(f"section {number} appears more than once; name the parts at each stop")
    missing = sorted(known - set(first_stop), key=_number_key)
    if missing:
        raise ValueError(f"configured sections missing from the route: {missing}")
    unrouted = sorted(companions - set(routed_companions))
    duplicated = sorted({unit for unit in routed_companions if routed_companions.count(unit) > 1})
    if unrouted or duplicated:
        raise ValueError(f"each companion needs exactly one route stop; unrouted {unrouted}, repeated {duplicated}")

    for number, section in extension.items():
        for prerequisite in section.get("prerequisites") or []:
            prerequisite = str(prerequisite)
            if prerequisite not in known:
                raise ValueError(f"{number} has unknown prerequisite {prerequisite}")
            if first_stop[prerequisite] >= first_stop[number]:
                raise ValueError(f"{number} needs {prerequisite}, which must come earlier on the route")
            if core_at_first[number] and not core_at_first[prerequisite]:
                raise ValueError(
                    f"core stop {number} needs {prerequisite}, which is optional on the route; "
                    "make it core or drop the prerequisite"
                )


def _number_key(number: str) -> tuple[int, ...]:
    return tuple(int(part) for part in number.split("."))


def days_from_minutes(minutes: int) -> float:
    """Spec #39 pacing: one day per 6 h of time lines, rounded to half days, at least one day."""
    return max(1.0, math.floor(minutes / MINUTES_PER_DAY * 2 + 0.5) / 2)


def time_line_minutes(page_text: str) -> int:
    """Sum a page's "You should spend up to N minutes" lines, taking the upper end of a range."""
    return sum(int(high or low) for low, high in TIME_LINE.findall(page_text))


@dataclass(frozen=True)
class Route:
    """The loaded route: stages in order, their stops, and what each stop links to."""

    stages: tuple[tuple[str, tuple[Stop, ...]], ...]
    prerequisites: dict[str, tuple[str, ...]]
    companions: dict[str, dict]
    pages: dict[str, Path]  # section → its page file (configured sections only)
    _original_urls: dict[str, str]

    @property
    def stops(self) -> tuple[Stop, ...]:
        return tuple(stop for _, stops in self.stages for stop in stops)

    def stage_days(self, stage: int) -> tuple[float, float]:
        """(core days, all days) for one stage; companions are unsized."""
        stops = [stop for stop in self.stages[stage - 1][1] if stop.days is not None]
        return sum(stop.days for stop in stops if stop.core), sum(stop.days for stop in stops)

    def totals(self) -> tuple[float, float]:
        days = [self.stage_days(stage) for stage in range(1, len(self.stages) + 1)]
        return sum(core for core, _ in days), sum(everything for _, everything in days)

    def original_url(self, stop: Stop) -> str | None:
        """learn.arena.education's page for an original ARENA stop; None for anything else."""
        return self._original_urls.get(stop.number) if stop.kind == "original" else None

    def first_stop(self, number: str) -> Stop | None:
        return next((stop for stop in self.stops if stop.number == number and stop.kind != "companion"), None)

    def locate(self, number: str, part_index: int, part_count: int) -> Stop | None:
        """The stop a reader part belongs to (10.1 part 5️⃣ is the finale, not stage 10)."""
        for stop in self.stops:
            if stop.number == number and stop.kind != "companion" and stop.covers(part_index, part_count):
                return stop
        return None

    def next_core(self, stop: Stop) -> Stop | None:
        """The next core stop after this one; None at the end of the route."""
        stops = self.stops
        position = stops.index(stop)
        return next((later for later in stops[position + 1 :] if later.core), None)


def load(root: Path = ROOT) -> Route:
    """Read and check the syllabus route, with titles, pacing and destinations for every stop."""
    root = Path(root)
    doc = yaml.safe_load((root / SYLLABUS).read_text(encoding="utf-8"))
    originals = original_sections(root)
    validate_route(doc, originals)
    extension = _extension_sections(doc)
    chapter_of = {
        str(section["number"]): chapter for chapter, data in doc["chapters"].items() for section in data["sections"]
    }
    merged = yaml.safe_load((root / "infrastructure/core/config.yaml").read_text(encoding="utf-8"))
    pages: dict[str, Path] = {}
    urls: dict[str, str] = {}
    for chapter, data in merged["chapters"].items():
        for section in data.get("sections", []):
            number = str(section.get("number"))
            if number in originals or number in extension:
                chapter_of.setdefault(number, chapter)
                if section.get("page_file"):
                    pages[number] = root / chapter / "instructions" / "pages" / section["page_file"]
                if number in originals:
                    urls[number] = f"{SITE_ROOT}{chapter}/{section['id']}/"
    companions = {entry["unit"]: entry for entry in doc["companions"]}

    stages = []
    for stage_number, stage in enumerate(doc["route"], start=1):
        stops = []
        for raw in stage["stops"]:
            common = dict(stage=stage_number, stage_title=stage["stage"], core=bool(raw.get("core")))
            if "companion" in raw:
                unit = str(raw["companion"])
                stops.append(Stop(kind="companion", number=unit, title=companions[unit]["title"], **common))
                continue
            if "planned" in raw:
                stops.append(Stop(kind="planned", number=str(raw["planned"]), title=raw["title"],
                                  days=float(raw["days"]), **common))
                continue
            number = str(raw["section"])
            first, last = _parse_parts(raw.get("parts"), number)
            days = raw.get("days")
            if days is None:
                days = days_from_minutes(time_line_minutes(pages[number].read_text(encoding="utf-8")))
            title = originals[number] if number in originals else extension[number]["title"]
            kind = "original" if number in originals else "extension"
            stops.append(Stop(kind=kind, number=number, title=title, first_part=first, last_part=last,
                              days=float(days), chapter=chapter_of.get(number), **common))
        stages.append((stage["stage"], tuple(stops)))
    prerequisites = {
        number: tuple(str(item) for item in section.get("prerequisites") or ()) for number, section in extension.items()
    }
    return Route(tuple(stages), prerequisites, companions, pages, urls)
