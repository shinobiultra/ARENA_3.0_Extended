"""The study route: one ordered list of 18 stages, read from the syllabus `route:` block.

The study guide, the reader's route sidebar and "Next on the route", and the extension chapter homes all
read the route through this module, so they agree on stage order, core flags, pacing and destinations.
`sync_config` calls `validate_route` (Spec #39, Study route).
"""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import yaml

ROOT = Path(__file__).resolve().parents[2]
SYLLABUS = Path("infrastructure/extension/syllabus.yaml")
ORIGINAL_CONFIG = Path("infrastructure/core/config_original.yaml")
SITE_ROOT = "https://learn.arena.education/"
TIME_LINE = re.compile(
    r"(?i)you should spend (?:up to |around |about |approximately )?~?(\d+)(?:\s*-\s*~?(\d+))? minutes"
)
PART_BREAK = "=== NEW CHAPTER ==="
FIRST_EXTENSION_CHAPTER = 5
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
            return f", from part {_emoji(self.first_part)}" if self.first_part > 1 else ""
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
                raise ValueError(
                    f"{where}: each stop needs exactly one of section, planned, companion: {stop}"
                )
            if "core" in stop and not isinstance(stop["core"], bool):
                raise ValueError(f"{where}: core must be true or false: {stop}")
            if "section" in stop:
                number = str(stop["section"])
                if number not in known:
                    raise ValueError(f"{where}: unknown section {number}")
                _parse_parts(stop.get("parts"), f"{where} {number}")
                if "parts" in stop and stop.get("days") is None:
                    raise ValueError(
                        f"{where}: section {number} names its parts, so it needs `days:`"
                    )
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
                    raise ValueError(
                        f"{where}: planned lesson {number} has no time lines yet, so it needs `days:`"
                    )
            else:
                unit = str(stop["companion"])
                if unit not in companions:
                    raise ValueError(
                        f"{where}: unknown companion {unit}; add it to `companions:` first"
                    )
                if stop.get("core"):
                    raise ValueError(
                        f"{where}: companion {unit} is an optional side trip, never core"
                    )
                routed_companions.append(unit)
            index += 1

    for number, has_parts in appearances.items():
        if len(has_parts) > 1 and not all(has_parts):
            raise ValueError(
                f"section {number} appears more than once; name the parts at each stop"
            )
    missing = sorted(known - set(first_stop), key=_number_key)
    if missing:
        raise ValueError(f"configured sections missing from the route: {missing}")
    unrouted = sorted(companions - set(routed_companions))
    duplicated = sorted({unit for unit in routed_companions if routed_companions.count(unit) > 1})
    if unrouted or duplicated:
        raise ValueError(
            f"each companion needs exactly one route stop; unrouted {unrouted}, repeated {duplicated}"
        )

    for number, section in extension.items():
        for prerequisite in section.get("prerequisites") or []:
            prerequisite = str(prerequisite)
            if prerequisite not in known:
                raise ValueError(f"{number} has unknown prerequisite {prerequisite}")
            if first_stop[prerequisite] >= first_stop[number]:
                raise ValueError(
                    f"{number} needs {prerequisite}, which must come earlier on the route"
                )
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
    part_counts: dict[str, int]  # extension section → number of reader parts on its page

    @property
    def stops(self) -> tuple[Stop, ...]:
        return tuple(stop for _, stops in self.stages for stop in stops)

    def stage_days(self, stage: int) -> tuple[float, float]:
        """(core days, all days) for one stage; companions are unsized."""
        sized = [
            (stop.core, stop.days) for stop in self.stages[stage - 1][1] if stop.days is not None
        ]
        return sum(days for core, days in sized if core), sum(days for _, days in sized)

    def totals(self) -> tuple[float, float]:
        days = [self.stage_days(stage) for stage in range(1, len(self.stages) + 1)]
        return sum(core for core, _ in days), sum(everything for _, everything in days)

    def original_url(self, stop: Stop) -> str | None:
        """learn.arena.education's page for an original ARENA stop; None for anything else."""
        return self._original_urls.get(stop.number) if stop.kind == "original" else None

    def first_stop(self, number: str) -> Stop | None:
        return next(
            (stop for stop in self.stops if stop.number == number and stop.kind != "companion"),
            None,
        )

    def stop_for(self, number: str) -> Stop:
        """The first stop of a section that `validate_route` guarantees is on the route."""
        stop = self.first_stop(number)
        if stop is None:
            raise KeyError(f"section {number} is not on the route")
        return stop

    def locate(self, number: str, part_index: int, part_count: int) -> Stop | None:
        """The stop a reader part belongs to (10.1 part 5️⃣ is the finale, not stage 10)."""
        for stop in self.stops:
            if (
                stop.number == number
                and stop.kind != "companion"
                and stop.covers(part_index, part_count)
            ):
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
        str(section["number"]): chapter
        for chapter, data in doc["chapters"].items()
        for section in data["sections"]
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
    # Only extension chapters (5+) own a chapter home; 0.7 is a shelf, not part of upstream's chapter 0.
    chapter_numbers = {
        data["number"]: chapter
        for chapter, data in doc["chapters"].items()
        if data["number"] >= FIRST_EXTENSION_CHAPTER
    }
    part_counts = {
        number: len(pages[number].read_text(encoding="utf-8").split(PART_BREAK))
        for number in extension
        if number in pages and pages[number].is_file()
    }

    stages = []
    for stage_number, stage in enumerate(doc["route"], start=1):
        stops = []
        for raw in stage["stops"]:
            title_of_stage, core = str(stage["stage"]), bool(raw.get("core"))
            if "companion" in raw:
                unit = str(raw["companion"])
                stops.append(
                    Stop(
                        stage_number,
                        title_of_stage,
                        "companion",
                        unit,
                        companions[unit]["title"],
                        core,
                    )
                )
                continue
            if "planned" in raw:
                number = str(raw["planned"])
                stops.append(
                    Stop(
                        stage_number,
                        title_of_stage,
                        "planned",
                        number,
                        str(raw["title"]),
                        core,
                        days=float(raw["days"]),
                        chapter=chapter_numbers.get(int(number.split(".")[0])),
                    )
                )
                continue
            number = str(raw["section"])
            first, last = _parse_parts(raw.get("parts"), number)
            count = part_counts.get(number)
            if first is not None and count is not None and max(first, last or first) > count - 1:
                raise ValueError(
                    f"route stage {stage_number}: {number} has parts 1–{count - 1} on its page, "
                    f"but the route names parts {raw['parts']}"
                )
            days = raw.get("days")
            if days is None:
                days = days_from_minutes(
                    time_line_minutes(pages[number].read_text(encoding="utf-8"))
                )
            title = originals[number] if number in originals else extension[number]["title"]
            kind = "original" if number in originals else "extension"
            stops.append(
                Stop(
                    stage_number,
                    title_of_stage,
                    kind,
                    number,
                    str(title),
                    core,
                    first_part=first,
                    last_part=last,
                    days=float(days),
                    chapter=chapter_of.get(number),
                )
            )
        stages.append((stage["stage"], tuple(stops)))
    prerequisites = {
        number: tuple(str(item) for item in section.get("prerequisites") or ())
        for number, section in extension.items()
    }
    return Route(tuple(stages), prerequisites, companions, pages, urls, part_counts)


def _stamp(root: Path) -> tuple[float, ...]:
    files = (SYLLABUS, Path("infrastructure/core/config.yaml"), ORIGINAL_CONFIG)
    return tuple((Path(root) / name).stat().st_mtime for name in files)


@functools.lru_cache(maxsize=4)
def _load_cached(root: Path, stamp: tuple[float, ...]) -> Route:
    return load(root)


def load_cached(root: Path = ROOT) -> Route:
    """`load`, reused across Streamlit reruns until the syllabus or configs change."""
    root = Path(root).resolve()
    return _load_cached(root, _stamp(root))


# ---------- companions (#322) ----------

COMPANION_PAGE = "ALCHEMY.md"  # the companion list (Batch 4, #269)
COMPANION_LABS = {"u02": "mastery_hub/app.py"}  # the optimizer lab rides with u02
LAB_COMMAND = "uv run streamlit run {path} --server.address=127.0.0.1 --server.port=18506"


def companion_materials(route: Route, unit: str, root: Path = ROOT) -> list[tuple[str, str]]:
    """What a side trip can link to: (label, repo path), only for files that exist in `root`."""
    root = Path(root)
    entry = route.companions[unit]
    materials = []
    if (
        entry.get("status") in ("written", "finished")
        and entry.get("dir")
        and (root / entry["dir"]).is_dir()
    ):
        materials.append(("companion notebook", str(entry["dir"])))
    lab = COMPANION_LABS.get(unit)
    if lab and (root / lab).is_file():
        materials.append(("optimizer lab", lab))
    if (root / COMPANION_PAGE).is_file():
        materials.append(("companion list", COMPANION_PAGE))
    return materials


# ---------- the study guide (#315) ----------

GUIDE = Path("mastery_hub/STUDY-GUIDE.md")
GUIDE_MARKERS = (
    (
        "<!-- The stage table below is generated from the syllabus route: "
        "uv run python infrastructure/extension/route.py --write -->"
    ),
    "<!-- End of the generated stage table. -->",
)


def _guide_target(route: Route, stop: Stop, root: Path) -> str | None:
    """Where the guide links a stop: originals to learn.arena.education, extension pages by relative path."""
    if stop.kind == "original":
        return route.original_url(stop)
    if (
        stop.kind == "extension"
        and stop.number in route.pages
        and route.pages[stop.number].is_file()
    ):
        relative = route.pages[stop.number].resolve().relative_to(Path(root).resolve())
        return "../" + quote(relative.as_posix())
    return None


def _guide_ref(route: Route, stop: Stop, root: Path, *, short: bool = False) -> str:
    text = f"[{stop.number}]{stop.parts_label}" if short else stop.label
    target = _guide_target(route, stop, root)
    if target is None:
        return text
    return (
        f"[{text.replace('[', '').replace(']', '')}]({target})" if short else f"[{text}]({target})"
    )


def _companion_line(route: Route, stop: Stop, root: Path) -> str:
    parts = []
    for label, target in companion_materials(route, stop.number, root):
        if label == "optimizer lab":
            parts.append(f"with the optimizer lab (`{LAB_COMMAND.format(path=target)}`)")
        else:
            parts.append(f"[{label}](../{quote(target)})")
    suffix = f": {', '.join(parts)}" if parts else ""
    return f"{stop.label}, an optional side trip{suffix}"


def _days(days: float) -> str:
    return "1 day" if days == 1 else f"{days:g} days"


def _weeks(days: float) -> str:
    return f"{days / 5:.0f}"


def render_guide(route: Route, root: Path = ROOT) -> str:
    """The guide's generated part: the stage table, totals, and each stage's stops."""
    root = Path(root)
    lines = [
        "| Stage | | Core | Optional | Core days | All days |",
        "|---|---|---|---|---|---|",
    ]
    for number, (title, stops) in enumerate(route.stages, start=1):
        core = ", ".join(_guide_ref(route, s, root, short=True) for s in stops if s.core) or "—"
        optional = [
            _guide_ref(route, s, root, short=True)
            for s in stops
            if not s.core and s.kind != "companion"
        ]
        optional += [f"{s.number} (side trip)" for s in stops if s.kind == "companion"]
        core_days, all_days = route.stage_days(number)
        lines.append(
            f"| {number} | {title} | {core} | {', '.join(optional) or '—'} | {core_days:g} | {all_days:g} |"
        )
    core_total, all_total = route.totals()
    lines += [
        "",
        (
            f"The core route is about {core_total:g} days (about {_weeks(core_total)} weeks full-time). Everything, "
            f"optional lessons included, is about {all_total:g} days (about {_weeks(all_total)} weeks). Alchemy side "
            "trips are not counted."
        ),
        "",
        "## The route, stage by stage",
    ]
    for number, (title, stops) in enumerate(route.stages, start=1):
        core_days, all_days = route.stage_days(number)
        lines += ["", f"### {number}. {title}", ""]
        core = [s for s in stops if s.core]
        optional = [s for s in stops if not s.core and s.kind != "companion"]
        trips = [s for s in stops if s.kind == "companion"]
        if core:
            lines.append(f"Core ({_days(core_days)}):")
            lines.append("")
            lines += [f"- {_guide_ref(route, s, root)}{_where(s)}" for s in core]
        if optional:
            lines += ["", "Optional, in any order:", ""]
            lines += [
                f"- {_guide_ref(route, s, root)}{_where(s)}{_needs(route, s)}" for s in optional
            ]
        if trips:
            lines += ["", "Side trips:", ""]
            lines += [f"- {_companion_line(route, s, root)}" for s in trips]
    return "\n".join(lines) + "\n"


def _where(stop: Stop) -> str:
    return " (original ARENA)" if stop.kind == "original" else ""


def _needs(route: Route, stop: Stop, ref: Callable[[Stop], str] | None = None) -> str:
    """For an optional lesson: its prerequisites that are themselves optional stops."""
    needed = [route.stop_for(number) for number in route.prerequisites.get(stop.number, ())]
    extra = [prerequisite for prerequisite in needed if not prerequisite.core]
    if not extra:
        return ""
    ref = ref or (lambda s: f"[{s.number}] {s.title}")
    return " (after " + _and([ref(s) for s in extra]) + ")"


def _and(items: list[str]) -> str:
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def write_guide(root: Path = ROOT, *, check: bool = False) -> bool:
    """Regenerate the guide's stage table; with check=True only report whether it is current."""
    root = Path(root)
    path = root / GUIDE
    text = path.read_text(encoding="utf-8")
    start, end = GUIDE_MARKERS
    if start not in text or end not in text:
        raise ValueError(f"{GUIDE} lacks the generated-table markers")
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    updated = f"{before}{start}\n\n{render_guide(load(root), root)}\n{end}{after}"
    if updated == text:
        return True
    if not check:
        path.write_text(updated, encoding="utf-8")
    return False


# ---------- chapter homes (#321) ----------


def _plain(stop: Stop) -> str:
    return stop.label


def chapter_tldr(route: Route, chapter: str, ref: Callable[[Stop], str] = _plain) -> str:
    """An ARENA-style TLDR for an extension chapter home, from the route: before, during and after."""
    match = re.match(r"chapter(\d+)_", chapter)
    if match is None or int(match[1]) < FIRST_EXTENSION_CHAPTER:
        return ""  # chapters 0–4 are upstream's; their homes are not the fork's to direct
    stops = route.stops
    segments: list[list[Stop]] = []
    for stop in stops:
        if stop.chapter == chapter and stop.kind in ("extension", "planned"):
            if segments and segments[-1][-1].stage == stop.stage:
                segments[-1].append(stop)
            else:
                segments.append([stop])
    if not segments:
        return ""
    lines = ["## TLDR (if you're in a hurry!)", ""]
    if len(segments) == 1:
        segment = segments[0]
        lines += [
            f"This chapter is stage {segment[0].stage} of the study route, *{segment[0].stage_title}*.",
            "",
        ]
        lines += _segment_directions(route, chapter, segment, ref)
    else:
        lines += ["This chapter's lessons sit at more than one stage of the study route.", ""]
        for segment in segments:
            lines += [f"**Stage {segment[0].stage} · {segment[0].stage_title}**", ""]
            lines += _segment_directions(route, chapter, segment, ref)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _segment_directions(
    route: Route, chapter: str, segment: list[Stop], ref: Callable[[Stop], str]
) -> list[str]:
    lines = []
    core = [stop for stop in segment if stop.core]
    optional = [stop for stop in segment if not stop.core]
    in_chapter = {stop.number for stop in route.stops if stop.chapter == chapter}
    if core:
        before = []
        continuing = all(route.first_stop(stop.number) is not stop for stop in core)
        for stop in core:
            for number in route.prerequisites.get(stop.number, ()):
                if number not in in_chapter and number not in {s.number for s in before}:
                    before.append(route.stop_for(number))
        before.sort(key=route.stops.index)
        if continuing:
            prepared = f"the core of stages 1–{core[0].stage - 1}"
        else:
            prepared = _and([ref(stop) for stop in before]) if before else "no other lessons"
        lines.append(f"- **Before you start:** {prepared}.")
        lines.append(f"- **Start with** {ref(core[0])}.")
        lines.append(f"- **Core:** {_and([ref(stop) for stop in core])}.")
    if optional:
        described = [f"{ref(stop)}{_needs(route, stop, ref)}" for stop in optional]
        lines.append(f"- **Optional, in any order:** {'; '.join(described)}.")
    following = route.next_core(segment[-1])
    while (
        following is not None
        and following.chapter == chapter
        and following.stage == segment[-1].stage
    ):
        following = route.next_core(following)
    if following is None:
        lines.append("- **Then:** this is the end of the route.")
    elif following.stage == segment[-1].stage:
        lines.append(f"- **Then:** the route continues with {ref(following)}.")
    else:
        lines.append(
            f"- **Then:** the route continues with stage {following.stage}, *{following.stage_title}*: "
            f"{ref(following)}."
        )
    return lines


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Regenerate or check the study guide's stage table."
    )
    parser.add_argument("--check", action="store_true", help="exit 1 if the guide is out of date")
    parser.add_argument("--write", action="store_true", help="rewrite the guide's generated table")
    args = parser.parse_args()
    current = write_guide(check=not args.write)
    if args.write:
        print("mastery_hub/STUDY-GUIDE.md " + ("was current" if current else "updated"))
    elif not current:
        raise SystemExit("mastery_hub/STUDY-GUIDE.md is out of date: run with --write")
