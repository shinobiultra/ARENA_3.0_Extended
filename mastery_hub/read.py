"""Read current ARENA extension lessons without loading chat or teaching code."""
from __future__ import annotations

from contextlib import nullcontext
from html import escape
from pathlib import Path
import re
import shlex
import sys
from urllib.parse import urlencode

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infrastructure.extension import route as study_route

_SUPPLEMENTS = {'*': (('Download study guide', 'mastery_hub/STUDY-GUIDE.md'),)}


def supplements(number: str, root: Path) -> list[tuple[str, Path]]:
    """Offer only these named public resources; section text is never a path."""
    return [(label, Path(root) / path) for label, path in _SUPPLEMENTS["*"] + _SUPPLEMENTS.get(number, ())]


def inventory(root: Path, *, configuration_bytes: dict[str, bytes] | None = None) -> list[dict]:
    """Resolve only configured lesson identities; never glob historical neighbors."""
    root = Path(root)
    if configuration_bytes is None:
        configuration_bytes = {
            name: (root / name).read_bytes()
            for name in ("infrastructure/extension/syllabus.yaml", "infrastructure/core/config_extension.yaml")
        }
    syllabus = yaml.safe_load(configuration_bytes["infrastructure/extension/syllabus.yaml"])
    config = yaml.safe_load(configuration_bytes["infrastructure/core/config_extension.yaml"])
    rows = []
    original_titles = {0: "Chapter 0: Fundamentals", 1: "Chapter 1: Transformer Interpretability"}
    for chapter, data in syllabus["chapters"].items():
        chapter_title = data.get(
            "title", original_titles[data["number"]] if data["number"] < 5 else chapter,
        )
        for section in data["sections"]:
            number = str(section["number"])
            mapping = config["conversion_map"][number]
            page_name = mapping["streamlit_page"]
            book_name = page_name.split("_[", 1)[1].replace("]", "")
            directory = root / chapter / "exercises" / mapping["exercise_dir"]
            page = root / chapter / "instructions/pages" / f"{page_name}.md"
            master = root / "infrastructure/chapters" / chapter / f"master_{number.replace('.', '_')}.py"
            exercise = directory / f"{book_name}_exercises.ipynb"
            reference = directory / f"{book_name}_solutions.ipynb"
            missing = [path for path in (page, exercise, reference) if not path.is_file()]
            available = master.is_file() and not missing
            status = "Available"
            if not master.is_file():
                status = "Not available yet: this configured lesson has no current authoring master."
            elif missing:
                status = "Lesson unavailable. Restore the configured files: " + ", ".join(str(p.relative_to(root)) for p in missing)
            rows.append(dict(number=number, title=section["title"], chapter=chapter,
                             chapter_title=chapter_title,
                             page=page, exercises=exercise, reference=reference,
                             available=available, status=status))
    return rows


def parts(page: Path, *, with_media: bool = False) -> list[str]:
    """Split a lesson page into its parts, in order."""
    text = page.read_bytes().decode("utf-8")
    if with_media:
        from infrastructure.extension.render_media import inline_page_media

        text = inline_page_media(text, page)
    return text.split("=== NEW CHAPTER ===")


def part_title(text: str, index: int) -> str:
    match = re.search(r"(?m)^#{1,6}\s+(.+)$", text)
    return match[1] if match else f"Part {index + 1}"


def lesson_url(section: str, part: int = 0) -> str:
    """Bookmark a configured lesson; public part numbers start at one."""
    return "?" + urlencode({"section": section, "part": part + 1})


def route_link(label: str, section: str, part: int = 0) -> str:
    # Streamlit's Markdown links default to a new tab; reader navigation stays here.
    return f'<a href="{escape(lesson_url(section, part), quote=True)}" target="_self">{escape(label)}</a>'


def resolve_route(query, rows: list[dict]) -> tuple[dict | None, int, str | None]:
    """Validate identities before reading a page; URL values never become paths."""
    values = {}
    for key in ("section", "part"):
        if hasattr(query, "get_all"):
            value = query.get_all(key)
        else:
            value = query.get(key, [])
            if not isinstance(value, list):
                value = [value]
        if len(value) > 1:
            return None, 0, f"Use only one {key} in the URL. Choose a Start here link below."
        values[key] = value[0] if value else None
    section, raw_part = values["section"], values["part"]
    if section is None:
        if raw_part is not None:
            return None, 0, "A part needs a section, for example ?section=7.1&part=1. Choose a Start here link below."
        return None, 0, None
    row = next((row for row in rows if row["number"] == section), None)
    if row is None:
        return None, 0, "Unknown configured section. Use a section number such as 7.1; choose a Start here link below."
    if not row["available"]:
        return None, 0, f"[{section}] {row['title']} is not available. {row['status']} Choose a Start here link below."
    count = len(parts(row["page"]))
    if raw_part is None:
        raw_part = "1"
    # Bound length before int conversion, including Python's integer-string limit.
    if not isinstance(raw_part, str) or not re.fullmatch(r"[0-9]{1,8}", raw_part) or not 1 <= int(raw_part) <= count:
        return None, 0, f"Part must be a whole number from 1 to {count} for [{section}]. Use {lesson_url(section)} or choose a Start here link below."
    return row, int(raw_part) - 1, None


def start_panel(st) -> None:
    st.markdown(
        "### Start here\n"
        "- **New to ARENA?** Start at original ARENA "
        '<a href="https://learn.arena.education/chapter0_fundamentals/00_prereqs/" target="_blank">0.0 Prerequisites</a>, '
        "stage 1 of the route, and follow the sidebar's stages in order.\n"
        f"- **Done ARENA chapters 0–1?** Start at {route_link('0.6 How to Know When an Interpretability Result Is Fake', '0.6')} "
        "(stage 4).\n\n"
        "Core lessons make the route; the rest of each stage is optional, in any order. Original ARENA lessons "
        "open on learn.arena.education. A lesson's last part links to the next stop on the route.",
        unsafe_allow_html=True,
    )


def _available(rows: list[dict]) -> dict[str, dict]:
    return {row["number"]: row for row in rows if row["available"]}


def opens_in_reader(stop, rows: list[dict]) -> bool:
    """Whether a route stop has a destination: learn.arena.education, or an available reader lesson."""
    return stop.kind == "original" or (stop.kind == "extension" and stop.number in _available(rows))


def stop_link(route, stop, rows: list[dict]) -> str:
    """A route stop as HTML: originals open on learn.arena.education, extension stops in the reader."""
    if stop.kind == "original":
        url = route.original_url(stop)
        return f'<a href="{escape(url, quote=True)}" target="_blank">{escape(stop.label)}</a> ↗'
    row = _available(rows).get(stop.number)
    if stop.kind == "extension" and row is not None:
        first, _ = stop.reader_parts(len(parts(row["page"])))
        return route_link(stop.label, stop.number, first)
    return escape(stop.label)


def next_on_route(route, rows: list[dict], stop):
    """(core stops the reader can't open yet, the first one it can open or None at the end of the route)."""
    waiting = []
    following = route.next_core(stop)
    while following is not None and not opens_in_reader(following, rows):
        waiting.append(following)
        following = route.next_core(following)
    return waiting, following


def side_trip(route, stop) -> str:
    """An Alchemy side trip, with the materials that exist in this checkout."""
    text = f"↪ {escape(stop.label)}, optional side trip"
    for label, target in study_route.companion_materials(route, stop.number, ROOT):
        if label == "optimizer lab":
            text += f"; {label}: <code>{escape(study_route.LAB_COMMAND.format(path=target))}</code>"
        else:
            text += f"; {label}: <code>{escape(target)}</code> in this checkout"
    return text + ' (<a href="https://deep-learning-alchemy.github.io/" target="_blank">CS 312</a> ↗)'


def route_sidebar(st, route, rows: list[dict], current) -> None:
    """All 18 stages; the current one is open and marked."""
    st.sidebar.markdown("## Study route")
    for number, (title, stops) in enumerate(route.stages, start=1):
        here = current is not None and current.stage == number
        core_days, all_days = route.stage_days(number)
        lines = []
        for stop in stops:
            if stop.kind == "companion":
                lines.append(f"- {side_trip(route, stop)}")
                continue
            text = stop_link(route, stop, rows)
            if stop == current:
                text = f"<b>{text}</b> ← you are here"
            lines.append(f"- {'●' if stop.core else '○'} {text}{'' if stop.core else ' (optional)'}")
        body = f"Core {core_days:g} days, {all_days:g} in all. ● core, ○ optional.\n\n" + "\n".join(lines)
        container = st.sidebar.expander(f"{'📍 ' if here else ''}{number}. {title}", expanded=here) \
            if hasattr(st.sidebar, "expander") else nullcontext()
        with container:
            (st if hasattr(st.sidebar, "expander") else st.sidebar).markdown(body, unsafe_allow_html=True)


def next_step(st, route, rows: list[dict], stop) -> None:
    """At the end of a route stop: "Next on the route →", or the end of the route."""
    waiting, following = next_on_route(route, rows, stop)
    if following is None and not waiting:
        trips = [s for s in route.stages[stop.stage - 1][1] if s.kind == "companion"]
        extra = "".join(f"<br>{side_trip(route, trip)}" for trip in trips)
        st.markdown(f"🏁 **You've reached the end of the route.**{extra}", unsafe_allow_html=True)
        return
    later = "".join(f"{escape(s.label)} (not in the reader yet), then " for s in waiting)
    target = stop_link(route, following, rows) if following is not None else "the end of the route"
    st.markdown(f"**Next on the route →** {later}{target}", unsafe_allow_html=True)


def at_stop_end(stop, page_parts: list[str], part: int) -> bool:
    """The stop's last part, or its last numbered part when only the ☆ Bonus follows."""
    first, last = stop.reader_parts(len(page_parts))
    if part == last:
        return True
    return part == last - 1 and part >= first and part_title(page_parts[last], last).startswith("☆")


def notebook_handoff(st, root: Path, lesson: dict) -> None:
    directory = lesson["exercises"].parent.resolve()
    name = lesson["exercises"].name
    root = root.resolve()
    python = Path(sys.executable).absolute()
    st.markdown(
        "### Open the exercises notebook locally\n"
        f"1. The exercises notebook **{name}** is in this directory of your checkout "
        "(if you downloaded a copy, save it here so its setup and assets resolve):\n\n"
        f"   `{directory}`\n\n"
        "2. From a terminal, use this checkout's existing environment to register a named kernel "
        "and open Jupyter. Run the kernel registration once; it does not install lesson dependencies.\n\n"
        "```bash\n"
        f"cd {shlex.quote(str(root))}\n"
        f"{shlex.quote(str(python))} -m ipykernel install --user --name arena-extended "
        "--display-name 'ARENA Extended (study kernel)'\n"
        f"{shlex.quote(str(python))} -m jupyter lab\n"
        "```\n\n"
        f"3. In Jupyter's file browser, open `{directory.relative_to(root) / name}`. "
        "Choose **Kernel → Change Kernel → ARENA Extended (study kernel)**. In a new cell, "
        f"`import sys; print(sys.executable)` should show `{python}`. "
        "Follow the notebook's setup and hardware requirements, implement the exercises, then use "
        "**Restart Kernel** and **Run All** when you are ready to check your complete work."
    )


def render(st, root: Path = ROOT) -> None:
    rows = inventory(root)
    st.set_page_config(page_title="ARENA Extended — Study", layout="wide")
    st.title("ARENA Extended")
    st.markdown("Choose a lesson, make a prediction, then work through its exercises and controls. "
                "The reference notebooks contain public worked answers and saved outputs.")
    query = getattr(st, "query_params", {})
    study = study_route.load(root)
    if hasattr(st, "expander"):
        with st.expander("Start here", expanded=not query.get("section")):
            start_panel(st)
    else:
        start_panel(st)
    st.sidebar.markdown("## [Original ARENA](https://learn.arena.education/)\n\n"
        "[Fundamentals](https://learn.arena.education/chapter0_fundamentals/) · "
        "[Transformer Interpretability](https://learn.arena.education/chapter1_transformer_interp/) · "
        "[Reinforcement Learning](https://learn.arena.education/chapter2_rl/) · "
        "[LLM Evaluations](https://learn.arena.education/chapter3_llm_evals/) · "
        "[Alignment Science](https://learn.arena.education/chapter4_alignment_science/)\n\n"
        "The whole route, with days per stage: `mastery_hub/STUDY-GUIDE.md` in this checkout.")
    chapters = list(dict.fromkeys(row["chapter"] for row in rows))
    names = {row["chapter"]: row["chapter_title"] for row in rows}
    route_lesson, route_part, route_error = resolve_route(query, rows)
    if route_error:
        st.info(route_error)
        return
    state = getattr(st, "session_state", {})
    route = (route_lesson["number"], route_part) if route_lesson else None
    new_route = route is not None and route != state.get("reader_url_route")
    if new_route:
        state["reader_chapter"] = route_lesson["chapter"]
    chapter = st.sidebar.selectbox("Chapter", chapters, format_func=names.get, key="reader_chapter")
    chapter_rows = [row for row in rows if row["chapter"] == chapter]
    available = [row for row in chapter_rows if row["available"]]
    for row in chapter_rows:
        if not row["available"]:
            st.sidebar.info(f"[{row['number']}] {row['title']} — {row['status']}")
    if not available:
        st.info("These configured lessons are not available yet. Choose another chapter or the original ARENA course.")
        return
    lesson_key = f"reader_lesson_{chapter}"
    if new_route:
        state[lesson_key] = route_lesson
    lesson = st.sidebar.selectbox("Lesson", available, format_func=lambda row: f"[{row['number']}] {row['title']}", key=lesson_key)
    st.markdown(f"## [{lesson['number']}] {lesson['title']}")
    try:
        page_parts = parts(lesson["page"], with_media=True)
    except ValueError as exc:
        st.info(f"This lesson's page could not be read: {exc}. Please report section {lesson['number']} and this message.")
        return
    st.markdown("Work through the exercises notebook; the solutions notebook has worked answers and saved outputs.")
    for role, label in (("exercises", "Download exercises notebook"), ("reference", "Download solutions notebook")):
        path = lesson[role]
        st.download_button(label, path.read_bytes(), file_name=path.name, mime="application/x-ipynb+json")
    if hasattr(st, "expander"):
        with st.expander("Open your notebook in Jupyter", expanded=False):
            notebook_handoff(st, root, lesson)
    else:
        notebook_handoff(st, root, lesson)
    st.markdown("### Practice and reading")
    for label, path in supplements(lesson["number"], root):
        mime = "text/markdown" if path.suffix == ".md" else "application/x-ipynb+json"
        st.download_button(label, path.read_bytes(), file_name=path.name, mime=mime)
        if path.suffix == ".ipynb":
            st.markdown(f"Save `{path.name}` beside its source in `{path.parent.resolve()}` for relative setup and assets.")
    part_key = f"reader_part_{lesson['number']}"
    if new_route:
        state[part_key] = route_part
    part = st.sidebar.selectbox("Part", range(len(page_parts)),
                                format_func=lambda index: f"{index + 1}. {part_title(page_parts[index], index)}", key=part_key)
    # Updating the URL after selectors preserves manual changes on the next rerun.
    current_route = (lesson["number"], part)
    if hasattr(st, "query_params"):
        if route != current_route:
            st.query_params.update({"section": lesson["number"], "part": str(part + 1)})
    state["reader_url_route"] = current_route
    links = [route_link("Resume / bookmark this part", lesson["number"], part)]
    if part > 0:
        links.append(route_link("← Previous part", lesson["number"], part - 1))
    if part + 1 < len(page_parts):
        links.append(route_link("Next part →", lesson["number"], part + 1))
    stop = study.locate(lesson["number"], part, len(page_parts))
    if stop is not None and part == stop.reader_parts(len(page_parts))[0]:
        tldr = study_route.chapter_tldr(study, lesson["chapter"], lambda s: stop_link(study, s, rows))
        if tldr:
            label = f"{names[lesson['chapter']]}: before, during and after"
            container = st.expander(label, expanded=False) if hasattr(st, "expander") else nullcontext()
            with container:
                st.markdown(tldr.replace("## TLDR (if you're in a hurry!)", "", 1), unsafe_allow_html=True)
    st.markdown(f"Part **{part + 1} of {len(page_parts)}** · " + " · ".join(links), unsafe_allow_html=True)
    st.markdown(page_parts[part], unsafe_allow_html=True)
    if stop is not None and at_stop_end(stop, page_parts, part):
        next_step(st, study, rows, stop)
    route_sidebar(st, study, rows, stop)


if __name__ == "__main__":
    import streamlit as st

    render(st)
