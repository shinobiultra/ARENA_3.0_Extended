"""Read current ARENA extension lessons without loading chat or teaching code."""
from __future__ import annotations

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


_SUPPLEMENTS = {'*': (('Download study guide', 'mastery_hub/STUDY-GUIDE.md'),),
 '6.5': (('Download crosscoder metric repair Practice', 'mastery_hub/practice_addons/crosscoder_metric_repair/exercises.ipynb'),
         ('Download crosscoder metric repair reference', 'mastery_hub/practice_addons/crosscoder_metric_repair/solutions.ipynb'),
         ('Download crosscoder metric repair instructions', 'mastery_hub/practice_addons/crosscoder_metric_repair/README.md')),
 '7.1': (('Download pilot reading notes', 'docs/mastery/pilot_reader_notes_20261023.md'),
         ('Download 7.1 guided public Practice', 'mastery_hub/units/pilot_practice/7_1_exercises.ipynb')),
 '9.1': (('Download 9.1 guided public Practice', 'mastery_hub/units/pilot_practice/9_1_exercises.ipynb'),),
 '12.3': (('Download pilot reading notes', 'docs/mastery/pilot_reader_notes_20261023.md'),
          ('Download checked recovery Practice', 'mastery_hub/practice_addons/mini_vlm_checked_recovery/exercises.ipynb'),
          ('Download Mini VLM memory handoff', 'mastery_hub/practice_addons/mini_vlm_memory_handoff/README.md')),
 '8.4': (('Download 8.4 own-method guided public Practice', 'mastery_hub/units/acdc_own_method/exercises.ipynb'),
         ('Download 8.4 Practice instructions', 'mastery_hub/units/acdc_own_method/README.md')),
 '0.6': (('Download historical 0.6 exercises',
          'chapter0_fundamentals/exercises/part6_fake_interpretability_results/0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_exercises.ipynb'),
         ('Download historical 0.6 reference (saved outputs)',
          'chapter0_fundamentals/exercises/part6_fake_interpretability_results/0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_solutions.ipynb')),
 '1.6': (('Download historical 1.6 exercises',
          'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure/1.6_Local_Frontier_ML_Infrastructure_exercises.ipynb'),
         ('Download historical 1.6 reference (saved outputs)',
          'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure/1.6_Local_Frontier_ML_Infrastructure_solutions.ipynb')),
 '16.3': (('Download historical TreeSHAP exercises (saved outputs)',
           'chapter16_shapley_attribution_baselines/exercises/part13_treeshap/16.13_TreeSHAP_from_First_Principles_exercises.ipynb'),
          ('Download historical TreeSHAP reference (saved outputs)',
           'chapter16_shapley_attribution_baselines/exercises/part13_treeshap/16.13_TreeSHAP_from_First_Principles_solutions.ipynb'),
          ('Download historical DeepSHAP exercises (saved outputs)',
           'chapter16_shapley_attribution_baselines/exercises/part14_deepshap/16.14_DeepSHAP_When_Does_Fast_Attribution_Match_Exact_SHAP_exercises.ipynb'),
          ('Download historical DeepSHAP reference (saved outputs)',
           'chapter16_shapley_attribution_baselines/exercises/part14_deepshap/16.14_DeepSHAP_When_Does_Fast_Attribution_Match_Exact_SHAP_solutions.ipynb'),
          ('Download historical Expected Gradients exercises (saved outputs)',
           'chapter16_shapley_attribution_baselines/exercises/part15_gradientshap/16.15_Expected_Gradients_Which_Expectation_Does_GradientSHAP_Estimate_exercises.ipynb'),
          ('Download historical Expected Gradients reference (saved outputs)',
           'chapter16_shapley_attribution_baselines/exercises/part15_gradientshap/16.15_Expected_Gradients_Which_Expectation_Does_GradientSHAP_Estimate_solutions.ipynb'))}


def supplements(number: str, root: Path) -> list[tuple[str, Path]]:
    """Offer only these named public resources; section text is never a path."""
    return [(label, Path(root) / path) for label, path in _SUPPLEMENTS["*"] + _SUPPLEMENTS.get(number, ())]


def inventory(root: Path, *, configuration_bytes: dict[str, bytes] | None = None) -> list[dict]:
    """Resolve only configured lesson identities; never glob historical neighbors."""
    root = Path(root)
    if configuration_bytes is None:
        from infrastructure.extension.blank_practice import _capture_configuration
        configuration_bytes = _capture_configuration(root)
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
        f"- **Small CPU route:** {route_link('16.1 Exact Shapley', '16.1')} → "
        f"{route_link('16.2 KernelSHAP / PartitionSHAP', '16.2')}, using the finite-game exercises.\n"
        f"- **Read internal answers:** {route_link('7.1 Logit Lens / Tuned Lens / Patchscopes', '7.1')}, after original ARENA 1.2.\n"
        f"- **Match images and text:** {route_link('12.1 CLIP / SigLIP', '12.1')} → "
        f"{route_link('12.3 Mini VLM', '12.3')}.\n"
        f"- **Intervene on a direction:** {route_link('9.1 Refusal Directions', '9.1')}, after residual-stream hooks.\n"
        "- **Build foundations:** [Original ARENA course](https://learn.arena.education/).\n\n"
        "Choose one route; follow the lesson's setup before running its cells. "
        "Saved reference outputs are separate from your own execution.",
        unsafe_allow_html=True,
    )


def notebook_handoff(st, root: Path, lesson: dict, blank_name: str) -> None:
    directory = lesson["exercises"].parent.resolve()
    root = root.resolve()
    python = Path(sys.executable).absolute()
    st.markdown(
        "### Open your Practice notebook locally\n"
        f"1. Save **{blank_name}** in this exact directory (beside the configured exercise notebook):\n\n"
        f"   `{directory}`\n\n"
        "2. From a terminal, use this checkout's existing environment to register a named kernel "
        "and open Jupyter. Run the kernel registration once; it does not install lesson dependencies.\n\n"
        "```bash\n"
        f"cd {shlex.quote(str(root))}\n"
        f"{shlex.quote(str(python))} -m ipykernel install --user --name arena-extended "
        "--display-name 'ARENA Extended (study kernel)'\n"
        f"{shlex.quote(str(python))} -m jupyter lab\n"
        "```\n\n"
        f"3. In Jupyter's file browser, open `{directory.relative_to(root) / blank_name}`. "
        "Choose **Kernel → Change Kernel → ARENA Extended (study kernel)**. In a new cell, "
        f"`import sys; print(sys.executable)` should show `{python}`. "
        "Follow the notebook's setup and hardware requirements, implement the exercises, then use "
        "**Restart Kernel** and **Run All** when you are ready to check your complete work.\n\n"
        "Keep the published exercises and worked reference as references. Blank copies are **PUBLIC Practice**: "
        "help, worked answers and expected answers remain public; clearing outputs does not make an unseen assessment."
    )


def blank_config_identity(root: Path) -> dict[str, str]:
    from infrastructure.extension.blank_practice import config_identity
    return config_identity(root)


def blank_practice_download(root: Path, section: str) -> tuple[bytes, str]:
    from infrastructure.extension.blank_practice import download
    return download(root, section, inventory)


def restore_blank_practice(projected: bytes, source: bytes, relative: str, authority: dict) -> dict:
    from infrastructure.extension.blank_practice import restore_download
    return restore_download(projected, source, relative, authority)


def render(st, root: Path = ROOT) -> None:
    rows = inventory(root)
    st.set_page_config(page_title="ARENA Extended — Study", layout="wide")
    st.title("ARENA Extended")
    st.markdown("Choose a lesson, make a prediction, then work through its exercises and controls. "
                "The reference notebooks contain public worked answers and saved outputs.")
    query = getattr(st, "query_params", {})
    if hasattr(st, "expander"):
        with st.expander("Choose a starting route", expanded=not query.get("section")):
            start_panel(st)
    else:
        start_panel(st)
    st.sidebar.markdown("## [Original ARENA](https://learn.arena.education/)\n\n"
        "[Fundamentals](https://learn.arena.education/chapter0_fundamentals/) · "
        "[Transformer Interpretability](https://learn.arena.education/chapter1_transformer_interp/) · "
        "[Reinforcement Learning](https://learn.arena.education/chapter2_rl/) · "
        "[LLM Evaluations](https://learn.arena.education/chapter3_llm_evals/) · "
        "[Alignment Science](https://learn.arena.education/chapter4_alignment_science/)\n\n"
        "Full study routes: `mastery_hub/STUDY-GUIDE.md` in this checkout.")
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
    if lesson["number"] == "12.3":
        st.markdown(
            "**Before the optional Step 5 Qwen bridge:** download the **Mini VLM memory handoff** "
            "below and follow its operational steps. The historical notebook keeps earlier models "
            "resident; use the handoff before loading the optional bridge. The original notebooks "
            "and their saved results remain unchanged."
        )
    try:
        blank_bytes, blank_name = blank_practice_download(root, lesson["number"])
        study_name = blank_name
        page_parts = parts(lesson["page"], with_media=True)
    except ValueError as exc:
        st.info(f"This lesson's study copy could not be prepared: {exc}. Please report section {lesson['number']} and this message.")
        return
    st.markdown("Start with blank Practice. The exercises and worked reference below include their saved outputs.")
    st.download_button("Download blank Practice exercises", blank_bytes,
                       file_name=blank_name, mime="application/x-ipynb+json")
    st.markdown("Blank copies are PUBLIC Practice: help, worked answers and expected answers remain public. "
                "Save beside the original exercise notebook for relative setup and assets. "
                "This is not blind evaluation or fresh execution evidence.")
    if hasattr(st, "expander"):
        with st.expander("Open your notebook in Jupyter", expanded=False):
            notebook_handoff(st, root, lesson, study_name)
    else:
        notebook_handoff(st, root, lesson, study_name)
    for role, label in (("exercises", "Download exercises"), ("reference", "Download worked reference")):
        path = lesson[role]
        st.download_button(label, path.read_bytes(), file_name=path.name, mime="application/x-ipynb+json")
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
    st.markdown(f"Part **{part + 1} of {len(page_parts)}** · " + " · ".join(links), unsafe_allow_html=True)
    st.markdown(page_parts[part], unsafe_allow_html=True)


if __name__ == "__main__":
    import streamlit as st

    render(st)
