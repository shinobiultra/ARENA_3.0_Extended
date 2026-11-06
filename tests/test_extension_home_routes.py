"""Chapter-home links must follow configured pages, including missing-page states."""

import pytest

import st_dependencies
from infrastructure.extension.home import render_chapter_home


@pytest.fixture
def chapter_home(tmp_path, monkeypatch):
    """Render real section metadata against temporary pages, without starting Streamlit."""
    chapter = tmp_path / "chapter5_fixture"
    pages = chapter / "instructions" / "pages"
    pages.mkdir(parents=True)
    metadata = {"title": "Chapter 5: Fixture", "header_image": "header.png", "sections": []}
    monkeypatch.setattr(st_dependencies, "_load_config", lambda: {"chapters": {chapter.name: metadata}})
    monkeypatch.setattr(st_dependencies, "styling", lambda title: None)
    rendered = []
    links = []
    monkeypatch.setattr(st_dependencies.st, "markdown", lambda body, **kwargs: rendered.append(body))
    monkeypatch.setattr(st_dependencies.st, "page_link", lambda page, *, label: links.append((page, label)))
    monkeypatch.syspath_prepend(str(tmp_path))

    def render(sections, filenames):
        metadata["sections"] = sections
        for filename in filenames:
            (pages / filename).write_text("# Fixture page\n", encoding="utf-8")
        render_chapter_home(str(chapter / "instructions" / "Home.py"), "Fixture introduction")
        return {"messages": rendered, "links": links}

    return render, pages


def section(number, title, **metadata):
    return {"number": number, "title": title, "streamlit_description": f"About {title}", **metadata}


def test_home_delegates_registered_python_page_navigation_to_streamlit(chapter_home, monkeypatch):
    render, pages = chapter_home
    calls = []
    monkeypatch.setattr(
        st_dependencies.st, "page_link", lambda page, *, label: calls.append((page, label))
    )

    render(
        [section("5.2", "Mixture of Experts", page_file="02_[5.2]_Mixture_of_Experts.md")],
        ["02_[5.2]_Mixture_of_Experts.md", "02_[5.2]_Mixture_of_Experts.py"],
    )

    assert calls == [(pages / "02_[5.2]_Mixture_of_Experts.py", "[5.2] Mixture of Experts")]


def test_home_uses_configured_page_when_legacy_page_has_same_section_number(chapter_home):
    render, pages = chapter_home

    rows = render(
        [section("5.2", "Mixture of Experts", page_file="02_[5.2]_Mixture_of_Experts.md")],
        [
            "01_[5.2]_Gemma.md", "01_[5.2]_Gemma.py",
            "02_[5.2]_Mixture_of_Experts.md", "02_[5.2]_Mixture_of_Experts.py",
        ],
    )

    assert rows["links"] == [
        (pages / "02_[5.2]_Mixture_of_Experts.py", "[5.2] Mixture of Experts")
    ]
    assert "About Mixture of Experts" in rows["messages"]


@pytest.mark.parametrize("page_file", ["02_[5.2]_Mixture_of_Experts.md", "", None])
def test_home_marks_missing_configured_page_unavailable_instead_of_linking_legacy_page(chapter_home, page_file):
    render, _ = chapter_home
    metadata = {} if page_file is None else {"page_file": page_file}

    rows = render([section("5.2", "Mixture of Experts", **metadata)], ["01_[5.2]_Gemma.md"])

    unavailable = [text for text in rows["messages"] if "unavailable" in text]
    assert len(unavailable) == 1
    assert "[5.2] Mixture of Experts" in unavailable[0]
    assert "pages/" not in unavailable[0]
    assert rows["links"] == []


def test_home_requires_configured_page_to_be_a_file(chapter_home):
    render, pages = chapter_home
    (pages / "02_[5.2]_Mixture_of_Experts.md").mkdir()

    rows = render(
        [section("5.2", "Mixture of Experts", page_file="02_[5.2]_Mixture_of_Experts.md")],
        [],
    )

    assert any("unavailable" in text for text in rows["messages"])
    assert rows["links"] == []


def test_home_marks_markdown_without_registered_python_page_unavailable(chapter_home):
    render, _ = chapter_home

    rows = render(
        [section("5.2", "Mixture of Experts", page_file="02_[5.2]_Mixture_of_Experts.md")],
        ["02_[5.2]_Mixture_of_Experts.md"],
    )

    assert any("unavailable" in text for text in rows["messages"])
    assert rows["links"] == []


def test_home_preserves_display_order_and_excludes_group_headers(chapter_home):
    render, pages = chapter_home

    rows = render(
        [
            section("5.3", "Second", page_file="03_[5.3]_Second.md"),
            section("", "Group", is_group=True),
            section("5.1", "First", page_file="01_[5.1]_First.md"),
        ],
        ["01_[5.1]_First.md", "01_[5.1]_First.py", "03_[5.3]_Second.md", "03_[5.3]_Second.py"],
    )

    assert rows["links"] == [
        (pages / "03_[5.3]_Second.py", "[5.3] Second"),
        (pages / "01_[5.1]_First.py", "[5.1] First"),
    ]
    assert [text for text in rows["messages"] if text.startswith("About ")] == [
        "About Second", "About First"
    ]
