"""Links to this repository from learner-facing documents resolve (Spec #39, P9; #312)."""

from pathlib import Path

from scripts import check_repo_links as links

REPO = "https://github.com/shinobiultra/ARENA_3.0_Extended"


def _problems(tmp_path: Path, text: str, *, learner: bool) -> list[str]:
    (tmp_path / "chapter5" / "exercises").mkdir(parents=True, exist_ok=True)
    (tmp_path / "chapter5" / "exercises" / "real.ipynb").write_text("{}")
    page = tmp_path / "chapter5" / "pages" / "page.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(text)
    return links.link_problems(page, root=tmp_path, learner=learner)


def test_valid_links_pass(tmp_path):
    text = "\n".join(
        [
            "[relative](../exercises/real.ipynb)",
            '<img src="../exercises/real.ipynb">',
            "[anchor](#setup)",
            f"[blob]({REPO}/blob/main/chapter5/exercises/real.ipynb)",
            f"[release]({REPO}/releases/latest)",
            f"[repo]({REPO})",
            "[elsewhere](https://github.com/ARENA-education/ARENA_materials/tree/some-branch/x)",
        ]
    )
    assert _problems(tmp_path, text, learner=True) == []


def test_broken_relative_and_blob_targets_are_caught(tmp_path):
    text = f"[gone](../exercises/missing.ipynb)\n[gone too]({REPO}/blob/main/chapter5/missing.py)"
    problems = _problems(tmp_path, text, learner=False)
    assert len(problems) == 2
    assert all("does not exist" in p for p in problems)


def test_branch_links_other_than_main_are_caught(tmp_path):
    text = "\n".join(
        [
            f"[branch]({REPO}/blob/integration/extension/chapter5/exercises/real.ipynb)",
            f"[tree]({REPO}/tree/batch/4-docs/chapter5)",
            "[raw](https://raw.githubusercontent.com/shinobiultra/ARENA_3.0_Extended/codex/x/chapter5/a.py)",
            "[colab](https://colab.research.google.com/github/shinobiultra/ARENA_3.0_Extended/blob/old/a.ipynb)",
        ]
    )
    problems = _problems(tmp_path, text, learner=False)
    assert len(problems) == 4
    assert all("not main" in p or "not a relative" in p for p in problems)


def test_issue_links_are_rejected_only_on_learner_pages(tmp_path):
    text = f"[issue]({REPO}/issues/39)\n[pr]({REPO}/pull/2)"
    assert len(_problems(tmp_path, text, learner=True)) == 2
    assert _problems(tmp_path, text, learner=False) == []


def test_covered_files_include_every_kind():
    covered = {
        path.relative_to(links.ROOT).as_posix(): learner for path, learner in links.covered_files()
    }
    assert covered["README.md"] is False
    assert covered["ALCHEMY.md"] is False
    assert covered["mastery_hub/STUDY-GUIDE.md"] is False
    assert covered["chapter8_automated_circuits/instructions/Home.py"] is True
    assert (
        covered[
            "chapter8_automated_circuits/instructions/pages/01_[8.1]_Activation_Patching_Refresher.md"
        ]
        is True
    )
    assert not any(p.startswith("chapter2_rl/") for p in covered)


def test_current_documents_have_no_broken_repo_links():
    problems = [
        p
        for path, learner in links.covered_files()
        for p in links.link_problems(path, learner=learner)
    ]
    assert problems == []


def test_code_is_not_a_link(tmp_path):
    text = "Use `resid[:,\ntarget_pos](batch, d)` here.\n\n```python\nx = model[\"norm\"](x)\n```\n"
    assert _problems(tmp_path, text, learner=True) == []
