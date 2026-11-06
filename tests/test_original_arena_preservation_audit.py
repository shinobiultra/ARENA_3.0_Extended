"""The chapters 0-4 preservation audit: originals byte-identical, config append-only."""

from scripts import audit_original_arena_preservation as audit


def test_original_paths_cover_chapters_and_their_masters():
    paths = audit.original_paths(audit.original_base_revision())
    assert any(path.startswith("chapter1_transformer_interp/exercises/") for path in paths)
    assert any(path.startswith("infrastructure/chapters/chapter0_fundamentals/master_0_") for path in paths)
    assert not any(path.startswith("chapter5_") for path in paths)


def test_edited_original_file_is_a_blocker(tmp_path, monkeypatch):
    revision = audit.original_base_revision()
    relative = "chapter0_fundamentals/exercises/part0_prereqs/solutions.py"
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    base = audit._git_bytes("show", f"{revision}:{relative}")
    target.write_bytes(base + b"\n# edited\n")
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "_git_bytes", lambda *args: base)
    monkeypatch.setattr(audit, "original_paths", lambda _revision: [relative, "chapter2_rl/missing.py"])
    assert audit.original_chapter_file_blockers(revision) == [
        f"{relative}: differs from pinned original ARENA base",
        "chapter2_rl/missing.py: original file deleted",
    ]


def test_current_worktree_preserves_original_arena_surface():
    assert audit.original_preservation_blockers() == []


def test_split_config_is_current_and_generated_equivalent():
    assert audit.split_config_blockers() == []
