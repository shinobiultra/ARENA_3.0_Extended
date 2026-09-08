from arena_ext.gated_artifacts import (
    GEMMA3_1B_IT_BASE,
    HFGatedArtifactSpec,
    hf_cache_repo_dir,
    hf_model_artifact_access_report,
)


def test_hf_cache_repo_dir_uses_hub_repo_encoding(tmp_path):
    assert hf_cache_repo_dir("google/gemma-3-1b-it", cache_root=tmp_path) == (
        tmp_path / "models--google--gemma-3-1b-it"
    )


def test_gemma3_required_artifact_uses_pinned_revision():
    assert GEMMA3_1B_IT_BASE.revision == "dcc83ea841ab6100d6b47a070329e1ba4cf78752"


def test_hf_model_artifact_access_report_detects_missing_local_required_files(tmp_path):
    spec = HFGatedArtifactSpec(
        repo_id="google/test-gated-model",
        revision="abc123",
        required_patterns=("config.json", "tokenizer.json", "model.safetensors"),
        download_patterns=("config.json", "tokenizer.json", "model.safetensors"),
        purpose="unit test",
    )

    report = hf_model_artifact_access_report(
        spec,
        allow_network=False,
        cache_root=tmp_path,
    )

    assert report["local_ready_for_direct_loading"] is False
    assert report["ready_for_direct_loading"] is False
    assert report["missing_local_patterns"] == [
        "config.json",
        "tokenizer.json",
        "model.safetensors",
    ]


def test_hf_model_artifact_access_report_accepts_complete_local_snapshot(tmp_path):
    spec = HFGatedArtifactSpec(
        repo_id="google/test-gated-model",
        revision="abc123",
        required_patterns=("config.json", "tokenizer.json", "model.safetensors"),
        download_patterns=("config.json", "tokenizer.json", "model.safetensors"),
        purpose="unit test",
    )
    snapshot = (
        hf_cache_repo_dir(spec.repo_id, cache_root=tmp_path)
        / "snapshots"
        / spec.revision
    )
    snapshot.mkdir(parents=True)
    for name in spec.required_patterns:
        (snapshot / name).write_text("{}")

    report = hf_model_artifact_access_report(
        spec,
        allow_network=False,
        cache_root=tmp_path,
    )

    assert report["local_non_ref_file_count"] == 3
    assert report["missing_local_patterns"] == []
    assert report["local_ready_for_direct_loading"] is True
    assert report["ready_for_direct_loading"] is True
