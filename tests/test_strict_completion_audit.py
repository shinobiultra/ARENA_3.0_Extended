from scripts.audit_extension_completion_strict import required_artifact_blocker_for_row


def test_required_artifact_blocker_flags_gated_pending_row():
    blocker = required_artifact_blocker_for_row(
        {
            "name": "EmbeddingGemma 300M",
            "repo_or_source_id": "google/embeddinggemma-300m",
            "local_status": "REQUIRED_GATED_PENDING",
            "revision": "abc123",
        }
    )

    assert blocker == (
        "EmbeddingGemma 300M: required artifact is gated/pending "
        "(google/embeddinggemma-300m)"
    )


def test_required_artifact_blocker_flags_unresolved_required_pin():
    blocker = required_artifact_blocker_for_row(
        {
            "name": "Stable Diffusion v1.5",
            "repo_or_source_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "local_status": "REQUIRED",
            "revision": "pin_before_use",
        }
    )

    assert blocker == (
        "Stable Diffusion v1.5: required artifact revision is unresolved "
        "(pin_before_use; stable-diffusion-v1-5/stable-diffusion-v1-5)"
    )


def test_required_artifact_blocker_flags_pending_verified_report():
    blocker = required_artifact_blocker_for_row(
        {
            "name": "Othello-GPT",
            "repo_or_source_id": "NeelNanda/Othello-GPT-Transformer-Lens",
            "local_status": "REQUIRED_PENDING_VERIFIED_REPORT",
            "revision": "905ca1a68b9f7dff77adc56af1962e5f6fcac274",
        }
    )

    assert blocker == (
        "Othello-GPT: required artifact lacks a modern verified report "
        "(NeelNanda/Othello-GPT-Transformer-Lens)"
    )


def test_required_artifact_blocker_allows_optional_future_pin():
    assert (
        required_artifact_blocker_for_row(
            {
                "name": "GPT-2 small",
                "repo_or_source_id": "gpt2-small",
                "local_status": "OPTIONAL_FUTURE",
                "revision": "pin_before_use",
            }
        )
        is None
    )


def test_required_artifact_blocker_allows_pinned_required_row():
    assert (
        required_artifact_blocker_for_row(
            {
                "name": "Pythia 70M deduped",
                "repo_or_source_id": "EleutherAI/pythia-70m-deduped",
                "local_status": "REQUIRED",
                "revision": "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
            }
        )
        is None
    )
