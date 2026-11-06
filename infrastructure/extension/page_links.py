"""Render only audited local page destinations without mutating preserved sources.

The allowlist is keyed by exact repository-relative page path and original Markdown
href. Target identities were verified by the 2026-10-23 breadth audit; changed or
missing bytes require an explicit map review. Downloads retain raw notebook bytes,
including historical links and identities inside those notebooks.
"""

from __future__ import annotations

import base64
import hashlib
import html
import re
from pathlib import Path


_PAGE_LINKS = {
    "chapter0_fundamentals/instructions/pages/06_[0.6]_How_to_Know_When_an_Interpretability_Result_Is_Fake_Practice.md": {
        "expected_outputs/fake_interpretability_signature_result.svg": (
            "chapter0_fundamentals/exercises/part6_fake_interpretability_results_practice/media/fake_interpretability_signature_result.svg",
            "994cad5444f744cb8ef05ab14bb17075bf2273f12e1c15a60d3d6a6c01f29d90",
            "source document",
        ),
    },
    "chapter5_modern_architectures/instructions/pages/05_[5.5]_State_Tracking_in_SSMs_and_Transformers.md": {
        "../part4_mamba_state_tracking/5.4_Mamba_State_Tracking_exercises.ipynb": (
            "chapter5_modern_architectures/exercises/part4_mamba_state_tracking/5.4_Mamba_State_Tracking_exercises.ipynb",
            "434a28a646ea0c4854f7ee9dfca1efcf2f8041c732548007c8a58617e2ca64d2",
            "historical raw notebook",
        ),
        "../part3_mamba_from_scratch/5.3_Mamba_from_Scratch_exercises.ipynb": (
            "chapter5_modern_architectures/exercises/part3_mamba_from_scratch/5.3_Mamba_from_Scratch_exercises.ipynb",
            "27026a893d5cb7138c83f0a943d3221cfa60d84dfca520f4ea35418d5aae7b81",
            "historical raw notebook",
        ),
    },
    "chapter13_image_generation_interpretability/instructions/pages/02_[13.2]_Denoising-Time_Causal_Patching.md": {
        "13.2_Denoising-Time_Causal_Patching_exercises.ipynb": (
            "chapter13_image_generation_interpretability/exercises/part2_denoising_time_causal_patching/13.2_Denoising-Time_Causal_Patching_exercises.ipynb",
            "55ae5246aae1603bd894f45aad45a065eb70f021fea92816723e6c2113a7b5b0",
            "historical raw notebook",
        ),
        "13.2_Denoising-Time_Causal_Patching_solutions.ipynb": (
            "chapter13_image_generation_interpretability/exercises/part2_denoising_time_causal_patching/13.2_Denoising-Time_Causal_Patching_solutions.ipynb",
            "a1b585423650db88b140b9ac23b05a3e266da57f0186c62554056a8419c5d897",
            "historical raw notebook",
        ),
    },
    "chapter14_jepa_world_models/instructions/pages/02_[14.2]_V-JEPA_and_Video_World_Models.md": {
        "14.1_JEPA_and_World_Model_Controls_exercises.ipynb": (
            "chapter14_jepa_world_models/exercises/part1_jepa_world_model_controls/14.1_JEPA_and_World_Model_Controls_exercises.ipynb",
            "d56188485421e54d6c144305134a6109225a2d83139f4308151d345252153b68",
            "historical raw notebook",
        ),
        "14.1_JEPA_and_World_Model_Controls_solutions.ipynb": (
            "chapter14_jepa_world_models/exercises/part1_jepa_world_model_controls/14.1_JEPA_and_World_Model_Controls_solutions.ipynb",
            "3636330f1f1bef5b02cff7d20a57c076ad46970c308287700ef2d02e07039e2e",
            "historical raw notebook",
        ),
    },
    "chapter15_peft_misalignment/instructions/pages/01_[15.1]_LoRA_and_DoRA_from_Scratch.md": {
        "media/lora_dora_adapter_controls_validation_loop.svg": (
            "chapter15_peft_misalignment/exercises/part1_lora_dora/media/lora_dora_adapter_controls_validation_loop.svg",
            "08091bc3d11c627a18819ae055a8c13b3d1874567aae96573bd33c936f3f8911",
            "source document",
        ),
        "media/lora_dora_adapter_controls_signature_panel.png": (
            "chapter15_peft_misalignment/exercises/part1_lora_dora/media/lora_dora_adapter_controls_signature_panel.png",
            "a101afe9432afd789e7bf4668bf8f11c05d84add7c54ed75db76970f7596fd9d",
            "source document",
        ),
        "media/lora_dora_adapter_controls_behavior_table.png": (
            "chapter15_peft_misalignment/exercises/part1_lora_dora/media/lora_dora_adapter_controls_behavior_table.png",
            "782402dbc7de25d708014a4cb8670d2de9aa3d27ad5894c07f280401ebd71fa4",
            "source document",
        ),
    },
    "chapter16_shapley_attribution_baselines/instructions/pages/02_[16.2]_KernelSHAP_and_PartitionSHAP.md": {
        "../part1_exact_shapley_ground_truth_games/16.1_Exact_Shapley_on_Ground_Truth_Games_exercises.ipynb": (
            "chapter16_shapley_attribution_baselines/exercises/part1_exact_shapley_ground_truth_games/16.1_Exact_Shapley_on_Ground_Truth_Games_exercises.ipynb",
            "425e9963d352048a6f9370feb5c9fa255dd72f7ca44f70d00f60980d74e4eaa7",
            "historical raw notebook",
        ),
    },
    "chapter16_shapley_attribution_baselines/instructions/pages/04_[16.4]_SAGE_Global_Importance.md": {
        "../part1_exact_shapley_ground_truth_games/16.1_Exact_Shapley_on_Ground_Truth_Games_exercises.ipynb": (
            "chapter16_shapley_attribution_baselines/exercises/part1_exact_shapley_ground_truth_games/16.1_Exact_Shapley_on_Ground_Truth_Games_exercises.ipynb",
            "425e9963d352048a6f9370feb5c9fa255dd72f7ca44f70d00f60980d74e4eaa7",
            "historical raw notebook",
        ),
    },
    "chapter16_shapley_attribution_baselines/instructions/pages/05_[16.5]_Shapley_Interactions.md": {
        "protocol.md": (
            "chapter16_shapley_attribution_baselines/exercises/part16_shapley_taylor_faith_shap/protocol.md",
            "c0d738cf953743f17b74ca5ced74c15330b8b3be7d7d128563ba36096986ee37",
            "source document",
        ),
    },
    "chapter16_shapley_attribution_baselines/instructions/pages/10_[16.10]_Influence_Functions_and_TracIn.md": {
        "../part9_data_shapley/16.9_Data_Shapley_in_One_Training_Run_exercises.ipynb": (
            "chapter16_shapley_attribution_baselines/exercises/part9_data_shapley/16.9_Data_Shapley_in_One_Training_Run_exercises.ipynb",
            "fd77f88df783a1757aa158a528da40ca1a639992c2282b4649c463c441a28a82",
            "historical raw notebook",
        ),
    },
    "chapter16_shapley_attribution_baselines/instructions/pages/11_[16.11]_Datamodels_and_Dataset_Cartography.md": {
        "../part9_influence_and_tracin/README.md": (
            "chapter16_shapley_attribution_baselines/exercises/part9_influence_and_tracin/README.md",
            "10c7e101e78cd62a19e94cb0e06e1b7e97a6e4f53c26ce3e21be92ad32cd74df",
            "source document",
        ),
        "../part10_datamodels/README.md": (
            "chapter16_shapley_attribution_baselines/exercises/part10_datamodels/README.md",
            "457adb74910cbe3a745129313626f8a6c410afc0fcffe9d1ca2ee0c95797a2e0",
            "source document",
        ),
    },
    "chapter7_activation_to_language/instructions/pages/01_[7.1]_Logit_Lens_Tuned_Lens_and_Patchscopes.md": {
        "../../../docs/mastery/pilot_reader_notes_20261023.md": (
            "docs/mastery/pilot_reader_notes_20261023.md",
            "68aa08ac199d56e712c02eb74e50cc7e275e816db0f8750b3390a1ef35413de1",
            "reading notes",
        ),
    },
    "chapter12_vlm_interpretability/instructions/pages/03_[12.3]_Mini_VLM_from_Scratch.md": {
        "../../../docs/mastery/pilot_reader_notes_20261023.md": (
            "docs/mastery/pilot_reader_notes_20261023.md",
            "68aa08ac199d56e712c02eb74e50cc7e275e816db0f8750b3390a1ef35413de1",
            "reading notes",
        ),
        "../../../mastery_hub/practice_addons/mini_vlm_checked_recovery/exercises.ipynb": (
            "mastery_hub/practice_addons/mini_vlm_checked_recovery/exercises.ipynb",
            "fa34480ae62f6e0d4201b151beed72c2ab1e07880accaafb620cd93376aba163",
            "checked recovery practice notebook",
        ),
    },
}
_LINK = re.compile(r"(?P<image>!)?\[(?P<label>[^\[\]]*)\]\((?P<target>[^)\s]+)\)")
_INLINE_CODE = re.compile(r"(`+).*?\1", re.S)
_FENCE = re.compile(r" {0,3}(?P<marker>`{3,}|~{3,})(?P<tail>.*)$")


def _verified_payload(page: Path, relative: str, expected_sha256: str) -> tuple[bytes, Path]:
    root = page.parents[3]
    target = root / relative
    # The finite table cannot silently start serving a symlink or another tree.
    if not target.is_relative_to(root):
        raise ValueError("Audited link target escapes the repository")
    current = target
    while current != root:
        if current.is_symlink():
            raise ValueError("Audited link target contains a symlink")
        current = current.parent
    if not target.resolve().is_relative_to(root):
        raise ValueError("Audited link target escapes the repository")
    if target.suffix in {".png", ".svg"} and (
        not target.is_relative_to(page.parents[2] / "exercises") or target.parent.name != "media"
    ):
        raise ValueError("Teaching image is outside this chapter's exercise media")
    payload = target.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"Audited link target hash changed: {relative}")
    return payload, target


def rewrite_page_links(text: str, markdown_path: Path) -> str:
    """Project exact allowlisted links to verified downloads or inline teaching figures.

    Unknown pages/destinations, fenced code, inline code, and plain path strings
    pass through unchanged. This function reads targets and returns rendered text;
    it never writes a page, notebook, source document, or image.
    """
    page = Path(markdown_path).resolve()
    routes = _PAGE_LINKS.get("/".join(page.parts[-4:]))
    if not routes:
        return text
    verified = {}

    def replace(match: re.Match[str]) -> str:
        route = routes.get(match["target"])
        if route is None:
            return match[0]
        relative, expected_sha256, kind = route
        if relative not in verified:
            verified[relative] = _verified_payload(page, relative, expected_sha256)
        payload, target = verified[relative]
        suffix = target.suffix
        mime = {
            ".png": "image/png",
            ".svg": "image/svg+xml",
            ".ipynb": "application/x-ipynb+json",
            ".md": "text/markdown",
        }[suffix]
        uri = f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"
        if suffix in {".png", ".svg"}:
            if not match["image"]:
                return match[0]
            return f"![{match['label']}]({uri})"
        if match["image"]:
            return match[0]
        # A wrapped Markdown blockquote label renders as one visible phrase.
        label = re.sub(r"\r?\n[ \t]*(?:>[ \t]*)*", " ", match["label"])
        label = html.escape(label, quote=True)
        filename = html.escape(target.name, quote=True)
        return f'<a download="{filename}" href="{uri}">{label} ({kind} download)</a>'

    def plain_text(chunk: str) -> str:
        pieces = []
        offset = 0
        for code in _INLINE_CODE.finditer(chunk):
            pieces.extend((_LINK.sub(replace, chunk[offset:code.start()]), code[0]))
            offset = code.end()
        pieces.append(_LINK.sub(replace, chunk[offset:]))
        return "".join(pieces)

    rendered = []
    prose = []
    fence = None
    for line in text.splitlines(keepends=True):
        marker = _FENCE.match(line.rstrip("\r\n"))
        if fence is not None:
            rendered.append(line)
            if (
                marker
                and marker["marker"][0] == fence[0]
                and len(marker["marker"]) >= fence[1]
                and not marker["tail"].strip()
            ):
                fence = None
        elif marker:
            rendered.append(plain_text("".join(prose)))
            prose.clear()
            fence = (marker["marker"][0], len(marker["marker"]))
            rendered.append(line)
        else:
            prose.append(line)
    rendered.append(plain_text("".join(prose)))
    return "".join(rendered)
