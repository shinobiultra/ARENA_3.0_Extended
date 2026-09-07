"""Build the paired ARENA learner surfaces for section 14.2."""

from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE_PATH = HERE / "solutions.py"
EXERCISE_PATH = HERE / "14.2_I-JEPA_from_Scratch_exercises.ipynb"
SOLUTION_PATH = HERE / "14.2_I-JEPA_from_Scratch_solutions.ipynb"
PAGE_PATH = ROOT / "chapter14_jepa_world_models/instructions/pages/02_[14.2]_I-JEPA_from_Scratch.md"


CELL_COUNTER = 0


def _cell(cell_type: str, source: str, **extra: object) -> dict[str, object]:
    global CELL_COUNTER
    CELL_COUNTER += 1
    cell: dict[str, object] = {
        "id": f"arena-14-2-{CELL_COUNTER:03d}",
        "cell_type": cell_type,
        "metadata": extra.pop("metadata", {}),
        "source": textwrap.dedent(source).strip() + "\n",
    }
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    cell.update(extra)
    return cell


def md(source: str) -> dict[str, object]:
    return _cell("markdown", source)


def code(source: str, *, tags: list[str] | None = None) -> dict[str, object]:
    metadata = {"tags": tags} if tags else {}
    return _cell("code", source, metadata=metadata)


SOURCE = SOURCE_PATH.read_text(encoding="utf-8")
SOURCE_LINES = SOURCE.splitlines()
TREE = ast.parse(SOURCE)


def implementation(*names: str) -> str:
    wanted = set(names)
    blocks: list[tuple[int, str, str]] = []
    for node in TREE.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name in wanted
        ):
            start = min([node.lineno, *[decorator.lineno for decorator in node.decorator_list]])
            end = node.end_lineno or node.lineno
            block = textwrap.dedent("\n".join(SOURCE_LINES[start - 1 : end])).strip()
            blocks.append((start, node.name, block))
    found = {name for _, name, _ in blocks}
    missing = wanted - found
    if missing:
        raise KeyError(f"missing source blocks: {sorted(missing)}")
    return "\n\n".join(block for _, _, block in sorted(blocks))


def exercise(
    *,
    number: int,
    title: str,
    difficulty: int,
    importance: int,
    minutes: int,
    question: str,
    learner_code: str,
    solution_names: tuple[str, ...],
    test_code: str,
    expected: str,
    help_text: str,
    interpretation: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    heading = md(f"""
        ### Exercise {number}: {title}

        > **Difficulty:** {difficulty}/5 &nbsp; | &nbsp; **Importance:** {importance}/5 &nbsp; | &nbsp; **Time:** {minutes} minutes

        {question}
        """)
    learner = code(learner_code, tags=["exercise"])
    solution_source = implementation(*solution_names)
    ast.parse(solution_source)
    solved = code(solution_source, tags=["solution"])
    tests_cell = code(test_code, tags=["tests"])
    dropdown_source = (
        "<details>\n"
        "<summary>Expected output</summary>\n\n"
        f"{expected}\n\n"
        "</details>\n\n"
        "<details>\n"
        "<summary>Help - reveal a hint</summary>\n\n"
        f"{help_text}\n\n"
        "</details>\n\n"
        "<details>\n"
        "<summary>Solution</summary>\n\n"
        "```python\n"
        f"{solution_source}\n"
        "```\n\n"
        "</details>\n\n"
        "<details>\n"
        "<summary>Interpretation</summary>\n\n"
        f"{interpretation}\n\n"
        "</details>"
    )
    dropdown = md(dropdown_source)
    return [heading, learner, tests_cell, dropdown], [heading, solved, tests_cell, dropdown]


intro = [
    md("""
        # [14.2] I-JEPA from Scratch

        **By the end of this notebook, you will have shown that an I-JEPA trained from scratch uses
        a visible object-class key to predict the latent target of a completely masked object on
        pixel-distinct scenes sharing all twelve object-position templates, because correct context
        reaches 100% object recovery while cyclically mismatched context reaches 0%, zero context
        raises latent loss, and an untrained predictor remains at chance.**

        Training and evaluation share every semantic template; only background pixels differ. This
        is an exact teaching world, not ImageNet or checkpoint parity. The point is to build the
        asymmetric latent-prediction mechanism yourself, then falsify easier explanations.

        ## Core Question

        When the target object is entirely hidden, does the online branch learn to use visible context to predict its target-encoder representation?
        """),
    code(
        """
        import copy
        import sys
        from typing import Any
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import torch as t
        from torch import Tensor, nn
        from torch.nn import functional as F

        SECTION_DIR = Path.cwd()
        if not (SECTION_DIR / "solutions.py").exists():
            SECTION_DIR = Path("chapter14_jepa_world_models/exercises/part2_ijepa_from_scratch").resolve()
        if str(SECTION_DIR.parent) not in sys.path:
            sys.path.insert(0, str(SECTION_DIR.parent))

        from part2_ijepa_from_scratch import solutions as reference
        from part2_ijepa_from_scratch import tests
        from part2_ijepa_from_scratch.solutions import (
            CollapseDiagnostics,
            IJEPARun,
            PredictionMetrics,
            ProbeMetrics,
            SceneBatch,
            TrainingTrace,
        )

        IMAGE_SIZE = 32
        PATCH_SIZE = 4
        GRID_SIZE = IMAGE_SIZE // PATCH_SIZE
        N_PATCHES = GRID_SIZE**2
        CONTRACT_SEEDS = (0, 1, 2)
        OBJECT_NAMES = ("red square", "blue circle", "green diamond")
        POSITION_NAMES = ("top left", "top right", "bottom left", "bottom right")
        KEY_ROW_SLICE = slice(0, PATCH_SIZE)
        KEY_COLUMN_SLICE = slice(3 * PATCH_SIZE, 4 * PATCH_SIZE)

        EXERCISE_ID = "14_2_ijepa_from_scratch"
        GT_TIER = "GT-0"
        EXPECTED_RUNTIME = "120-180 minutes; under one minute for release verification"
        REQUIRES_GPU = True
        """,
        tags=["setup"],
    ),
    md("""
        ## Learning Objectives

        You will be able to:

        - generate an exact image world whose target mask hides every class-coloured object pixel;
        - implement patchification, structured masking, a context encoder, and a latent predictor;
        - enforce the stop-gradient and exponential-moving-average target boundary;
        - distinguish low latent loss from non-collapsed, causally useful prediction;
        - fit transfer probes, then recognize when random features make those probes non-evidence;
        - intervene on visible context and measure prediction accuracy and latent loss on disjoint scenes.

        ## Cold Open: What Must the Predictor Know?

        Every scene contains a small colour key at the top and one geometric object elsewhere.
        During training, the entire object is hidden from the online branch. The target encoder sees
        the full image, but it never receives gradients. If the predictor learns the task, swapping
        the visible key for another object's key should change the predicted latent class while
        leaving the target fixed.

        Before coding, predict all four outcomes:

        | condition | expected object accuracy | expected latent loss |
        |---|---:|---:|
        | correct held-out context | high | low |
        | cyclic object mismatch | near 0 | higher |
        | zero context | chance | higher |
        | untrained model | chance | much higher |

        ### Common Bugs

        - Masking only the object's centre leaves class-coloured edge pixels visible and invalidates the context claim.
        - Updating the target with gradients turns the matched no-EMA failure mode into the main model.
        - Fitting and testing a probe on the same generated scenes measures memorization, not transfer.
        - Reading high probe accuracy without a random-encoder control mistakes easy random features for learned structure.
        """),
]


architecture_primer = md(r"""
    ## From Masked Patches to a Predictive Representation

    You now have three pieces with exact meanings: patch index `j` names one spatial
    location, `target_mask[:, j]` says whether that location is hidden from the online
    branch, and `gather_masked_tokens` preserves both batch identity and row-major order.
    We can therefore build the actual I-JEPA asymmetry without hand-waving about what
    either branch saw.

    ### Why Predict a Latent Instead of Pixels?

    Pixel reconstruction rewards every detail, including background noise. I-JEPA asks a
    different question: can visible context predict the *representation* that a slowly
    moving target encoder assigns to a hidden region? In our world, the noisy background
    is unpredictable but the visible colour key determines the hidden object's class.
    A useful predictor should therefore preserve the predictable factor and need not
    reproduce each background pixel.

    Let $x \in \mathbb{R}^{N \times d_p}$ be the patch vectors, $M$ the target positions,
    and $C=\lnot M$ the visible context positions. The forward pass is

    $$
    \begin{aligned}
    z_C &= f_\theta(x_C + p_C), \\
    z_M &= \operatorname{stopgrad}\!\left(f_\xi(x + p)_M\right), \\
    \hat z_M &= g_\phi\!\left(\operatorname{scatter}(z_C, C) + m_M + q\right)_M, \\
    \mathcal{L} &= \frac{1}{|M|}\sum_{j \in M}
      \left\|\frac{\hat z_j}{\|\hat z_j\|_2}
      - \frac{z_j}{\|z_j\|_2}\right\|_2^2, \\
    \xi &\leftarrow \tau\xi + (1-\tau)\theta.
    \end{aligned}
    $$

    There are three distinct learned objects here. The online encoder $f_\theta$ turns
    visible patches into context latents. The predictor $g_\phi$ reasons over a complete
    spatial grid containing context latents and learned mask tokens. The target encoder
    $f_\xi$ defines the latent targets but is updated only by EMA.

    ### Visual Map of One 8 by 8 Scene

    ```text
    row 0:  C  C  C  K  C  C  C  C       K = visible class key
    row 1:  C  T  T  C  C  C  C  C       T = masked target patch
    row 2:  C  T  T  C  C  C  C  C       C = visible context patch
    rows 3-7: all C for this top-left example

    online encoder input: 60 visible patch vectors, in row-major order
    predictor input:       64 latent slots after context scatter + mask-token fill
    target encoder input:  all 64 patch vectors, followed by a gather of the 4 T slots
    loss operands:         4 predicted vectors and 4 stop-gradient target vectors
    ```

    The key is visible, but every class-coloured object pixel is inside the four `T`
    patches. This is why a key-only swap is a meaningful intervention rather than a
    cosmetic edit.

    ### A Four-Patch Calculation Before the Transformer

    Suppose visible positions are `[0, 2]`. After a linear patch projection and positional
    addition, the encoder must return exactly `[Wx_0+b+p_0, Wx_2+b+p_2]`, not the first two
    projected patches. The predictor then reconstructs the spatial sequence

    ```text
    [context_0, mask_token, context_2, mask_token] + predictor_position
    ```

    before mixing positions and gathering targets `[1, 3]`. The immediate test replaces
    both transformers with deterministic maps and checks these values exactly. A module
    that returns correctly shaped zeros, ignores positions, or packs context contiguously
    cannot pass.

    ### Architecture Decomposition

    | component | input | output | what the exercise must preserve |
    |---|---|---|---|
    | patch projection | `[B, N, patch_dim]` | `[B, N, d_model]` | patch content and learned position |
    | online encoder | visible rows only | `[B, |C|, d_model]` | the original visible-position order |
    | predictor scatter | context + target mask | `[B, N, d_model]` | context at `C`, mask token at `M` |
    | predictor transformer | complete latent grid | `[B, N, d_model]` | communication from context to targets |
    | target encoder | all patches | `[B, N, d_model]` | no gradient path from the loss |
    | target gather | full target encoding + `M` | `[B, |M|, d_model]` | exact alignment with predictions |

    <details>
    <summary>Question - why scatter context back into a full grid?</summary>

    A compact list of 60 vectors does not by itself say which four of the 64 positions are
    absent. Scattering restores that spatial identity. Predictor positional embeddings
    then tell the model which context and target locations it is relating.

    </details>

    <details>
    <summary>Question - why does stop-gradient need EMA?</summary>

    Stop-gradient prevents the target branch from moving to reduce the current loss.
    EMA still lets the target track the online encoder slowly, so its representation is a
    stable learning target rather than a permanently random network. Later you will train
    a matched jointly optimized target to see why low loss alone is insufficient.

    </details>

    <details>
    <summary>Question - would zero latent loss prove semantic learning?</summary>

    No. Constant target and prediction vectors can make the objective easy. That is why
    collapse diagnostics, an untrained architecture, a matched compound target ablation, and causal
    key interventions are part of one result rather than optional extras.

    </details>

    ### Common Bugs in the Core Architecture

    - Adding positions *after* visible-token gathering with indices `0..59` silently gives
      a bottom-right patch the wrong spatial identity.
    - Concatenating mask tokens after context tokens is not spatial scatter; the predictor
      must receive all 64 slots in their original locations.
    - Gathering the target encoder with `~target_mask` swaps the teaching signal and the
      context by shape-compatible accident.
    - Calling `.detach()` on the online context removes the learning path; detach only the
      target vectors.
    - Putting target parameters in the optimizer invalidates the EMA comparison even if
      `requires_grad_(False)` is restored later.
    """)


specs = [
    dict(
        number=1,
        title="Build an Exact Predictive World",
        difficulty=2,
        importance=5,
        minutes=15,
        question="Draw three objects in four locations. Keep the object fully inside its future target block and place a separated class-colour key in visible context. Labels are never used by the self-supervised loss.",
        learner_code="""
        def _draw_scene(object_id: int, position_id: int, *, jitter: int, seed: int) -> Tensor:
            # Return one [3, 32, 32] scene. The key is patch [row 0, column 3].
            raise NotImplementedError()

        def build_scene_dataset(
            examples_per_combination: int = 24, *, seed: int = 0, jitter: int = 0
        ) -> SceneBatch:
            raise NotImplementedError()
        """,
        solution_names=("_draw_scene", "build_scene_dataset"),
        test_code="tests.test_build_scene_dataset(build_scene_dataset)",
        expected="`All tests in test_build_scene_dataset passed!` The test also proves that no class-coloured object pixel leaks outside the target region and visible key.",
        help_text="Generate in semantic order: object, position, example. Use deterministic per-scene seeds. Make the objects small enough to fit inside their 2x2 target-patch blocks.",
        interpretation="Exact generated worlds are useful only when their ground truth removes shortcuts. Here, object identity outside the mask is restricted to one inspectable key.",
    ),
    dict(
        number=2,
        title="Patchify Without Mixing Space",
        difficulty=2,
        importance=4,
        minutes=10,
        question="Map channel-first images to row-major non-overlapping patch vectors. Start from the hand-computed 4 by 4 oracle.",
        learner_code="""
        def patchify(images: Tensor, patch_size: int = PATCH_SIZE) -> Tensor:
            raise NotImplementedError()
        """,
        solution_names=("patchify",),
        test_code="tests.test_patchify_hand_computed(patchify)\ntests.test_patchify_rejects_bad_shape(patchify)",
        expected="Both tests pass; the 4 by 4 oracle becomes four row-major vectors with no channel or spatial mixing.",
        help_text="Use two `unfold` operations, move patch-grid axes before channel and within-patch axes, call `contiguous`, then flatten.",
        interpretation="A patch index now has a precise spatial meaning. Later mask and intervention claims depend on preserving it.",
    ),
    dict(
        number=3,
        title="Construct Random and Ground-Truth Target Blocks",
        difficulty=2,
        importance=5,
        minutes=12,
        question="Implement generic contiguous masks, then the exact object mask used by the signature experiment. The object mask must leave the key patch visible.",
        learner_code="""
        def sample_block_masks(
            batch_size: int, *, grid_size: int = GRID_SIZE, block_size: int = 2,
            generator: t.Generator | None = None, device: t.device | str = "cpu",
        ) -> Tensor:
            raise NotImplementedError()

        def object_target_masks(
            position_labels: Tensor, *, grid_size: int = GRID_SIZE, block_size: int = 2,
            device: t.device | str | None = None,
        ) -> Tensor:
            raise NotImplementedError()
        """,
        solution_names=("sample_block_masks", "object_target_masks"),
        test_code="tests.test_sample_block_masks(sample_block_masks)\ntests.test_object_target_masks(object_target_masks)",
        expected="Every random block is contiguous; every object target selects four patches; patch `(0, 3)` is always visible.",
        help_text="Represent a mask as `[batch, 8, 8]` while constructing it, then flatten. The four object-block starts are `(1,1)`, `(1,5)`, `(5,1)`, and `(5,5)`.",
        interpretation="Random blocks teach the general operation. Exact object blocks make the final causal question unambiguous.",
    ),
    dict(
        number=4,
        title="Gather Masked Tokens Safely",
        difficulty=2,
        importance=3,
        minutes=8,
        question="Gather a fixed number of target or visible positions from every batch row without collapsing the batch axis.",
        learner_code="""
        def gather_masked_tokens(tokens: Tensor, mask: Tensor) -> Tensor:
            raise NotImplementedError()
        """,
        solution_names=("gather_masked_tokens",),
        test_code="tests.test_gather_masked_tokens(gather_masked_tokens)",
        expected="The exact two-row oracle passes and unequal selected-token counts raise a useful error.",
        help_text="Boolean indexing flattens the first two axes. Validate equal row counts, index once, then reshape back to `[batch, selected, width]`.",
        interpretation="This helper is small, but a silent batch mix-up would invalidate every later target and control.",
    ),
    dict(
        number=5,
        title="Build the Context Encoder and Latent Predictor",
        difficulty=4,
        importance=5,
        minutes=25,
        question="The encoder sees only visible tokens. The predictor scatters them into a full grid, inserts learned mask tokens, and returns only target positions.",
        learner_code="""
        class PatchTransformerEncoder(nn.Module):
            def __init__(self, patch_dim: int = 3 * PATCH_SIZE**2, d_model: int = 48) -> None:
                super().__init__()
                raise NotImplementedError()

            def forward(self, patches: Tensor, visible_mask: Tensor | None = None) -> Tensor:
                raise NotImplementedError()

        class LatentPredictor(nn.Module):
            def __init__(self, d_model: int = 48) -> None:
                super().__init__()
                raise NotImplementedError()

            def forward(self, context: Tensor, target_mask: Tensor) -> Tensor:
                raise NotImplementedError()
        """,
        solution_names=("PatchTransformerEncoder", "LatentPredictor"),
        test_code="tests.test_encoder_and_predictor_semantics(PatchTransformerEncoder, LatentPredictor)",
        expected="The exact four-patch oracle verifies patch projection, positional addition, visible-position gathering, context scattering, mask-token insertion, predictor mixing, and target gathering. Shape-correct zero modules fail.",
        help_text="Give both modules positional embeddings. In the predictor, initialize the full sequence from a learned mask token and overwrite visible rows with context tokens.",
        interpretation="The online branch cannot copy target pixels because those patch embeddings never enter its encoder.",
    ),
    dict(
        number=6,
        title="Implement the Latent Objective and EMA",
        difficulty=3,
        importance=5,
        minutes=15,
        question="Compare unit-normalized latent directions, stop gradients through targets, and implement the target update exactly.",
        learner_code="""
        def normalized_prediction_loss(predicted: Tensor, target: Tensor) -> Tensor:
            raise NotImplementedError()

        @t.no_grad()
        def update_target_encoder(online: nn.Module, target: nn.Module, ema_decay: float) -> None:
            raise NotImplementedError()
        """,
        solution_names=("normalized_prediction_loss", "update_target_encoder"),
        test_code="tests.test_normalized_prediction_loss(normalized_prediction_loss)\ntests.test_update_target_encoder(update_target_encoder)",
        expected="Identical directions have zero loss, positive rescaling changes nothing, targets receive no gradients, and the hand-computed EMA update is exact.",
        help_text="Normalize over the final axis. Detach the target before subtraction. `Tensor.lerp_` implements the EMA update without constructing new parameters.",
        interpretation="JEPA predicts representations, not pixels. The asymmetry comes from stop-gradient plus a slowly moving target, not a different architecture.",
    ),
    dict(
        number=7,
        title="Assemble I-JEPA and Enforce the Gradient Boundary",
        difficulty=3,
        importance=5,
        minutes=15,
        question="Deep-copy the online encoder into a frozen target encoder. Return predicted and target vectors for exactly the masked positions.",
        learner_code="""
        class IJEPAModel(nn.Module):
            def __init__(self, d_model: int = 48) -> None:
                super().__init__()
                raise NotImplementedError()

            def forward(self, patches: Tensor, target_mask: Tensor) -> tuple[Tensor, Tensor]:
                raise NotImplementedError()

            @t.no_grad()
            def representations(self, patches: Tensor) -> Tensor:
                raise NotImplementedError()
        """,
        solution_names=("IJEPAModel",),
        test_code="tests.test_ijepa_gradient_boundary(IJEPAModel)\ntests.test_one_real_optimization_step(IJEPAModel)",
        expected=(
            "The target has no gradients, online parameters do have gradients, and one "
            "real finite-loss optimizer step changes both the online encoder and the EMA "
            "target. A single stochastic step is not claimed to reduce the "
            "objective."
        ),
        help_text="Freeze the copied target with `requires_grad_(False)` and evaluate it under `no_grad`. Mean full-image target tokens form the later diagnostic representation.",
        interpretation="A copied target encoder is not enough. The tests verify the actual gradient graph and a real optimization step.",
    ),
    dict(
        number=8,
        title="Detect Collapse Before Interpreting Features",
        difficulty=3,
        importance=5,
        minutes=15,
        question="Measure feature scale, spectral effective rank, and off-diagonal cosine similarity after centering.",
        learner_code="""
        def collapse_diagnostics(features: Tensor) -> CollapseDiagnostics:
            raise NotImplementedError()
        """,
        solution_names=("collapse_diagnostics",),
        test_code="tests.test_collapse_diagnostics(collapse_diagnostics)",
        expected="Constant features have zero spread and rank one; a known orthogonal toy matrix has larger spread and effective rank.",
        help_text="Square singular values to obtain a covariance spectrum, normalize it, and exponentiate its entropy. Clamp before logarithms.",
        interpretation="Effective rank is scale-invariant. Always read it beside feature standard deviation rather than treating one number as a collapse oracle.",
    ),
    dict(
        number=9,
        title="Use Probes, Then Try to Disqualify Them",
        difficulty=3,
        importance=5,
        minutes=20,
        question="Fit closed-form ridge probes on pretraining scenes and test on pixel-distinct scenes sharing all twelve object-position templates. Keep random-encoder, shuffled-label, and constant-feature controls.",
        learner_code="""
        def fit_ridge_classifier(train_features: Tensor, train_labels: Tensor, ridge: float = 1e-4) -> Tensor:
            raise NotImplementedError()

        def _probe_accuracy(
            train_features: Tensor, train_labels: Tensor, test_features: Tensor, test_labels: Tensor
        ) -> float:
            raise NotImplementedError()

        def evaluate_probes(
            features: Tensor, object_labels: Tensor, position_labels: Tensor,
            *, seed: int = 0, train_fraction: float = 0.65,
        ) -> ProbeMetrics:
            raise NotImplementedError()

        def evaluate_transfer_probes(
            train_features: Tensor, train_object_labels: Tensor, train_position_labels: Tensor,
            test_features: Tensor, test_object_labels: Tensor, test_position_labels: Tensor,
        ) -> ProbeMetrics:
            raise NotImplementedError()
        """,
        solution_names=(
            "fit_ridge_classifier",
            "_probe_accuracy",
            "evaluate_probes",
            "evaluate_transfer_probes",
        ),
        test_code="tests.test_ridge_probe_toy_oracle(evaluate_probes)\ntests.test_transfer_probe_toy_oracle(evaluate_transfer_probes)",
        expected="The exact one-hot toy labels decode perfectly. In the signature run, random vision-transformer features also decode object and position well: this is a negative result, not evidence for I-JEPA learning.",
        help_text="Append a bias column, solve regularized normal equations, and leave the bias unregularized. Train and evaluation features must come from different generated scene seeds.",
        interpretation="A high probe score is not automatically evidence that pretraining created a representation. The random encoder is the decisive control in this easy world.",
    ),
    dict(
        number=10,
        title="Intervene on Context and Score the Prediction",
        difficulty=4,
        importance=5,
        minutes=25,
        question=(
            "Build target-class centroids from training scenes, predict pixel-distinct "
            "target latents, then replace or erase only the visible key patch while "
            "preserving every other pixel. Erasure copies neutral grayscale from the "
            "background; compare this with an all-zero latent "
            "context."
        ),
        learner_code="""
        def cyclic_object_mismatch_indices(
            object_labels: Tensor, position_labels: Tensor, *, shift: int = 1
        ) -> Tensor:
            raise NotImplementedError()

        def replace_scene_keys(scenes: SceneBatch, source_indices: Tensor) -> SceneBatch:
            raise NotImplementedError()

        def erase_scene_keys(scenes: SceneBatch) -> SceneBatch:
            raise NotImplementedError()

        @t.no_grad()
        def evaluate_prediction(
            model: IJEPAModel, centroid_scenes: SceneBatch, evaluation_scenes: SceneBatch,
            *, device: t.device, mismatch_shift: int = 1,
        ) -> PredictionMetrics:
            raise NotImplementedError()
        """,
        solution_names=(
            "cyclic_object_mismatch_indices",
            "replace_scene_keys",
            "erase_scene_keys",
            "evaluate_prediction",
        ),
        test_code=(
            "tests.test_cyclic_object_mismatch_preserves_position(cyclic_object_mismatch_in"
            "dices)\n"
            "tests.test_replace_scene_keys(replace_scene_keys)\n"
            "tests.test_erase_scene_keys(erase_scene_keys)\n"
            "tests.test_evaluate_prediction_exact_oracle(evaluate_prediction)"
        ),
        expected=(
            "Both nontrivial cyclic shifts are exact permutations. An exact key-reading "
            "model produces losses `0`, `2`, and `1` for correct, mismatched, and zero "
            "context; its accuracies are `1`, `0`, and `1` for target, "
            "original-after-swap, and alternate-key class; erasing the key gives accuracy "
            "`1/3` and loss `4/3` for this exact oracle. Every non-key pixel remains "
            "bit-identical."
        ),
        help_text=(
            "For each row, preserve its ordinal within an object-position cell and choose "
            "`(object + shift) mod 3`. Clone the original image and overwrite only rows "
            "`0:4`, columns `12:16`; encode that image while keeping the original target "
            "latent."
        ),
        interpretation="Unlike the easy probe, this isolates one visible class-bearing patch. Background noise, object pixels, target masks, positions, and target latents remain fixed.",
    ),
    dict(
        number=11,
        title="Train Matched EMA and Joint-Target Systems",
        difficulty=4,
        importance=5,
        minutes=35,
        question=(
            "Train the real objective, then rerun the same initialization, data order, "
            "optimizer, learning rate, epochs, masks, architecture, and loss while "
            "replacing stop-gradient/EMA with joint target optimization. Clip shared and "
            "target parameters separately so adding target parameters cannot rescale the "
            "shared update."
        ),
        learner_code="""
        def _extract_features(model: IJEPAModel, patches: Tensor, batch_size: int = 96) -> Tensor:
            raise NotImplementedError()

        def _train_model(
            scenes: SceneBatch, *, device: t.device, seed: int, epochs: int,
            batch_size: int, learning_rate: float, ema_decay: float, joint_target_ablation: bool,
        ) -> tuple[IJEPAModel, TrainingTrace, float]:
            raise NotImplementedError()

        """,
        solution_names=("_extract_features", "_train_model"),
        test_code="tests.test_training_loop_one_epoch(_train_model)\ntests.test_matched_joint_target_first_shared_update(_train_model)",
        expected="One real epoch produces finite diagnostics and moves the EMA target without gradients. On the first matched batch, every online-encoder and predictor parameter update is exactly identical between conditions.",
        help_text="Use the exact object masks for every batch. Clip online-encoder plus predictor gradients as one shared group in both conditions; clip target gradients separately only in the joint-target condition.",
        interpretation=(
            "The compound treatment replaces both stop-gradient and EMA with joint "
            "optimization. It cannot identify the effect of either change alone. The "
            "first-step parity test rules out global-clipping scale as a hidden treatment "
            "difference."
        ),
    ),
    dict(
        number=12,
        title="Orchestrate the Disjoint Evaluation Contract",
        difficulty=4,
        importance=5,
        minutes=25,
        question="Train both systems, evaluate prediction and probes on a separately generated scene seed, construct untrained, shuffled-label, zero-context, and constant-feature controls, and validate the complete scoped claim.",
        learner_code="""
        def run_experiment(
            *, device: str | t.device = "cpu", seed: int = 0,
            examples_per_combination: int = 20, epochs: int = 30,
        ) -> IJEPARun:
            raise NotImplementedError()

        def validate_run(
            run: IJEPARun, *, require_cuda: bool = False, max_vram_gb: float = 24.0
        ) -> None:
            raise NotImplementedError()
        """,
        solution_names=("run_experiment", "validate_run"),
        test_code="tests.test_run_experiment_structure(run_experiment)\ntests.test_validate_run_contract(validate_run)",
        expected=(
            "A one-epoch orchestration produces matched traces, zero exact "
            "train/evaluation overlap, and a key intervention whose non-key pixel "
            "difference is exactly zero. The learner's validator accepts one exact valid "
            "fixture and rejects nine independently corrupted fixtures, so a no-op "
            "validator fails immediately."
        ),
        help_text="Generate evaluation scenes with `seed + 50_000`. Keep target centroids on training scenes and every reported accuracy on evaluation scenes. Compute exact image overlap rather than trusting seed names.",
        interpretation="This final orchestration is intentionally separate from the training loop: evaluation provenance and causal-control construction are independently inspectable before the full CUDA result.",
    ),
]


learner_cells = list(intro)
solution_cells = list(intro)
for spec in specs:
    if spec["number"] == 5:
        learner_cells.append(architecture_primer)
        solution_cells.append(architecture_primer)
    learner_part, solution_part = exercise(**spec)
    learner_cells.extend(learner_part)
    solution_cells.extend(solution_part)


signature_md = md("""
    ## Signature Result: Does Visible Context Determine the Masked Latent?

    Run the complete experiment on CUDA. The learner implementations above produce every metric in
    the six-panel figure and per-seed table. `reference.plot_signature_result` only formats the
    resulting `IJEPARun`; it does not load `verification_report.json` or recompute the method.

    A convincing run must show all of the following together:

    - 100% held-out object recovery from correct context;
    - 0% original-class recovery and 100% alternate-class recovery after a key-only swap;
    - exactly zero pixel change outside the declared key patch;
    - higher loss under mismatched and zero context;
    - chance prediction from the same untrained architecture;
    - much smaller feature spread and weaker prediction in the matched joint-target/no-EMA system;
    - the negative result that random encoder features already support strong probes.

    ![I-JEPA signature result](../../instructions/assets/ijepa_signature_result.png)
    """)
release_contract = code(
    "_report_metrics = reference._report_metrics\n\n"
    + implementation("run_seed_suite")
    + "\n\n"
    + textwrap.dedent(
        """
    def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
        if not t.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU fallback is forbidden.")
        _, metrics = run_seed_suite(device="cuda", max_vram_gb=max_vram_gb)
        return metrics

    def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
        return run_gpu_test(max_vram_gb=max_vram_gb)
    """
    ),
    tags=["verification-contract"],
)
signature_code = code(
    """
    if not t.cuda.is_available():
        raise RuntimeError("Section 14.2 release result requires CUDA; no CPU fallback is used.")

    print(f"Torch {t.__version__} | CUDA runtime {t.version.cuda} | {t.cuda.get_device_name()}")
    seed_runs, seed_metrics = run_seed_suite(device="cuda", max_vram_gb=24.0)
    run = seed_runs[0]
    display(pd.DataFrame(seed_metrics["per_seed"]).round(4))

    signature_table = pd.DataFrame(
        [
            {"condition": "target latent", "object_accuracy": run.prediction.target_object_accuracy, "latent_loss": np.nan},
            {"condition": "trained / correct context", "object_accuracy": run.prediction.predicted_object_accuracy, "latent_loss": run.prediction.latent_loss},
            {"condition": "original class after key swap", "object_accuracy": run.prediction.mismatched_context_object_accuracy, "latent_loss": run.prediction.mismatched_context_loss},
            {"condition": "alternate key class", "object_accuracy": run.prediction.alternate_key_object_accuracy, "latent_loss": run.prediction.mismatched_context_loss},
            {"condition": "trained / erased key", "object_accuracy": run.prediction.key_erased_context_object_accuracy, "latent_loss": run.prediction.key_erased_context_loss},
            {"condition": "trained / zero context", "object_accuracy": np.nan, "latent_loss": run.prediction.zero_context_loss},
            {"condition": "matched joint target / no EMA", "object_accuracy": run.joint_target_prediction.predicted_object_accuracy, "latent_loss": run.joint_target_prediction.latent_loss},
            {"condition": "untrained architecture", "object_accuracy": run.random_prediction.predicted_object_accuracy, "latent_loss": run.random_prediction.latent_loss},
        ]
    )
    display(signature_table.round(4))
    print(
        f"feature spread: EMA={run.diagnostics.feature_std:.4f}, "
        f"compound ablation={run.joint_target_diagnostics.feature_std:.4f}; "
        f"exact pixel overlap={run.train_eval_exact_overlap}; "
        f"shared semantic templates={run.shared_semantic_template_count}/12; "
        f"non-key pixel diff={run.prediction.non_key_pixel_max_abs_diff:.1f}; "
        f"peak VRAM={run.peak_vram_gb:.3f} GiB"
    )
    print(
        "Probe warning: trained object/position "
        f"{run.probes.object_accuracy:.3f}/{run.probes.position_accuracy:.3f}, "
        "random encoder "
        f"{run.random_encoder_probes.object_accuracy:.3f}/{run.random_encoder_probes.position_accuracy:.3f}."
    )
    reference.plot_signature_result(run)
    print("SIGNATURE PASSED for all three declared seeds.")
    plt.show()
    """,
    tags=["signature", "cuda"],
)
signature_interp = md("""
    <details>
    <summary>Expected output</summary>

    On the pinned CUDA 13.2 run, correct-context prediction accuracy should be near `1.000`,
    original-class accuracy after a key-only swap near `0.000`, alternate-key accuracy near `1.000`,
    and untrained accuracy near chance. The non-key pixel difference must be exactly zero. Correct
    latent loss should be lower than key-mismatched and zero-context losses. Exact values are
    generated and checked below rather than treated as immutable constants.

    </details>

    <details>
    <summary>Help</summary>

    If prediction remains at chance, inspect the target mask first: every object pixel must be
    hidden and the key patch must remain visible. Then verify that target vectors are detached, EMA
    updates happen after the optimizer step, and the predictor scatters context into the correct
    patch positions.

    </details>

    <details>
    <summary>Interpretation</summary>

    The causal result is the paired prediction comparison, not the probe panel. Correct visible
    context determines the predicted target class on unseen scenes; replacing only the key changes
    the prediction to the alternate key's class and raises loss while every other pixel remains
    fixed. The untrained architecture rules out an architectural nearest-centroid shortcut. The
    matched compound target ablation can attain a small training-style loss while compressing
    feature scale and failing the held-out predictive criterion, demonstrating why low JEPA loss
    alone is insufficient.

    The probe panel is deliberately a negative result: random transformer features already linearly decode this simple synthetic world. Therefore trained probe accuracy cannot support the headline claim.

    `verification_report.json` is release evidence for CUDA/runtime/contracts only. It is not read by this notebook and is not part of the lesson's result.

    </details>
    """)


play_md = md("""
    ## Try It Yourself: Choose the Wrong Key

    Set `MISMATCH_SHIFT` to `1` or `2`. Both interventions preserve every non-key pixel and
    substitute only a different object's visible key. Predict the new class before running. You can
    also change `PLAY_SEED` to generate a disjoint world.
    """)
play_code = code(
    """
    PLAY_SEED = 17
    MISMATCH_SHIFT = 2  # choose 1 or 2; 0 is rejected because it is not an intervention

    play_train = build_scene_dataset(10, seed=PLAY_SEED)
    play_eval = build_scene_dataset(10, seed=PLAY_SEED + 50_000)
    play_model, _, _ = _train_model(
        play_train,
        device=t.device("cuda"),
        seed=PLAY_SEED,
        epochs=30,
        batch_size=48,
        learning_rate=2e-3,
        ema_decay=0.995,
        joint_target_ablation=False,
    )
    play = evaluate_prediction(
        play_model,
        play_train,
        play_eval,
        device=t.device("cuda"),
        mismatch_shift=MISMATCH_SHIFT,
    )
    display(pd.DataFrame([
        {"context": "correct", "accuracy": play.predicted_object_accuracy, "loss": play.latent_loss},
        {"context": "swapped: original label", "accuracy": play.mismatched_context_object_accuracy, "loss": play.mismatched_context_loss},
        {"context": "swapped: alternate label", "accuracy": play.alternate_key_object_accuracy, "loss": play.mismatched_context_loss},
        {"context": "erased key", "accuracy": play.key_erased_context_object_accuracy, "loss": play.key_erased_context_loss},
        {"context": "zero latent", "accuracy": np.nan, "loss": play.zero_context_loss},
    ]).round(4))
    print(
        f"correct={play.predicted_object_accuracy:.3f}, "
        f"original after key shift {MISMATCH_SHIFT}={play.mismatched_context_object_accuracy:.3f}, "
        f"alternate key={play.alternate_key_object_accuracy:.3f}, "
        f"non-key diff={play.non_key_pixel_max_abs_diff:.1f}, "
        f"loss ratio={play.mismatched_context_loss / play.latent_loss:.2f}x"
    )
    """,
    tags=["try-it-yourself", "cuda"],
)
play_interp = md("""
    <details>
    <summary>Interpretation</summary>

    A successful intervention changes the predicted object because the original object is completely
    absent from visible patches. If one shift behaves differently from the other, inspect
    class-centroid distances rather than averaging the anomaly away.

    </details>

    ## Bonus: Hunt the Anomalies

    1. The jointly optimized target can have lower latent loss but worse prediction. Plot its feature norm and centroid separation through training.
    2. Random encoder probes are strong. Remove positional embeddings or randomize patch projection width and determine which easy factors survive.
    3. Increase jitter until the exact 2x2 target no longer contains every object pixel. The leakage test should fail before the scientific claim does.
    4. Remove the key patch while keeping the target fixed. Is zero context equivalent to ordinary noisy background context?
    5. The signature reports every declared seed `0, 1, 2`. Add further seeds and report the worst case; do not hide an unstable seed behind a mean.

    ## Limitations and Paper Connection

    This notebook demonstrates the I-JEPA mechanism on a deliberately exact world with an explicit
    predictive cue. It does not establish natural-image semantic learning, ImageNet transfer, scale
    behavior, multi-block masking parity, or parity with Meta's pretrained checkpoints. Our
    objective is unit-vector squared distance with learned positions and a single label-located
    target block. The released implementation uses layer-normalized targets, smooth-L1 loss,
    sine/cosine positions, and sampled multi-block targets. The object-position labels here select
    the mask, so this is not label-free pretraining. The original I-JEPA work uses large vision
    transformers and large-scale image data; those claims require a separate real-checkpoint
    reproduction.

    - [I-JEPA paper](https://arxiv.org/abs/2301.08243)
    - [Official I-JEPA code](https://github.com/facebookresearch/ijepa)
    """)

learner_cells.extend(
    [
        signature_md,
        release_contract,
        signature_code,
        signature_interp,
        play_md,
        play_code,
        play_interp,
    ]
)
solution_cells.extend(
    [
        signature_md,
        release_contract,
        signature_code,
        signature_interp,
        play_md,
        play_code,
        play_interp,
    ]
)


def notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cells": [{**cell, "source": str(cell["source"]).splitlines(keepends=True)} for cell in cells],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.14"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def page_from_cells(cells: list[dict[str, object]]) -> str:
    chunks: list[str] = []
    for item in cells:
        source = (
            str(item["source"])
            .rstrip()
            .replace(
                "../../instructions/assets/ijepa_signature_result.png",
                "../assets/ijepa_signature_result.png",
            )
        )
        if item["cell_type"] == "code":
            chunks.append(f"```python\n{source}\n```")
        else:
            chunks.append(source)
    return "\n\n".join(chunks) + "\n"


# Refuse to publish malformed learner code or dropdowns.
for surface in (learner_cells, solution_cells):
    for item in surface:
        if item["cell_type"] == "code":
            ast.parse(str(item["source"]))

EXERCISE_PATH.write_text(json.dumps(notebook(learner_cells), indent=1) + "\n", encoding="utf-8")
SOLUTION_PATH.write_text(json.dumps(notebook(solution_cells), indent=1) + "\n", encoding="utf-8")
PAGE_PATH.write_text(page_from_cells(learner_cells), encoding="utf-8")
print(f"wrote {EXERCISE_PATH.name}, {SOLUTION_PATH.name}, and {PAGE_PATH.name}")
