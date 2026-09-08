from __future__ import annotations

import argparse
from datetime import datetime, timezone
import time
import ast
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import nbformat
import yaml

SECTION_DIR = Path(__file__).resolve().parent
ROOT = SECTION_DIR.parents[2]
ASSET_DIR = ROOT / "chapter12_vlm_interpretability/instructions/assets"
PAGE_DIR = ROOT / "chapter12_vlm_interpretability/instructions/pages"
EXPECTED_DIR = SECTION_DIR / "expected_outputs"
SOLUTION_PATH = SECTION_DIR / "solutions.py"
EXERCISE_NOTEBOOK = (
    SECTION_DIR / "12.5_Object_Hallucination_and_Modality_Arbitration_exercises.ipynb"
)
SOLUTION_NOTEBOOK = (
    SECTION_DIR / "12.5_Object_Hallucination_and_Modality_Arbitration_solutions.ipynb"
)
PAGE_PATH = PAGE_DIR / "05_[12.5]_Object_Hallucination_and_Modality_Arbitration.md"


SOURCE = SOLUTION_PATH.read_text()
TREE = ast.parse(SOURCE)
LINES = SOURCE.splitlines()


def _node_start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(decorator.lineno for decorator in decorators)])


def extract(names: list[str]) -> str:
    nodes = {
        getattr(node, "name", None): node
        for node in TREE.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    chunks = []
    for name in names:
        node = nodes[name]
        chunks.append("\n".join(LINES[_node_start(node) - 1 : node.end_lineno]))
    return "\n\n\n".join(chunks) + "\n"


def prelude() -> str:
    chunks = []
    for node in TREE.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
            chunks.append("\n".join(LINES[_node_start(node) - 1 : node.end_lineno]))
    chunks.append(extract(["ArbitrationExample", "RenderedScene", "ControlRegions"]))
    chunks.append(
        extract(
            [
                "_draw_shape",
                "_to_tensor",
                "tensor_to_pil",
                "render_scene",
                "build_prompt",
            ]
        )
    )
    return "\n\n".join(chunks)


EXERCISES = [
    {
        "number": 1,
        "title": "Encode a question without encoding its answer",
        "claim": "Parse only the queried color, queried shape, and stated prior. The presence label is not a model input.",
        "time": "15 minutes",
        "names": ["encode_prompt"],
        "stub": '''def encode_prompt(prompt: str) -> Tensor:
    """Return [color_id, shape_id, prior_id]; never encode the answer label."""
    raise NotImplementedError("Parse the visible prompt text")
''',
        "test": "tests.test_encode_prompt_contains_query_and_prior_not_answer(encode_prompt)",
        "expected": "`[2, 1, PRIOR_PRESENT]` for a green-circle query with a positive prior; tensor shape `(3,)`.",
        "help": "Match exactly one color and one shape from the declared vocabulary, then map the opening prior clause to one of three IDs.",
        "interpretation": "The prior is legitimate input context, but it is balanced across true and false labels. There is no answer token in this encoding.",
    },
    {
        "number": 2,
        "title": "Build the factorial arbitration benchmark",
        "claim": "Generate all eight color-shape queries in development, validation, and test renders, with balanced labels and priors.",
        "time": "25 minutes",
        "names": ["build_arbitration_benchmark", "make_counterfactual", "batch_inputs"],
        "stub": '''def build_arbitration_benchmark(variants_per_target: int = 3) -> list[ArbitrationExample]:
    """Return the audited 8 targets x 3 splits x 8 conditions benchmark."""
    raise NotImplementedError("Render the full factorial without label leakage")
''',
        "test": "tests.test_build_arbitration_benchmark(build_arbitration_benchmark)",
        "expected": "`192 examples`; exactly `64` per split; every split contains all `4 x 2` color-shape combinations.",
        "help": "Loop over `TARGETS`, then `SPLITS`, then the eight condition specifications. Positive priors must occur once with each label for every target.",
        "interpretation": "Because color and shape are fully crossed, neither feature can act as a proxy for the other. Validation and test use untouched render variants.",
    },
    {
        "number": 3,
        "title": "Construct fair image-region controls",
        "claim": "Map boxes to patches and choose two equal-size controls from image geometry outside both target and distractor boxes.",
        "time": "30 minutes",
        "names": [
            "bbox_to_patch_indices",
            "geometry_background_indices",
            "_translated_region_candidates",
            "matched_control_regions",
        ],
        "stub": """def bbox_to_patch_indices(
    bbox: tuple[int, int, int, int], *, image_size: int = IMAGE_SIZE, patch_grid: int = PATCH_GRID
) -> tuple[int, ...]:
    raise NotImplementedError("Return every positive-overlap patch")

def matched_control_regions(example: ArbitrationExample, *, seed: int = 0) -> ControlRegions:
    raise NotImplementedError("Return object/background/random regions that are pairwise disjoint")
""",
        "test": "tests.test_bbox_to_patch_indices(bbox_to_patch_indices)\ntests.test_matched_controls_are_true_background_equal_distinct_and_nonoverlapping(matched_control_regions)",
        "expected": "A 32x32 aligned object maps to four patches. Object, background, and random controls each contain four pairwise-disjoint indices.",
        "help": "Create a geometry exclusion set from the target and distractor boxes. Enumerate translated regions of the object's patch height and width, then choose two that do not overlap.",
        "interpretation": "A same-size control is not fair if it silently touches another object or duplicates the first control in a different order.",
    },
    {
        "number": 4,
        "title": "Measure hallucinations separately from accuracy",
        "claim": "Report accuracy, false-positive hallucinations on absent objects, and misses on present objects.",
        "time": "15 minutes",
        "names": [
            "binary_metrics",
            "benchmark_metrics",
            "condition_metrics",
            "modality_baseline_report",
        ],
        "stub": """def binary_metrics(logits: Tensor, labels: Tensor) -> dict[str, float]:
    raise NotImplementedError("Compute accuracy, hallucination rate, and miss rate")
""",
        "test": "tests.test_binary_metrics_are_semantic(binary_metrics)",
        "expected": "The four-row test fixture returns `0.5` for accuracy, hallucination rate, and miss rate.",
        "help": "Argmax the no/yes logits. Condition false-positive rate on `label == 0` and false-negative rate on `label == 1`.",
        "interpretation": "Overall accuracy can hide asymmetric object hallucinations, so the real-model gate requires joint accuracy, conflict accuracy, and hallucination rate separately.",
    },
    {
        "number": 5,
        "title": "Capture the residual stream",
        "claim": "Return logits and all five cache states, from pixel/prompt embeddings through visual routing, prior routing, integration, and readout.",
        "time": "15 minutes",
        "names": ["capture_arbitration_cache", "answer_margin"],
        "stub": """def capture_arbitration_cache(
    model: PixelPromptGroundTruthVLM,
    images: Tensor,
    prompt_tokens: Tensor,
    *,
    visual_scale: float = 1.0,
    text_scale: float = 1.0,
) -> tuple[Tensor, Tensor]:
    raise NotImplementedError("Stack every state returned by the organism")
""",
        "test": "tests.test_cache_has_pixel_and_prompt_routing_stages(capture_arbitration_cache)",
        "expected": "Cache shape `(5, batch, 38, 16)`; the visual margin appears after layer 0 and the stated prior after layer 1.",
        "help": "Call the model with `return_cache=True`, then stack the returned list on a new leading layer axis.",
        "interpretation": "The cache makes a precise causal question possible: at which state can replacing object-token rows still alter the answer?",
    },
    {
        "number": 6,
        "title": "Implement activation replacement",
        "claim": "Clone a recipient state and interpolate donor values into exactly the selected token rows.",
        "time": "20 minutes",
        "names": ["replace_token_activations"],
        "stub": """def replace_token_activations(
    recipient: Tensor,
    donor: Tensor,
    token_indices: Sequence[int],
    *,
    strength: float = 1.0,
) -> Tensor:
    raise NotImplementedError("Replace only selected rows without mutating either cache")
""",
        "test": "tests.test_replace_token_activations_is_exact_and_stub_sensitive(replace_token_activations)",
        "expected": "At strength `1.0`, selected rows equal the donor exactly; unselected rows and the recipient tensor remain unchanged. Strength `0.5` gives the midpoint.",
        "help": "Validate shapes and unique indices, clone the recipient, then assign `recipient + strength * (donor - recipient)` only at those rows.",
        "interpretation": "This is the actual causal intervention. Returning either input, replacing every token, or mutating the clean cache fails immediately.",
    },
    {
        "number": 7,
        "title": "Run the controlled counterfactual patch sweep",
        "claim": "Patch object, background, random, and full-visual rows at every cache state and score movement toward the donor answer.",
        "time": "35 minutes",
        "names": ["counterfactual_patch_sweep", "summarize_patch_rows"],
        "stub": """def counterfactual_patch_sweep(
    model: PixelPromptGroundTruthVLM,
    recipient: ArbitrationExample,
    donor: ArbitrationExample,
    *,
    strength: float = 1.0,
) -> pd.DataFrame:
    raise NotImplementedError("Patch each region at each layer and continue the forward pass")
""",
        "test": "tests.test_counterfactual_patch_sweep_is_causal_and_controlled(counterfactual_patch_sweep)",
        "expected": "At layer 0, object and full-visual patches move the margin by `8.0` and flip the answer; both disjoint matched controls have `0.0` effect.",
        "help": "Cache prompt-identical recipient/donor runs. For each layer and region, call your replacement function and resume with `model.continue_from`.",
        "interpretation": "A directional effect is positive only when the intervention moves the recipient toward its donor. Full visual is a positive control, not an equality oracle after modalities have mixed.",
    },
    {
        "number": 8,
        "title": "Select on validation, report once on test",
        "claim": "Choose the layer maximizing object directional effect over matched controls using validation pairs only.",
        "time": "25 minutes",
        "names": [
            "select_layer_on_validation",
            "intervention_playground",
            "_counterfactual_pairs",
            "toy_signature_result",
        ],
        "stub": """def select_layer_on_validation(
    model: PixelPromptGroundTruthVLM,
    validation_pairs: Sequence[tuple[ArbitrationExample, ArbitrationExample]],
    *,
    candidate_layers: Sequence[int] = tuple(range(N_LAYERS + 1)),
) -> tuple[int, pd.DataFrame]:
    raise NotImplementedError("Aggregate only validation rows and select the best layer")
""",
        "test": "tests.test_validation_selects_layer_and_test_remains_untouched(select_layer_on_validation)",
        "expected": "Validation selects layer `0` with score `8.0`; passing test recipients into the selector raises.",
        "help": "Aggregate each candidate's object directional effect, subtract the larger absolute control effect, and break ties toward the earlier layer.",
        "interpretation": "The signature test effect is confirmatory because no test row participates in layer choice.",
    },
]


MODEL_CODE = extract(["PixelPromptGroundTruthVLM"])
FIGURE_CODE = extract(
    ["make_cold_open_figure", "make_signature_figure", "make_layer_intervention_figure"]
)
REAL_CODE = extract(
    [
        "visual_token_indices",
        "bbox_to_vlm_token_indices",
        "geometry_matched_vlm_controls",
        "capture_module_output",
        "_require_cuda",
        "load_pinned_real_vlm",
        "_qwen_messages",
        "prepare_real_vlm_batch",
        "answer_token_ids",
        "_real_margin_from_outputs",
        "collect_real_vlm_evidence",
        "patch_real_vlm_layer",
        "real_activation_patch_controls",
        "_candidate_real_layers",
        "_summarize_real_selection",
        "select_real_layer_on_validation",
        "evaluate_real_acceptance",
        "run_real_vlm_benchmark",
        "run_smoke_test",
        "run_gpu_test",
        "run_full_experiment",
        "write_real_result",
    ]
)


def md(text: str):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip() + "\n")


def code(text: str):
    return nbformat.v4.new_code_cell(textwrap.dedent(text).strip() + "\n")


def exercise_markdown(exercise: dict[str, object]) -> str:
    return f"""### Exercise {exercise["number"]} - {exercise["title"]}

> **Claim tested.** {exercise["claim"]}
>
> **Difficulty:** 3/5 &nbsp;&nbsp; **Importance:** 5/5 &nbsp;&nbsp; **Time:** {exercise["time"]}

<details><summary>Expected output</summary>

{exercise["expected"]}

</details>

<details><summary>Help</summary>

{exercise["help"]}

</details>

<details><summary>Solution</summary>

```python
{extract(exercise["names"])}
```

</details>

<details><summary>Interpretation</summary>

{exercise["interpretation"]}

</details>"""


def learner_code(exercise: dict) -> str:
    """Place provided support definitions beside the method stubs in source order."""
    stub_source = exercise["stub"]
    stub_lines = stub_source.splitlines(keepends=True)
    stubs = {
        node.name: "".join(stub_lines[_node_start(node) - 1 : node.end_lineno]).rstrip()
        for node in ast.parse(stub_source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    chunks = [stubs.get(name, extract([name]).rstrip()) for name in exercise["names"]]
    return "\n\n\n".join(chunks) + "\n"


def intro_markdown(asset_prefix: str) -> str:
    return f"""# [12.5] Object Hallucination and Modality Arbitration

> **Claim.** In an explicit pixel-and-prompt ground-truth organism, a conflicting textual prior can
> cause object hallucination, while prompt-identical counterfactual activation patches isolate the
> causal visual region against two genuinely distinct geometry-matched controls.

## Core Question

**When a multimodal model says an object is present, did the answer come from pixels or from a stated textual prior?**

### Learning objectives

1. Build a full color x shape factorial without placing answer labels in model inputs.
2. Define true background from image geometry outside target and distractor boxes.
3. Implement token activation replacement and a layerwise counterfactual patch sweep.
4. Select intervention layers on validation and report only on untouched test data.
5. Distinguish a visual-counterfactual preflight from evidence that an intervention repairs hallucinations.

## Cold Open

<div class="arena-side-by-side" style="display:grid;grid-template-columns:minmax(240px,0.85fr) minmax(360px,1.4fr);gap:22px;align-items:center;">
<div>
<p>The pixels contain no red square. A prior caption claims one is present. The joint organism answers <strong>yes</strong>; removing only the prior restores <strong>no</strong>.</p>
<p>This is a visible failure to explain, not evidence by itself. The causal test must move only the target patch tokens while equal-size regions outside both objects remain inert.</p>
</div>
<img src="{asset_prefix}/vlm_hallucination_cold_open.png" alt="Absent red square with joint and image-question margins" style="width:100%;max-width:820px;">
</div>

## Scope and runtime

The chapter plan numbers this topic **12.5**. The folder retains its historical
`part2_object_hallucination_arbitration` path only to avoid changing shared chapter configuration.
The CPU ground-truth lab takes about 4-6 hours for a learner and seconds to execute solved. The
optional pinned-Qwen release run is budgeted at **12-25 minutes on CUDA** and may take up to 40
minutes before the acceptance timeout; this notebook never silently falls back to CPU.

The organism is deliberately analytic: it measures target-color occupancy from pixels, compares 2x2
regions with square/circle geometry from the prompt, routes the prompt's stated prior separately,
and then adds the channels. It is ground truth for causal mechanics, not evidence about
natural-image VLMs."""


def build_cells(solved: bool) -> list:
    cells = [md(intro_markdown("../../instructions/assets"))]
    cells.append(code(prelude() + "\nfrom IPython.display import display\nimport sys\n"))
    cells.append(code("""root_dir = next(
    path for path in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
    if (path / "chapter12_vlm_interpretability").exists()
)
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
import chapter12_vlm_interpretability.exercises.part2_object_hallucination_arbitration.tests as tests

t.manual_seed(0)
print(f"PyTorch {t.__version__}; CUDA intentionally unused in the core lab")"""))
    cells.append(
        md(
            "## 1. Build inputs without target leakage\n\nThe prompt contains a query and a possibly wrong prior. It never contains the image-ground-truth answer."
        )
    )
    for exercise in EXERCISES[:3]:
        cells.append(md(exercise_markdown(exercise)))
        cells.append(code(extract(exercise["names"]) if solved else learner_code(exercise)))
        cells.append(
            code(
                f"# Immediate test: fails while the stub is present\n{exercise['test']}\nprint('passed')"
            )
        )
        if exercise["number"] == 2:
            cells.append(code("""examples = build_arbitration_benchmark()
summary = pd.crosstab(
    [pd.Series([example.split for example in examples], name="split")],
    [pd.Series([f"{example.target_color} {example.target_shape}" for example in examples], name="query")],
)
display(summary)
print(f"{len(examples)} examples; labels per split:", {
    split: sum(example.label for example in examples if example.split == split)
    for split in SPLITS
})"""))

    cells.append(
        md(
            (
                "## 2. Inspect the pixel-and-prompt ground-truth organism\n"
                "\n"
                "Read this implementation before intervening. Its `forward` signature accepts "
                "image tensors and encoded prompt tokens only; labels, boxes, conditions, and "
                "split names are absent."
            )
        )
    )
    cells.append(code(MODEL_CODE))
    cells.append(
        code(
            """model = PixelPromptGroundTruthVLM()
test_examples = [example for example in examples if example.split == "test"]
images, prompt_tokens, labels = batch_inputs(test_examples)
logits = model(images, prompt_tokens)
print(benchmark_metrics(test_examples, logits) if "benchmark_metrics" in globals() else "metrics next")"""
        )
    )
    cells.append(md("## 3. Measure behavior and expose the causal method"))
    for exercise in EXERCISES[3:]:
        cells.append(md(exercise_markdown(exercise)))
        cells.append(code(extract(exercise["names"]) if solved else learner_code(exercise)))
        cells.append(
            code(
                f"# Immediate test: fails while the stub is present\n{exercise['test']}\nprint('passed')"
            )
        )
        if exercise["number"] == 4:
            cells.append(code("""metrics = benchmark_metrics(test_examples, logits)
baselines = modality_baseline_report(model, test_examples)
display(pd.DataFrame([metrics]))
display(baselines)"""))

    cells.append(
        md(
            "## Signature Result\n\nThe selected layer comes only from validation pairs. The rightmost panel reports object, background, random, and full-visual effects once on eight untouched test pairs."
        )
    )
    cells.append(code(FIGURE_CODE))
    cells.append(code("""signature = toy_signature_result()
asset_dir = root_dir / "chapter12_vlm_interpretability/instructions/assets"
asset_dir.mkdir(parents=True, exist_ok=True)
make_cold_open_figure(asset_dir / "vlm_hallucination_cold_open.png")
make_signature_figure(signature, asset_dir / "vlm_hallucination_signature.png")
make_layer_intervention_figure(signature, asset_dir / "vlm_hallucination_layer_interventions.png")
display(signature["validation_curve"])
display(pd.DataFrame([signature["test_patch_summary"]]))
print("accepted pixel/prompt ground-truth result:", signature["accepted"])"""))
    cells.append(
        md(
            (
                "## Try It Yourself\n"
                "\n"
                "Change all four parameters. `strength=0` is the recipient, `1` is exact donor "
                "replacement, and values above `1` extrapolate. The region control and layer "
                "choice are real interventions, not labels for precomputed "
                "outputs."
            )
        )
    )
    cells.append(code("""TARGET_COLOR = "blue"
TARGET_SHAPE = "circle"
LAYER = 0
REGION = "object_patch"  # object_patch | background_patch | same_size_random | full_visual_patch
STRENGTH = 1.0

play_example = next(
    example for example in examples
    if example.split == "test"
    and example.condition == "counterfactual_present"
    and example.target_color == TARGET_COLOR
    and example.target_shape == TARGET_SHAPE
)
display(tensor_to_pil(play_example.image))
print(intervention_playground(
    model, play_example, layer=LAYER, region=REGION, strength=STRENGTH
))"""))
    cells.append(
        md(
            (
                "## Real VLM Escalation\n"
                "\n"
                "**Visual-counterfactual preflight only.** These eight pairs have neutral "
                "priors; they do not intervene on the hallucinating conflict cases. Passing "
                "these gates cannot establish hallucination repair or distinguish perception, "
                "transfer, arbitration, and decoding. Those experiments remain "
                "unfinished.\n"
                "\n"
                "The pinned Qwen path uses the same prompt-identical image pairs and geometry "
                "controls. Candidate language layers are scored on all eight validation pairs. "
                "Only the selected layer is evaluated on eight untouched test pairs. "
                "Acceptance requires joint and conflict accuracy, hallucination rate, "
                "image-question and text-only baselines, donor-directed object effects or "
                "answer flips, two disjoint same-size controls, and a full-visual positive "
                "control. It deliberately does **not** require full-visual patching at an "
                "intermediate language layer to reconstruct the donor output after token "
                "mixing."
            )
        )
    )
    cells.append(code(REAL_CODE))
    cells.append(
        code(
            """RUN_CUDA_RELEASE = False
if RUN_CUDA_RELEASE:
    real_result = run_full_experiment(max_vram_gb=24.0)
    write_real_result(real_result, section_dir / "expected_outputs/real_qwen_result.json")
else:
    print("CUDA release run not requested; no model is loaded.")""".replace(
                "section_dir",
                "root_dir / 'chapter12_vlm_interpretability/exercises/part2_object_hallucination_arbitration'",
            )
        )
    )
    cells.append(
        md(
            (
                "## Interpreting the result\n"
                "\n"
                "The toy organism's conflict failure is programmed as a transparent "
                "arbitration rule, but its object evidence is computed from pixels and its "
                "query from prompt tokens. The causal result is therefore exact for this "
                "organism: replacing the four target-region token rows before visual routing "
                "flips the answer; replacing either of two distinct nonobject regions does "
                "not. The disappearance of patch effects at later states is expected because "
                "visual evidence has already moved into the answer "
                "row."
            )
        )
    )
    cells.append(
        md(
            (
                "## Bonus: Hunt an Anomaly\n"
                "\n"
                "Sweep intervention strength from 0 to 1.5 for both shapes. Look for a shape "
                "or position where the answer changes non-monotonically. Then perturb "
                "background brightness and verify that geometry controls remain outside both "
                "object boxes."
            )
        )
    )
    cells.append(
        md(
            (
                "## Limitations\n"
                "\n"
                "- The CPU organism is analytic ground truth over rendered shapes, not a "
                "trained or natural-image VLM.\n"
                "- The prompt encoder supports only the declared template and vocabulary.\n"
                "- A causal effect on this benchmark does not establish a universal "
                "hallucination circuit.\n"
                "- The pinned Qwen path remains unaccepted until the parent CUDA run records "
                "all behavioral and intervention gates.\n"
                "- Full-visual replacement at an intermediate language layer is only a "
                "positive control because prompt-token states may already contain mixed visual "
                "information."
            )
        )
    )
    cells.append(
        md(
            (
                "## Readings\n"
                "\n"
                "- Li et al., *Evaluating Object Hallucination in Large Vision-Language "
                "Models* (POPE).\n"
                "- Jiang et al., *Devils in Middle Layers of Large Vision-Language Models*.\n"
                "- Meng et al., *Locating and Editing Factual Associations in GPT* (causal "
                "tracing methodology)."
            )
        )
    )
    cells.append(
        md(
            "## Verification Appendix\n\nThe smoke fixture is generated from this executed CPU organism. `verification_report.json` is intentionally unaccepted for the real-VLM claim until CUDA evidence is supplied."
        )
    )
    cells.append(code("smoke = run_smoke_test(cpu=True)\nprint(json.dumps(smoke, indent=2))"))
    return cells


def notebook(solved: bool):
    nb = nbformat.v4.new_notebook(cells=build_cells(solved))
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "name": "python",
            "version": f"{sys.version_info.major}.{sys.version_info.minor}",
        },
    }
    for index, cell in enumerate(nb.cells):
        cell.id = f"arbitration-{index:03d}"
        if cell.cell_type == "code":
            ast.parse(cell.source)
    return nb


def page_text() -> str:
    parts = [intro_markdown("../assets")]
    parts.append(
        (
            "## Setup\n"
            "\n"
            "The notebooks contain the complete imports, renderer, data contracts, and "
            "learner tests. Work in the exercise notebook so every implementation is "
            "immediately executable. Provided support functions accompany each method "
            "stub; their definitions stay visible and "
            "editable."
        )
    )
    for exercise in EXERCISES:
        parts.append(exercise_markdown(exercise))
        parts.append(f"```python\n{learner_code(exercise)}\n```")
        parts.append(
            f"```python\n# Immediate test: fails while the stub is present\n{exercise['test']}\n```"
        )
    parts.append("""## Signature Result

<div class="arena-side-by-side" style="display:grid;grid-template-columns:minmax(240px,0.8fr) minmax(420px,1.6fr);gap:22px;align-items:center;">
<div>
<p>Validation selects the only state where target patch rows still carry causal pixel evidence. On
the untouched test split, object and full-visual patches move every answer toward the donor;
equal-size background and random controls are distinct, nonoverlapping, and inert.</p>
<p>The behavioral panels keep conflict accuracy and hallucination rate visible rather than hiding them inside aggregate accuracy.</p>
</div>
<img src="../assets/vlm_hallucination_signature.png" alt="Factorial examples, behavior metrics, validation layer selection, and test intervention controls" style="width:100%;max-width:1100px;">
</div>

![Layerwise object and control intervention effects](../assets/vlm_hallucination_layer_interventions.png)

## Try It Yourself

Use the parameterized notebook cell to choose target color, target shape, layer, region, and interpolation strength. It reruns the cache replacement and continuation; it does not index a saved answer.

## Real VLM Escalation

**Visual-counterfactual preflight only:** the intervention pairs have neutral priors.
They do not test repair of actual hallucinations, and the four-stage failure
diagnosis remains unfinished. Passing the preflight does not complete that claim.

The optional pinned Qwen run selects among four candidate layers using eight validation
counterfactual pairs, then freezes that layer and reports behavior plus intervention controls on 64
untouched test examples and eight untouched test pairs. Acceptance requires all declared behavior
and causal gates. A full-visual patch is a positive control, not a demand for invalid donor-output
equality after modalities have mixed.

## Limitations

- The CPU organism is analytic ground truth over rendered shapes, not a trained or natural-image VLM.
- The real-Qwen mechanism claim is unaccepted until the parent CUDA run passes every gate and updates the report.
- This controlled object-presence result does not establish a general hallucination circuit.
""")
    return "\n\n".join(parts).strip() + "\n"


def write_generated_sources() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook(False), EXERCISE_NOTEBOOK)
    nbformat.write(notebook(True), SOLUTION_NOTEBOOK)
    PAGE_PATH.write_text(page_text())

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from chapter12_vlm_interpretability.exercises.part2_object_hallucination_arbitration import (
        solutions,
    )

    result = solutions.toy_signature_result()
    import matplotlib.pyplot as plt

    plt.close(solutions.make_cold_open_figure(ASSET_DIR / "vlm_hallucination_cold_open.png"))
    plt.close(
        solutions.make_signature_figure(result, ASSET_DIR / "vlm_hallucination_signature.png")
    )
    plt.close(
        solutions.make_layer_intervention_figure(
            result, ASSET_DIR / "vlm_hallucination_layer_interventions.png"
        )
    )
    smoke = solutions.run_smoke_test(cpu=True)
    (EXPECTED_DIR / "smoke_test.json").write_text(
        json.dumps(smoke, indent=2, sort_keys=True) + "\n"
    )
    reference = {
        "status": "executed_cpu_ground_truth",
        "scope": "analytic pixel-and-prompt organism only",
        "metrics": smoke,
        "validation_layer_curve": result["validation_curve"].to_dict(orient="records"),
        "test_patch_summary": result["test_patch_summary"],
    }
    (EXPECTED_DIR / "reference_metrics.json").write_text(
        json.dumps(reference, indent=2, sort_keys=True) + "\n"
    )


def finalize_report() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from arena_ext.verification import build_report_input_manifest, current_git_commit
    from chapter12_vlm_interpretability.exercises.part2_object_hallucination_arbitration import (
        solutions,
    )

    inputs = [
        SOLUTION_PATH,
        SECTION_DIR / "tests.py",
        Path(__file__),
        SECTION_DIR / "README.md",
        SECTION_DIR / "artifacts.lock.yml",
        SECTION_DIR / "verification_report.schema.json",
        EXERCISE_NOTEBOOK,
        SOLUTION_NOTEBOOK,
        PAGE_PATH,
        ASSET_DIR / "vlm_hallucination_cold_open.png",
        ASSET_DIR / "vlm_hallucination_signature.png",
        ASSET_DIR / "vlm_hallucination_layer_interventions.png",
        EXPECTED_DIR / "README.md",
        EXPECTED_DIR / "reference_metrics.json",
        EXPECTED_DIR / "smoke_test.json",
    ]
    started = time.perf_counter()
    # Manifest freshness is checked after writing it; all executable section tests
    # must pass before this report can state tests_passed=True.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(SECTION_DIR / "tests.py"),
            "-k",
            "not local_report_schema_and_manifest_are_strict_and_fresh",
        ],
        cwd=ROOT,
        check=True,
    )
    report_inputs = build_report_input_manifest(ROOT, inputs)
    smoke = solutions.run_smoke_test(cpu=True)
    lock = yaml.safe_load((SECTION_DIR / "artifacts.lock.yml").read_text())
    report = {
        "notebook_id": lock["notebook_id"],
        "date_run": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(ROOT),
        "report_inputs": report_inputs,
        "gt_tier": lock["gt_tier"],
        "evidence_level": lock["evidence_level"],
        "claim_scope": lock["claim_scope"],
        "gpu_name": "NOT_RUN_CPU_ONLY_TASK",
        "peak_vram_gb": 0.0,
        "wall_clock_seconds": time.perf_counter() - started,
        "models": lock["models"],
        "datasets": lock["datasets"],
        "tests_passed": True,
        "metrics": {
            "cpu_smoke": smoke,
            "gpu_evidence": {
                "category": "cuda_section_metric",
                "status": "not_run",
                "uses_cuda": False,
                "section_specific_metric_keys": [
                    "joint_accuracy",
                    "conflict_accuracy",
                    "hallucination_rate",
                    "object_patch_directional_effect_mean",
                    "object_patch_flip_fraction",
                ],
            },
        },
        "baselines": {
            "declared_controls": lock["controls"],
            "expected_metrics": lock["expected_metrics"],
        },
        "negative_controls": {
            "cpu_status": "passed",
            "real_vlm_status": "not_run",
        },
        "ood_tests": {"natural_images": "out_of_scope"},
        "known_failures": [
            "Pinned Qwen visual-counterfactual preflight was not run; intervention on actual hallucinations and four-stage failure diagnosis remain unimplemented."
        ],
        "safety_notes": lock["safety_notes"],
        "accepted": False,
    }
    (SECTION_DIR / "verification_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        finalize_report()
    else:
        write_generated_sources()


if __name__ == "__main__":
    main()
