from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path
from textwrap import dedent

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapter9_alignment_interpretability.exercises.part5_rome_model_editing import solutions


ROOT = REPO_ROOT
SECTION = Path(__file__).resolve().parent
INSTRUCTION_PAGE = (
    ROOT / "chapter9_alignment_interpretability" / "instructions" / "pages"
    / "05_[9.5]_ROME_Model_Editing.md"
)
EXERCISE_NOTEBOOK = SECTION / "9.5_ROME_Model_Editing_exercises.ipynb"
SOLUTION_NOTEBOOK = SECTION / "9.5_ROME_Model_Editing_solutions.ipynb"


def source(*names: str) -> str:
    return "\n\n".join(inspect.getsource(getattr(solutions, name)).strip() for name in names)


def details(expected: str, help_text: str, interpretation: str, solution_code: str) -> str:
    return (
        "<details>\n<summary>Expected output</summary>\n\n"
        f"{expected}\n\n</details>\n\n"
        "<details>\n<summary>Help</summary>\n\n"
        f"{help_text}\n\n</details>\n\n"
        "<details>\n<summary>Interpretation</summary>\n\n"
        f"{interpretation}\n\n</details>\n\n"
        "<details>\n<summary>Solution</summary>\n\n```python\n"
        f"{solution_code}\n```\n\n</details>"
    )


def exercise_header(number: int, title: str, difficulty: int, minutes: int, body: str) -> str:
    red = "🔴" * difficulty + "⚪" * (5 - difficulty)
    return dedent(
        f"""
        ### Exercise {number} - {title}

        > Difficulty: {red}  
        > Importance: 🔵🔵🔵🔵🔵  
        > Suggested time: {minutes} minutes

        {body}
        """
    ).strip()


def build_cells(solution: bool) -> list:
    cells = []

    def md(text: str) -> None:
        cells.append(nbformat.v4.new_markdown_cell(dedent(text).strip()))

    def code(student_code: str, solution_code: str | None = None) -> None:
        selected = solution_code if solution and solution_code is not None else student_code
        cells.append(nbformat.v4.new_code_cell(dedent(selected).strip()))

    md(
        """
        # [9.5] Course ROME-like Model Editing

        > **Falsifiable claim:** On one pinned 70M decoder and one harmless Louvre counterfactual,
        > a course ROME-like covariance-aware rank-one edit can be written exactly into a real affine
        > MLP projection, but its wrong-position and multi-layer success are a negative localization
        > result rather than evidence for a privileged factual layer.

        The concrete question is: **can we change the Louvre completion from Paris toward Rome without
        mistaking a two-token comparison for knowledge editing or hiding collateral behavior?**

        ## Learning objectives

        By the end, you will be able to:

        1. derive the minimum-norm and covariance-aware rank-one equations;
        2. compare the learner write against an independent constrained-solve oracle;
        3. audit single-token and multi-token continuation scoring;
        4. run layer-by-position activation patching over nine independent noise directions;
        5. optimize a complete target sequence through the frozen decoder;
        6. qualify locality facts on the unedited model before measuring retention;
        7. inspect per-paraphrase outputs rather than only aggregate rates;
        8. compare a norm-matched layer sweep, a norm-matched wrong-position edit, shuffled-city targets,
           and five same-norm random rank-one updates;
        9. stratify target generalization, context override, non-target leakage, and legitimate Rome answers.

        ## The result you are building toward

        The signature figure is generated from live model runs. It combines activation-patching maps,
        fair norm-matched layer comparisons, base-qualified known-fact retention, stratified leakage,
        and the large value-optimizer KL.

        <img src="../assets/rome_model_editing_signature.png" width="1100">

        **Scope:** official ROME parity is not established. This course implementation lacks
        Wikipedia-derived covariance statistics, random-prefix key/value contexts, and released
        value-objective parity. It is not a CounterFact replication.
        """
    )
    code(
        """
        import json
        import math
        import os
        import sys
        from contextlib import contextmanager
        from dataclasses import asdict
        from pathlib import Path
        from typing import Any, Callable, Iterator, Sequence

        import matplotlib.pyplot as plt
        import pandas as pd
        import torch as t
        from IPython.display import display
        from torch import Tensor

        root_dir = next(
            p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "chapter9_alignment_interpretability").exists()
        )
        section_dir = root_dir / "chapter9_alignment_interpretability" / "exercises" / "part5_rome_model_editing"
        assets_dir = root_dir / "chapter9_alignment_interpretability" / "instructions" / "assets"
        if str(root_dir) not in sys.path:
            sys.path.append(str(root_dir))

        from chapter9_alignment_interpretability.exercises.part5_rome_model_editing import tests
        from chapter9_alignment_interpretability.exercises.part5_rome_model_editing import solutions as reference

        MODEL_ID = reference.MODEL_ID
        MODEL_REVISION = reference.MODEL_REVISION
        SUBJECT = reference.SUBJECT
        OLD_OBJECT = reference.OLD_OBJECT
        NEW_OBJECT = reference.NEW_OBJECT
        PromptRecord = reference.PromptRecord
        AdversarialRecord = reference.AdversarialRecord
        LearnerFunctions = reference.LearnerFunctions
        PARAPHRASE_PROVENANCE = reference.PARAPHRASE_PROVENANCE
        OPTIMIZATION_RECORDS = reference.OPTIMIZATION_RECORDS
        HELDOUT_PARAPHRASE_RECORDS = reference.HELDOUT_PARAPHRASE_RECORDS
        OPTIMIZATION_PROMPTS = reference.OPTIMIZATION_PROMPTS
        HELDOUT_PARAPHRASES = reference.HELDOUT_PARAPHRASES
        COVARIANCE_FACTS = reference.COVARIANCE_FACTS
        KNOWN_FACT_CANDIDATES = reference.KNOWN_FACT_CANDIDATES
        ADVERSARIAL_RECORDS = reference.ADVERSARIAL_RECORDS
        SIDE_EFFECT_PROMPTS = reference.SIDE_EFFECT_PROMPTS
        FLUENCY_TEXTS = reference.FLUENCY_TEXTS
        SAFE_GENERATION_PROMPTS = reference.SAFE_GENERATION_PROMPTS
        SHUFFLED_CONTROL_TARGETS = reference.SHUFFLED_CONTROL_TARGETS
        _batch_inputs = reference._batch_inputs
        _single_token_id = reference._single_token_id
        _normalize_target_list = reference._normalize_target_list
        _positions_for_mode = reference._positions_for_mode
        _mean = reference._mean
        normalize_target_text = reference.normalize_target_text
        load_pinned_model = reference.load_pinned_model
        next_token_probability_table = reference.next_token_probability_table
        next_token_distribution_table = reference.next_token_distribution_table
        text_perplexities = reference.text_perplexities
        generate_safe_continuations = reference.generate_safe_continuations
        match_update_norm = reference.match_update_norm

        t.manual_seed(0)
        device = t.device("cpu")
        print(f"torch={t.__version__}, device={device}, model={MODEL_ID}@{MODEL_REVISION[:8]}")
        """
    )
    md(
        """
        ## Cold open: what does the unedited model actually say?

        Load the pinned model and inspect both candidate probabilities and the true top token. A pairwise
        win is weaker than target top-1: Rome can beat Paris while some third token still wins.
        """
    )
    code(
        """
        model, tokenizer = load_pinned_model(device=device)
        cold = reference.score_rewrite_prompts(model, tokenizer, [record.prompt for record in HELDOUT_PARAPHRASE_RECORDS[:5]])
        cold_rows = []
        for index, record in enumerate(HELDOUT_PARAPHRASE_RECORDS[:5]):
            cold_rows.append({
                "id": record.prompt_id,
                "prompt": record.prompt,
                "P(Paris)": float(cold["selected_probs"][index, 0]),
                "P(Rome)": float(cold["selected_probs"][index, 1]),
                "Rome rank": int(cold["target_first_ranks"][index]),
                "top token": tokenizer.decode([int(cold["top_ids"][index])]),
            })
        display(pd.DataFrame(cold_rows))
        """
    )

    md("## 1. Exact rank-one writes")
    md(
        exercise_header(
            1,
            "the minimum-norm linear write",
            2,
            15,
            "Implement the smallest Frobenius-norm update satisfying `key @ (W + delta) = desired`.",
        )
    )
    minimum_stub = """
    def minimum_norm_rank_one_update(weight: Tensor, key: Tensor, desired_value: Tensor) -> Tensor:
        \"\"\"Return the minimum-Frobenius rank-one update.\"\"\"
        raise NotImplementedError()
    """
    code(minimum_stub, source("minimum_norm_rank_one_update"))
    code(
        """
        tests.test_minimum_norm_rank_one_update_exact(minimum_norm_rank_one_update)
        tests.test_minimum_norm_update_rejects_zero_key(minimum_norm_rank_one_update)
        tests.test_minimum_norm_matches_independent_constrained_oracle(minimum_norm_rank_one_update)
        """
    )
    md(
        details(
            "The write is exact, rank one, and agrees numerically with an independent full constrained solve.",
            "Write the residual first. The left factor must have dot product one with `key`.",
            "This proves one linear constraint. It says nothing about factual specificity.",
            source("minimum_norm_rank_one_update"),
        )
    )

    md(
        exercise_header(
            2,
            "covariance preconditioning",
            4,
            25,
            "Use Woodbury to compute `C^-1 key` without forming a 2048 by 2048 inverse.",
        )
    )
    covariance_stub = """
    def covariance_preconditioned_key(key: Tensor, key_bank: Tensor, *, ridge: float = 1e-2) -> Tensor:
        \"\"\"Compute C^-1 key for C = K.T K / n + ridge I.\"\"\"
        raise NotImplementedError()
    """
    code(covariance_stub, source("covariance_preconditioned_key"))
    code("tests.test_covariance_preconditioned_key_matches_dense_solve(covariance_preconditioned_key)")
    md(
        details(
            "The float64 Woodbury result matches a dense solve to tight tolerance.",
            "Set `A = K / sqrt(n)` and solve the smaller `ridge I + A A.T` system.",
            "The covariance corpus defines the geometry of interference. It must not double as the locality test set.",
            source("covariance_preconditioned_key"),
        )
    )

    md(
        exercise_header(
            3,
            "the affine-aware course ROME-like update",
            3,
            20,
            dedent(
                """
                Pythia's output projection is affine: `module(key) = key @ W + bias`. Implement the
                course ROME-like covariance-preconditioned equation using the complete module-output
                residual. This checks our declared algebra only; official ROME parity is not established.
                """
            ),
        )
    )
    rome_stub = """
    def course_rome_like_rank_one_update(
        weight: Tensor,
        key: Tensor,
        desired_module_output: Tensor,
        key_bank: Tensor,
        *,
        output_bias: Tensor | None = None,
        ridge: float = 1e-2,
    ) -> Tensor:
        \"\"\"Return the affine-aware course ROME-like update.\"\"\"
        raise NotImplementedError()
    """
    code(rome_stub, source("course_rome_like_rank_one_update"))
    code(
        """
        tests.test_course_rome_like_update_matches_declared_equation(course_rome_like_rank_one_update)
        tests.test_course_update_hits_actual_affine_module_output(course_rome_like_rank_one_update)
        """
    )
    md(
        details(
            "The declared course equation is satisfied and a nonzero-bias `nn.Linear` output is hit exactly.",
            "Compute the current complete module output before forming the residual.",
            "The test checks the actual affine operation, not a bias-free surrogate equation.",
            source("course_rome_like_rank_one_update"),
        )
    )

    code(
        """
        # This oracle vectorizes every matrix entry and solves the constraints independently;
        # it never calls minimum_norm_rank_one_update.
        oracle_delta = reference.independent_constrained_solve_oracle(
            toy_weight := t.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]], dtype=t.float64),
            toy_key := t.tensor([1.0, 1.0, 0.0], dtype=t.float64),
            toy_desired := t.tensor([-2.0, 3.0], dtype=t.float64),
        )
        learner_delta = minimum_norm_rank_one_update(toy_weight, toy_key, toy_desired)
        print("independent oracle max error", float((oracle_delta - learner_delta).abs().max()))
        toy = reference.exact_rank_one_toy_oracle()
        tests.test_toy_oracle_changes_only_the_requested_association(reference.exact_rank_one_toy_oracle)
        reference.plot_toy_oracle(str(assets_dir / "rome_rank_one_toy_oracle.png"))
        plt.show()
        """
    )

    md("## 2. Token positions and complete target scoring")
    md(
        exercise_header(
            4,
            "subject-token localization",
            2,
            15,
            "Use character-offset overlap so multi-token subjects and varying paraphrase positions are handled explicitly.",
        )
    )
    token_stub = """
    def token_positions_from_offsets(
        offsets: Sequence[Sequence[int]], subject_start: int, subject_end: int
    ) -> list[int]:
        raise NotImplementedError()


    def locate_subject_tokens(tokenizer: Any, prompt: str, subject: str) -> list[int]:
        raise NotImplementedError()
    """
    code(token_stub, source("token_positions_from_offsets", "locate_subject_tokens"))
    code(
        """
        tests.test_token_positions_from_offsets_handles_multitoken_subject(token_positions_from_offsets)
        tests.test_token_positions_ignores_empty_special_token_offsets(token_positions_from_offsets)
        print(locate_subject_tokens(tokenizer, "Visitors can find the Louvre in", SUBJECT))
        """
    )
    md(
        details(
            "The hand-built multi-token subject returns every overlapping token; the real tokenizer locates `Louvre` without a global index.",
            "Two half-open intervals overlap when `token_end > subject_start` and `token_start < subject_end`.",
            "A wrong token can still create a broad output bias, so token lookup belongs in the causal method.",
            source("token_positions_from_offsets", "locate_subject_tokens"),
        )
    )

    md(
        exercise_header(
            5,
            "single-token and multi-token target scoring",
            4,
            30,
            dedent(
                """
                Implement `target_token_ids` and `continuation_score_table`. Teacher-force every target
                token, sum log probabilities for the full target likelihood, and separately return the
                first target token's rank and full-sequence greedy match. Never silently truncate
                `San Francisco` to `San`.
                """
            ),
        )
    )
    scoring_stub = """
    def target_token_ids(tokenizer: Any, target: str) -> list[int]:
        raise NotImplementedError()


    def continuation_score_table(
        model: Any, tokenizer: Any, prompts: Sequence[str], targets: str | Sequence[str]
    ) -> dict[str, Any]:
        raise NotImplementedError()
    """
    code(scoring_stub, source("target_token_ids", "continuation_score_table"))
    code(
        """
        tests.test_continuation_scoring_uses_all_target_tokens(
            continuation_score_table, model, tokenizer
        )
        target_demo = continuation_score_table(
            model,
            tokenizer,
            ["The Colosseum is in", "The Golden Gate Bridge is in"],
            [" Rome", " San Francisco"],
        )
        display(pd.DataFrame({
            "target": target_demo["targets"],
            "token_count": target_demo["target_token_counts"],
            "full_target_likelihood": target_demo["target_likelihoods"],
            "first_token_rank": target_demo["first_target_ranks"],
            "full_greedy_match": target_demo["full_target_greedy_matches"],
        }))
        """
    )
    md(
        details(
            "A direct real-model calculation proves all target tokens are scored. The table shows that first-token rank and full-sequence probability answer different questions.",
            "Logits at `prompt_len - 1 + j` predict target token `j`.",
            "For this lab Rome is one token, but the evaluator is not built around that accident.",
            source("target_token_ids", "continuation_score_table"),
        )
    )

    md("## 3. Activation patching under independent corruptions")
    md(
        exercise_header(
            6,
            "capture complete MLP outputs",
            2,
            15,
            "Register one hook per MLP, clone each complete output, and remove every hook even if the forward pass fails.",
        )
    )
    capture_stub = """
    def capture_mlp_outputs(model: Any, tokenizer: Any, prompt: str) -> list[Tensor]:
        raise NotImplementedError()
    """
    code(capture_stub, source("capture_mlp_outputs"))
    code(
        """
        tests.test_capture_mlp_outputs_matches_direct_modules(capture_mlp_outputs, model, tokenizer)
        captured = capture_mlp_outputs(model, tokenizer, OPTIMIZATION_PROMPTS[0])
        assert len(captured) == model.config.num_hidden_layers
        print([tuple(output.shape) for output in captured])
        """
    )
    md(
        details(
            "One `[batch, position, hidden]` tensor is captured for each layer.",
            "Use a `finally` block to remove hooks.",
            "The traced state is the complete MLP contribution, which is also the value optimized later.",
            source("capture_mlp_outputs"),
        )
    )

    md(
        exercise_header(
            7,
            "layer-by-position activation patching over nine independent corruptions",
            5,
            45,
            dedent(
                """
                Implement one clean/corrupt patching grid, then aggregate three embedding-noise scales
                with three fresh Gaussian directions per scale. Reusing one direction at several scales
                is not independent evidence. This is activation patching, not protocol-parity causal tracing.
                """
            ),
        )
    )
    trace_stub = """
    def activation_patching_mlp_grid(
        model: Any,
        tokenizer: Any,
        prompt: str,
        subject: str,
        target: str,
        *,
        samples: int = 4,
        noise_scale: float = 3.0,
        seed: int = 0,
    ) -> dict[str, Any]:
        raise NotImplementedError()


    def independent_noise_conditions(
        noise_scales: Sequence[float],
        *,
        directions_per_scale: int = 3,
        seed: int = 0,
    ) -> list[tuple[float, int]]:
        raise NotImplementedError()


    def activation_patching_sweep(
        model: Any,
        tokenizer: Any,
        prompt: str,
        subject: str,
        target: str,
        *,
        noise_scales: Sequence[float] = (1.5, 3.0, 4.5),
        directions_per_scale: int = 3,
        seed: int = 0,
        samples: int = 4,
    ) -> dict[str, Any]:
        raise NotImplementedError()
    """
    code(
        trace_stub,
        source(
            "activation_patching_mlp_grid",
            "independent_noise_conditions",
            "activation_patching_sweep",
        ),
    )
    code(
        """
        tests.test_activation_patching_sweep_rejects_less_than_nine_conditions(independent_noise_conditions)
        tests.test_activation_patching_conditions_use_nine_independent_directions(independent_noise_conditions)
        trace_demo = activation_patching_sweep(
            model,
            tokenizer,
            OPTIMIZATION_PROMPTS[0],
            SUBJECT,
            OLD_OBJECT,
            noise_scales=(1.5, 3.0, 4.5),
            directions_per_scale=3,
            seed=0,
            samples=2,
        )
        display(pd.DataFrame({
            "condition": trace_demo["condition_labels"],
            "direction_seed": trace_demo["direction_seeds"],
            "best_layer": trace_demo["condition_best_layers"].tolist(),
        }))
        assert len(set(trace_demo["direction_seeds"])) == 9
        print({
            "selected_layer": trace_demo["selected_layer"],
            "selected_layer_win_fraction": trace_demo["selected_layer_win_fraction"],
            "layer_0_mean": trace_demo["layer_zero_mean_recovery"],
            "layer_0_std": trace_demo["layer_zero_std_recovery"],
        })
        """
    )
    md(
        details(
            "Nine independent directions produce mean and standard-deviation layer-by-position maps plus a condition-level winning-layer table.",
            "Reuse the exact same sampled noise for the corrupt baseline and every patched run within one condition.",
            "Layer 0 can win the aggregate while moving substantially across seeds and noise scales. That instability is evidence against treating one argmax as a unique storage location.",
            source(
                "activation_patching_mlp_grid",
                "independent_noise_conditions",
                "activation_patching_sweep",
            ),
        )
    )

    md("## 4. Keys, values, and the applied weight edit")
    md(
        exercise_header(
            8,
            "extract post-activation keys and complete values",
            3,
            20,
            "Hook `dense_h_to_4h`, apply the model activation, and separately capture the complete MLP output.",
        )
    )
    extraction_stub = """
    def extract_mlp_keys_and_values(
        model: Any,
        tokenizer: Any,
        prompts: Sequence[str],
        subjects: Sequence[str],
        layer: int,
        *,
        positions: Sequence[int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError()
    """
    code(extraction_stub, source("extract_mlp_keys_and_values"))
    code(
        """
        selected_layer = trace_demo["selected_layer"]
        keys, values = extract_mlp_keys_and_values(
            model, tokenizer, OPTIMIZATION_PROMPTS, [SUBJECT] * len(OPTIMIZATION_PROMPTS), selected_layer
        )
        assert keys.shape == (len(OPTIMIZATION_PROMPTS), model.config.intermediate_size)
        assert values.shape == (len(OPTIMIZATION_PROMPTS), model.config.hidden_size)
        assert t.isfinite(keys).all() and t.isfinite(values).all()
        print("keys", tuple(keys.shape), "complete MLP outputs", tuple(values.shape))
        """
    )
    md(
        details(
            "Pythia-70M returns 2048-dimensional keys and 512-dimensional complete MLP outputs.",
            "The key is after GELU; the value is the whole MLP module output, including the projection bias.",
            "Confusing residual states, pre-activation states, or a bias-free projection output changes the method.",
            source("extract_mlp_keys_and_values"),
        )
    )

    md(
        exercise_header(
            9,
            "optimize a complete target sequence",
            5,
            40,
            "Append every target prefix token, score every target token, and optimize one MLP-output delta while measuring KL on two other Louvre contexts.",
        )
    )
    optimize_stub = """
    def optimize_value_delta(
        model: Any,
        tokenizer: Any,
        prompts: Sequence[str],
        subject: str,
        targets: str | Sequence[str],
        layer: int,
        *,
        positions: Sequence[int] | None = None,
        preservation_prompts: Sequence[str] = ("The Louvre is a", "The Louvre museum has"),
        steps: int = 24,
        learning_rate: float = 0.25,
        kl_weight: float = 0.15,
        norm_weight: float = 0.02,
    ) -> tuple[Tensor, list[dict[str, float]]]:
        raise NotImplementedError()
    """
    code(optimize_stub, source("optimize_value_delta"))
    code(
        """
        tests.test_value_optimizer_history_describes_returned_delta(optimize_value_delta, model, tokenizer)
        demo_delta, demo_history = optimize_value_delta(
            model, tokenizer, OPTIMIZATION_PROMPTS[:2], SUBJECT, NEW_OBJECT, selected_layer, steps=3
        )
        display(pd.DataFrame(demo_history))
        assert demo_history[-1]["target_nll"] < demo_history[0]["target_nll"]
        assert math.isfinite(demo_history[-1]["preservation_kl"])
        print("demo preservation KL", demo_history[-1]["preservation_kl"])
        assert all(not parameter.requires_grad for parameter in model.parameters())
        """
    )
    md(
        details(
            "Target token NLL decreases while model parameters remain frozen.",
            "Optimize only `delta`; target token `j` is scored at `prompt_len - 1 + j`.",
            "The value is optimized through the nonlinear downstream model. This course objective is not released ROME value-objective parity, and the final run's large value-optimizer KL is negative evidence, not a footnote.",
            source("optimize_value_delta"),
        )
    )

    md(
        exercise_header(
            10,
            "temporary edits and matched random updates",
            3,
            20,
            "Implement exact restoration plus a seeded rank-one matrix with the requested Frobenius norm.",
        )
    )
    controls_stub = """
    def make_same_norm_random_rank_one(
        shape: Sequence[int],
        target_norm: float,
        *,
        seed: int = 0,
        device: t.device | str = "cpu",
        dtype: t.dtype = t.float32,
    ) -> Tensor:
        raise NotImplementedError()


    @contextmanager
    def temporary_mlp_weight_edit(
        model: Any, layer: int, delta_row_weight: Tensor
    ) -> Iterator[None]:
        raise NotImplementedError()
        yield
    """
    code(controls_stub, source("make_same_norm_random_rank_one", "temporary_mlp_weight_edit"))
    code(
        """
        tests.test_temporary_edit_restores_weights_after_errors(temporary_mlp_weight_edit)
        tests.test_random_rank_one_has_exact_norm_and_rank(make_same_norm_random_rank_one)
        tests.test_five_random_controls_are_seeded_distinct_and_norm_matched(make_same_norm_random_rank_one)
        """
    )
    md(
        details(
            "Five seeds are deterministic, distinct, rank one, and exactly norm matched.",
            "Clone the original parameter and restore with `copy_` in `finally`.",
            "A single random update is not a distribution. Rank and norm both need matching.",
            source("make_same_norm_random_rank_one", "temporary_mlp_weight_edit"),
        )
    )

    md("## 5. Labeled evaluation and experiment composition")
    md(
        exercise_header(
            11,
            "separate efficacy, qualified retention, and stratified leakage",
            4,
            30,
            dedent(
                """
                Implement the metric summary. Report `Rome > Paris` only as pairwise preference.
                Qualify known facts using base-model top-1 before scoring retention. Compute Rome leakage
                only on explicit context-override and non-target rows; legitimate Rome targets and
                target-generalization rows must never enter that leakage average.
                """
            ),
        )
    )
    summary_stub = """
    def qualify_known_fact_mask(base_scores: dict[str, Any]) -> Tensor:
        raise NotImplementedError()


    def summarize_edit_metrics(
        base: dict[str, Any],
        edited: dict[str, Any],
        known_fact_base: dict[str, Any],
        known_fact_edited: dict[str, Any],
        adversarial_base: dict[str, Any],
        adversarial_edited: dict[str, Any],
        side_base: dict[str, Tensor],
        side_edited: dict[str, Tensor],
        fluency_base: Tensor,
        fluency_edited: Tensor,
    ) -> dict[str, float | int]:
        raise NotImplementedError()
    """
    code(summary_stub, source("qualify_known_fact_mask", "summarize_edit_metrics"))
    code(
        """
        tests.test_known_fact_qualification_uses_only_base_top1(qualify_known_fact_mask)
        tests.test_adversarial_leakage_mask_excludes_all_legitimate_rome_targets()
        tests.test_summarize_edit_metrics_does_not_conflate_pairwise_and_top1(summarize_edit_metrics)
        """
    )
    md(
        details(
            "The fixture has 100% pairwise preference but only 50% target top-1, proving the metrics cannot collapse into one success rate.",
            "Use sequence log probabilities for pairwise comparison and the full distributions only for KL.",
            "A factual-locality denominator exists only for rows the unedited model answers top-1. Legitimate Rome answers are correctness checks, not leakage.",
            source("qualify_known_fact_mask", "summarize_edit_metrics"),
        )
    )

    md(
        exercise_header(
            12,
            "evaluate disjoint held-out suites",
            4,
            30,
            dedent(
                """
                Expose the complete evaluation path. Covariance uses `COVARIANCE_FACTS`; locality uses
                disjoint `KNOWN_FACT_CANDIDATES` and records its base-qualified denominator. Evaluate
                `ADVERSARIAL_RECORDS` by stratum, plus unrelated distributions, held-out perplexity texts,
                per-paraphrase rows, and optional deterministic generation probes. Implement the repetition
                metric as the fraction of token positions that repeat an earlier token. The three
                underscore-prefixed row formatters below are provided; edit the scorers and evaluator.
                """
            ),
        )
    )
    evaluate_stub = """
    def score_rewrite_prompts(
        model: Any,
        tokenizer: Any,
        prompts: Sequence[str],
        *,
        old_target: str = OLD_OBJECT,
        new_target: str = NEW_OBJECT,
    ) -> dict[str, Any]:
        raise NotImplementedError()


    def score_labeled_facts(
        model: Any,
        tokenizer: Any,
        facts: Sequence[tuple[str, str] | tuple[str, str, str]],
    ) -> dict[str, Any]:
        raise NotImplementedError()


    def score_adversarial_records(
        model: Any,
        tokenizer: Any,
        records: Sequence[AdversarialRecord],
    ) -> dict[str, Any]:
        raise NotImplementedError()


    def evaluate_weight_edit(
        model: Any,
        tokenizer: Any,
        *,
        layer: int,
        delta_weight: Tensor,
        rewrite_scorer: Callable[..., dict[str, Any]],
        fact_scorer: Callable[..., dict[str, Any]],
        adversarial_scorer: Callable[..., dict[str, Any]],
        metric_summarizer: Callable[..., dict[str, float | int]],
        paraphrase_records: Sequence[PromptRecord] = HELDOUT_PARAPHRASE_RECORDS,
        include_generations: bool = False,
    ) -> dict[str, Any]:
        raise NotImplementedError()
    """
    # Preserve helper availability in the learner while leaving its method stubs open.
    evaluate_stub = dedent(evaluate_stub).strip()
    provided_rows = source("_rewrite_rows", "_labeled_rows", "_adversarial_rows")
    repetition_stub = (
        "def continuation_repetition_fraction(tokenizer: Any, text: str) -> float:\n"
        "    raise NotImplementedError()"
    )
    evaluate_stub = evaluate_stub.replace(
        "def evaluate_weight_edit(",
        repetition_stub + "\n\n" + provided_rows + "\n\ndef evaluate_weight_edit(",
    )
    code(
        evaluate_stub,
        source(
            "score_rewrite_prompts",
            "score_labeled_facts",
            "score_adversarial_records",
            "continuation_repetition_fraction",
            "_rewrite_rows",
            "_labeled_rows",
            "_adversarial_rows",
            "evaluate_weight_edit",
        ),
    )
    code(
        """
        tests.test_covariance_and_evaluation_sets_are_disjoint_and_auditable()
        tests.test_real_evaluation_sizes_are_not_one_prompt_demos()
        tests.test_repetition_fraction_exposes_local_generation_collapse(tokenizer, continuation_repetition_fraction)
        tests.test_pinned_model_has_nontrivial_base_known_fact_denominator(model, tokenizer)
        tests.test_real_evaluator_zero_edit_has_immediate_numerical_contract(
            evaluate_weight_edit,
            model,
            tokenizer,
            score_rewrite_prompts,
            score_labeled_facts,
            score_adversarial_records,
            summarize_edit_metrics,
        )
        display(pd.DataFrame([record.__dict__ for record in HELDOUT_PARAPHRASE_RECORDS]))
        display(pd.DataFrame([record.__dict__ for record in ADVERSARIAL_RECORDS]))
        """
    )
    md(
        details(
            "The zero edit gives exactly 14 base-qualified facts, 100% retention, zero unrelated KL, and perplexity ratio 1.0.",
            "Cache base outputs before the temporary edit and return example-level rows alongside aggregates.",
            "These are course-authored probes, not a random population sample or a standard benchmark.",
            source(
                "score_rewrite_prompts",
                "score_labeled_facts",
                "score_adversarial_records",
                "continuation_repetition_fraction",
                "_rewrite_rows",
                "_labeled_rows",
                "_adversarial_rows",
                "evaluate_weight_edit",
            ),
        )
    )

    md(
        exercise_header(
            13,
            "compose the affine write and matched experiment",
            5,
            45,
            dedent(
                """
                Build the update from optimization keys, a disjoint covariance key bank, the complete
                captured module output, and the projection bias. Measure write error by temporarily
                editing the actual `dense_4h_to_h` module. Then run a norm-matched layer comparison, a
                wrong-position edit, norm-matched shuffled-city targets, and five random updates.
                """
            ),
        )
    )
    compose_stub = """
    def build_course_rome_like_update(
        model: Any,
        tokenizer: Any,
        *,
        layer: int,
        position_mode: str = "subject",
        targets: str | Sequence[str] = NEW_OBJECT,
        steps: int = 24,
        ridge: float = 1e-2,
    ) -> tuple[Tensor, dict[str, Any]]:
        raise NotImplementedError()


    def run_course_rome_like_experiment(
        *,
        implementation: LearnerFunctions,
        device: str | t.device = "cpu",
        steps: int = 12,
        patching_samples: int = 3,
    ) -> dict[str, Any]:
        raise NotImplementedError()
    """
    code(
        compose_stub,
        source("build_course_rome_like_update", "run_course_rome_like_experiment"),
    )
    code(
        """
        tests.test_real_update_builder_has_immediate_numerical_contract(
            build_course_rome_like_update, model, tokenizer
        )
        """
    )
    md(
        details(
            "The actual affine module output matches the optimized value to numerical precision and the applied matrix has effective rank one.",
            "Transpose Pythia's stored `[d_value, d_key]` weight once into row-vector convention; pass its bias into the write equation.",
            "Exact insertion is necessary. The held-out rows and controls decide whether the inserted association is behaviorally selective.",
            source("build_course_rome_like_update", "run_course_rome_like_experiment"),
        )
    )

    md(
        """
        ## 6. Signature result

        This run deliberately avoids a binary “knowledge edited” score. The headline is the negative
        localization result: multiple norm-matched layers and the wrong-position edit rewrite the target.
        The same result reports base-qualified fact retention, leakage-only adversarial strata, unrelated
        KL, held-out perplexity, and the large value-optimizer KL.
        """
    )
    code(
        """
        learner_functions = LearnerFunctions(
            activation_patching_sweep=activation_patching_sweep,
            build_update=build_course_rome_like_update,
            evaluate_edit=evaluate_weight_edit,
            rewrite_scorer=score_rewrite_prompts,
            fact_scorer=score_labeled_facts,
            adversarial_scorer=score_adversarial_records,
            metric_summarizer=summarize_edit_metrics,
            random_rank_one=make_same_norm_random_rank_one,
        )
        tests.check_learner_bundle_uses_notebook_functions(learner_functions)
        result = run_course_rome_like_experiment(
            implementation=learner_functions,
            device=device,
            steps=12,
            patching_samples=3,
        )
        primary_metrics = result["metrics"]["course_rome_like"]
        print("NEGATIVE LOCALIZATION RESULT", result["localization_result"])
        assert result["localization_result"]["negative_localization_result"]
        display(pd.DataFrame(result["layer_sweep"])[[
            "layer",
            "pairwise_preference_rate_after",
            "target_top1_rate_after",
            "mean_full_target_likelihood_after",
            "known_fact_top1_retention_after",
            "side_effect_mean_kl",
            "fluency_perplexity_ratio",
            "native_update_norm",
            "applied_update_norm",
            "native_write_error",
        ]])
        display(pd.DataFrame(result["metrics"]).T[[
            "pairwise_preference_rate_after",
            "target_top1_rate_after",
            "mean_full_target_likelihood_after",
            "known_fact_top1_retention_after",
            "leakage_sensitive_mean_rome_probability_change",
            "side_effect_mean_kl",
            "fluency_perplexity_ratio",
        ]])
        print({
            "selected_layer": result["selected_layer"],
            "condition_best_layers": result["trace"]["condition_best_layers"].tolist(),
            "independent_direction_seeds": result["trace"]["direction_seeds"],
            "layer_0_mean": result["trace"]["layer_zero_mean_recovery"],
            "layer_0_std": result["trace"]["layer_zero_std_recovery"],
            "actual_affine_write_error": result["primary_details"]["write_error"],
            "effective_rank": result["primary_details"]["effective_rank"],
            "value_optimizer_preservation_KL": result["primary_details"]["optimization_history"][-1]["preservation_kl"],
            "value_optimizer_KL_is_large": result["primary_details"]["optimization_history"][-1]["preservation_kl"] > 0.1,
        })

        paraphrase_table = pd.DataFrame(result["primary_evaluation"]["paraphrase_rows"])
        display(paraphrase_table[[
            "prompt_id", "family", "prompt", "base_top_token", "edited_top_token",
            "base_target_rank", "edited_target_rank", "base_new_likelihood",
            "edited_new_likelihood", "pairwise_after", "target_top1_after",
        ]])
        known_fact_table = pd.DataFrame(result["primary_evaluation"]["known_fact_rows"])
        display(known_fact_table[known_fact_table["base_qualified_known_fact"]])
        adversarial_table = pd.DataFrame(result["primary_evaluation"]["adversarial_rows"])
        display(adversarial_table[[
            "prompt_id", "stratum", "counts_as_leakage", "prompt", "label",
            "base_label_rank", "edited_label_rank", "rome_probability_change",
        ]])
        display(pd.DataFrame(result["primary_evaluation"]["generation_rows"]))
        print({
            "localized_louvre_repetition_before": primary_metrics["localized_louvre_repetition_before"],
            "localized_louvre_repetition_after": primary_metrics["localized_louvre_repetition_after"],
            "generation_repetition_failure_count": primary_metrics["generation_repetition_failure_count"],
            "scope": "one localized generation failure; not a general fluency conclusion",
            "leakage_only_rome_probability_change": primary_metrics["leakage_sensitive_mean_rome_probability_change"],
        })

        reference.plot_signature_result(result, str(assets_dir / "rome_model_editing_signature.png"))
        plt.show()

        summary = reference.serializable_experiment_summary(result)
        (section_dir / "expected_outputs" / "reference_metrics.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\\n"
        )
        """
    )
    md(
        """
        <details>
        <summary>How to interpret the live result</summary>

        1. `pairwise_preference_rate_after` means only that Rome outranks Paris.
        2. `target_top1_rate_after` asks whether Rome is the model's actual first choice.
        3. `mean_full_target_likelihood_after` scores every target token.
        4. Known-fact retention uses only the 14/24 rows answered top-1 before editing; unqualified rows are never a factual-locality denominator.
        5. All nine activation-patching conditions use different Gaussian directions.
        6. Every layer and wrong-position comparison uses the same update norm. Their broad success is a negative localization result, so no layer is privileged.
        7. Leakage excludes `target_generalization` and `legitimate_rome_target` rows by construction.
        8. The value optimizer's preservation KL exceeds the 0.1 warning threshold and is reported as large.
        9. Repetition increased on one Louvre generation. That is one localized generation failure, not evidence of general fluency loss.

        </details>
        """
    )

    md(
        """
        ## Try It Yourself

        This is a **scoring-only** play cell for the already-built fixed Louvre-to-Rome edit.
        Changing `target` changes only what the scorer measures. It does not optimize a new value,
        alter the fixed edit, or implement a multi-token edit.
        """
    )
    code(
        """
        def score_fixed_louvre_edit(
            prompt: str,
            target: str = NEW_OBJECT,
            scales: Sequence[float] = (0.0, 0.5, 1.0, 1.5),
        ) -> pd.DataFrame:
            rows = []
            for scale in scales:
                with temporary_mlp_weight_edit(
                    model,
                    result["selected_layer"],
                    result["primary_update"] * scale,
                ):
                    scored = continuation_score_table(model, tokenizer, [prompt], [target])
                rows.append({
                    "scale": scale,
                    "fixed_edit": "Louvre -> Rome",
                    "scored_target_only": target,
                    "target_token_count": int(scored["target_token_counts"][0]),
                    "full_target_likelihood": float(scored["target_likelihoods"][0]),
                    "first_target_rank": int(scored["first_target_ranks"][0]),
                    "full_target_greedy_match": bool(scored["full_target_greedy_matches"][0]),
                })
            return pd.DataFrame(rows)


        play = score_fixed_louvre_edit("A first-time visitor can see the Louvre in")
        display(play)
        play.plot(x="scale", y="full_target_likelihood", marker="o")
        plt.show()

        # Scoring-only multi-token diagnostic: this still applies the fixed Louvre -> Rome matrix.
        # It does not train or claim a New York edit.
        display(score_fixed_louvre_edit("Times Square is in", target=" New York", scales=(0.0, 1.0)))
        """
    )

    md(
        """
        ## Further research / anomaly hunting

        - Add more harmless counterfactuals and report a distribution, not another anecdote.
        - Establish official parity separately with Wikipedia-derived covariance, random-prefix contexts,
          and the released value objective before using an unqualified ROME label.
        - Increase activation-patching samples and test other corruption families.
        - Expand the predeclared candidate pool before qualifying base-known facts.
        - Repeat edits sequentially and measure whether earlier edits and broad language metrics survive.
        - Test whether the negative localization result persists across models and facts.

        ## Limitations

        This notebook uses one harmless counterfactual, one small pretrained decoder, 24 covariance
        contexts, 20 held-out paraphrases, 24 known-fact candidates, 16 stratified adversarial contexts,
        and short course-authored perplexity probes. Only base-top-1 candidate facts enter the retention
        denominator. Activation patching is not protocol-parity causal tracing. The course edit omits
        Wikipedia covariance, random-prefix contexts, and released value-objective parity. Its large
        optimizer KL, broad norm-matched layer success, wrong-position success, and one repetitive Louvre
        continuation are negative evidence. The repetition example is not a general fluency claim. The lab
        does not establish durable knowledge change, unique factual storage, repeated-edit safety, or
        benchmark-level ROME performance.

        ## Reading

        - Meng et al., *Locating and Editing Factual Associations in GPT* and the official ROME code.
        - Hase et al., *Does Localization Inform Editing?*
        - The original ARENA Function Vectors and Model Steering section for intervention pedagogy.
        """
    )
    code(
        """
        tests.test_smoke_contract_is_exact(reference.run_smoke_test)
        tests.test_model_and_revision_are_pinned()
        tests.test_section_local_release_report_generation_inputs_are_strict()
        tests.test_notebook_contract()
        """
    )
    return cells


def to_markdown(cells: list) -> str:
    blocks = []
    for cell in cells:
        if cell.cell_type == "markdown":
            blocks.append(cell.source.rstrip())
        elif cell.cell_type == "code":
            blocks.append(f"```python\n{cell.source.rstrip()}\n```")
    return "\n\n\n".join(blocks).rstrip() + "\n"


def write_notebook(path: Path, cells: list) -> None:
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
    )
    for index, cell in enumerate(notebook.cells):
        cell.id = f"rome-{index:03d}"
        if cell.cell_type == "code":
            ast.parse(cell.source)
    nbformat.write(notebook, path)


def main() -> None:
    exercise_cells = build_cells(solution=False)
    solution_cells = build_cells(solution=True)
    INSTRUCTION_PAGE.write_text(to_markdown(exercise_cells))
    write_notebook(EXERCISE_NOTEBOOK, exercise_cells)
    write_notebook(SOLUTION_NOTEBOOK, solution_cells)
    print(
        json.dumps(
            {
                "instruction_page": str(INSTRUCTION_PAGE),
                "exercise_notebook": str(EXERCISE_NOTEBOOK),
                "solution_notebook": str(SOLUTION_NOTEBOOK),
                "cell_count": len(exercise_cells),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
