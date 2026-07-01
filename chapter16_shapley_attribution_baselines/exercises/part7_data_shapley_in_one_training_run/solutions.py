# %%
"""Reference solutions for [16.7] Data Shapley in One Training Run."""

import itertools
import sys
import time
from pathlib import Path

import torch as t

chapter = "chapter16_shapley_attribution_baselines"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from arena_ext.shapley_attribution import (
    data_coalition_values,
    data_shapley_report,
    exact_data_shapley_values,
    exact_shapley_values,
    in_run_data_shapley_report,
    in_run_first_order_data_scores,
    monte_carlo_data_shapley_report,
    pearson_correlation,
    sampled_permutation_shapley_values,
    toy_data_shapley_problem,
    one_step_linear_utility,
)

MAIN = __name__ == "__main__"
DATA_SHAPLEY_LR = 0.5
DATA_SHAPLEY_MC_SAMPLES = 512
DATA_SHAPLEY_MC_MAX_ERROR = 0.08
DATA_SHAPLEY_MIN_CORRELATION = 0.99
DATA_SHAPLEY_RANDOM_CONTROL_SEED = 13
DATA_SHAPLEY_RANDOM_CONTROL_MAX_ABS_CORRELATION = 0.25
DATA_SHAPLEY_LABEL_SHUFFLE_MAX_CORRELATION = 0.0
DATA_SHAPLEY_LABEL_SHUFFLE_PERMUTATION = (0, 3, 2, 1)
DATA_SHAPLEY_RUNTIME_REPEATS = 128
DATA_SHAPLEY_RUNTIME_IN_RUN_VS_EXACT_MAX_RATIO = 2.0


# %%
def _tensor_report(report) -> dict:
    result = report.__dict__.copy()
    for key, value in list(result.items()):
        if hasattr(value, "tolist"):
            result[key] = value.tolist()
    return result


def _to_serializable_metrics(metrics: dict) -> dict:
    result = metrics.copy()
    for key, value in list(result.items()):
        if hasattr(value, "tolist"):
            result[key] = value.tolist()
    return result


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, 1e-12)


def _measure_wall_seconds(fn, *, repeats: int) -> float:
    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def _random_data_shapley_problem() -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    generator = t.Generator().manual_seed(DATA_SHAPLEY_RANDOM_CONTROL_SEED)
    train_x = t.randn(4, 3, generator=generator, dtype=t.float64)
    train_y = t.randn(4, generator=generator, dtype=t.float64)
    val_x = t.randn(2, 3, generator=generator, dtype=t.float64)
    val_y = t.randn(2, generator=generator, dtype=t.float64)
    return train_x, train_y, val_x, val_y


def _label_shuffled_data_shapley_problem() -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    train_x, train_y, val_x, val_y = toy_data_shapley_problem()
    permutation = t.tensor(DATA_SHAPLEY_LABEL_SHUFFLE_PERMUTATION, dtype=t.long)
    return train_x, train_y[permutation], val_x, val_y


def _signal_failure_metrics(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
    *,
    signal_exact: t.Tensor,
) -> dict:
    exact = exact_data_shapley_values(train_x, train_y, val_x, val_y)
    scores = in_run_first_order_data_scores(train_x, train_y, val_x, val_y)
    exact_signal_correlation = pearson_correlation(signal_exact, exact)
    in_run_signal_correlation = pearson_correlation(signal_exact, scores)
    return {
        "exact_values": exact,
        "in_run_scores": scores,
        "harmful_index": int(exact.argmin().item()),
        "helpful_index": int(exact.argmax().item()),
        "in_run_harmful_index": int(scores.argmin().item()),
        "in_run_helpful_index": int(scores.argmax().item()),
        "signal_correlation": exact_signal_correlation,
        "in_run_signal_correlation": in_run_signal_correlation,
        "max_abs_signal_correlation": max(
            abs(exact_signal_correlation),
            abs(in_run_signal_correlation),
        ),
    }


def random_data_attribution_failure_smoke_test() -> dict:
    signal_exact = exact_data_shapley_values(*toy_data_shapley_problem())
    train_x, train_y, val_x, val_y = _random_data_shapley_problem()
    metrics = _signal_failure_metrics(
        train_x,
        train_y,
        val_x,
        val_y,
        signal_exact=signal_exact,
    )
    original_harmful_index = int(signal_exact.argmin().item())
    original_helpful_index = int(signal_exact.argmax().item())
    metrics["random_data_attribution_fails"] = (
        metrics["harmful_index"] != original_harmful_index
        and metrics["helpful_index"] != original_helpful_index
        and metrics["max_abs_signal_correlation"]
        <= DATA_SHAPLEY_RANDOM_CONTROL_MAX_ABS_CORRELATION
    )
    metrics["original_harmful_index"] = original_harmful_index
    metrics["original_helpful_index"] = original_helpful_index
    metrics["seed"] = DATA_SHAPLEY_RANDOM_CONTROL_SEED
    return _to_serializable_metrics(metrics)


def label_shuffled_attribution_failure_smoke_test() -> dict:
    signal_exact = exact_data_shapley_values(*toy_data_shapley_problem())
    train_x, train_y, val_x, val_y = _label_shuffled_data_shapley_problem()
    metrics = _signal_failure_metrics(
        train_x,
        train_y,
        val_x,
        val_y,
        signal_exact=signal_exact,
    )
    original_harmful_index = int(signal_exact.argmin().item())
    metrics["label_shuffled_attribution_fails"] = (
        metrics["harmful_index"] != original_harmful_index
        and metrics["in_run_harmful_index"] != original_harmful_index
        and metrics["signal_correlation"] <= DATA_SHAPLEY_LABEL_SHUFFLE_MAX_CORRELATION
        and metrics["in_run_signal_correlation"] <= DATA_SHAPLEY_LABEL_SHUFFLE_MAX_CORRELATION
    )
    metrics["original_harmful_index"] = original_harmful_index
    metrics["label_permutation"] = list(DATA_SHAPLEY_LABEL_SHUFFLE_PERMUTATION)
    metrics["shuffled_train_y"] = train_y.tolist()
    return _to_serializable_metrics(metrics)


def runtime_overhead_smoke_test(repeats: int = DATA_SHAPLEY_RUNTIME_REPEATS) -> dict:
    train_x, train_y, val_x, val_y = toy_data_shapley_problem()
    full = frozenset(range(int(train_x.shape[0])))
    full_update_seconds = _measure_wall_seconds(
        lambda: one_step_linear_utility(train_x, train_y, val_x, val_y, full),
        repeats=repeats,
    )
    exact_enumeration_seconds = _measure_wall_seconds(
        lambda: data_coalition_values(train_x, train_y, val_x, val_y),
        repeats=repeats,
    )
    in_run_scores_seconds = _measure_wall_seconds(
        lambda: in_run_first_order_data_scores(train_x, train_y, val_x, val_y),
        repeats=repeats,
    )
    return {
        "runtime_measurement_repeats": repeats,
        "runtime_full_update_seconds": full_update_seconds,
        "runtime_exact_enumeration_seconds": exact_enumeration_seconds,
        "runtime_in_run_scores_seconds": in_run_scores_seconds,
        "runtime_exact_vs_full_update_overhead_ratio": _ratio(
            exact_enumeration_seconds,
            full_update_seconds,
        ),
        "runtime_in_run_vs_full_update_overhead_ratio": _ratio(
            in_run_scores_seconds,
            full_update_seconds,
        ),
        "runtime_in_run_vs_exact_ratio": _ratio(
            in_run_scores_seconds,
            exact_enumeration_seconds,
        ),
        "runtime_overhead_reported": (
            full_update_seconds > 0.0
            and exact_enumeration_seconds > 0.0
            and in_run_scores_seconds > 0.0
        ),
    }


def exact_data_shapley_smoke_test() -> dict:
    train_x, train_y, val_x, val_y = toy_data_shapley_problem()
    return _tensor_report(data_shapley_report(train_x, train_y, val_x, val_y))


def monte_carlo_data_shapley_smoke_test() -> dict:
    train_x, train_y, val_x, val_y = toy_data_shapley_problem()
    return _tensor_report(
        monte_carlo_data_shapley_report(
            train_x,
            train_y,
            val_x,
            val_y,
            num_samples=512,
            seed=0,
        )
    )


def in_run_data_shapley_smoke_test() -> dict:
    train_x, train_y, val_x, val_y = toy_data_shapley_problem()
    return _tensor_report(in_run_data_shapley_report(train_x, train_y, val_x, val_y))


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "exact": exact_data_shapley_smoke_test(),
        "monte_carlo": monte_carlo_data_shapley_smoke_test(),
        "in_run": in_run_data_shapley_smoke_test(),
        "random_data_control": random_data_attribution_failure_smoke_test(),
        "label_shuffle_control": label_shuffled_attribution_failure_smoke_test(),
        "runtime_overhead": runtime_overhead_smoke_test(),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    if not t.cuda.is_available():
        raise RuntimeError("16.7 GPU preflight requires CUDA; no CPU fallback is accepted.")

    return run_cuda_data_shapley_preflight(max_vram_gb=max_vram_gb)


def _cuda_one_step_utility(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
    coalition: frozenset[int],
) -> float:
    weight = t.zeros(train_x.shape[1], dtype=t.float64, device=train_x.device)
    baseline_loss = ((val_x @ weight - val_y) ** 2).mean()
    if not coalition:
        return 0.0
    indices = t.tensor(sorted(coalition), dtype=t.long, device=train_x.device)
    selected_x = train_x[indices]
    selected_y = train_y[indices]
    train_error = selected_x @ weight - selected_y
    gradient = (2 * train_error.unsqueeze(-1) * selected_x).mean(dim=0)
    updated_weight = weight - DATA_SHAPLEY_LR * gradient
    updated_loss = ((val_x @ updated_weight - val_y) ** 2).mean()
    return float((baseline_loss - updated_loss).item())


def _cuda_data_coalition_values(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
) -> dict[frozenset[int], float]:
    num_examples = int(train_x.shape[0])
    return {
        frozenset(group): _cuda_one_step_utility(train_x, train_y, val_x, val_y, frozenset(group))
        for size in range(num_examples + 1)
        for group in itertools.combinations(range(num_examples), size)
    }


def _cuda_in_run_gradient_scores(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
) -> t.Tensor:
    weight = t.nn.Parameter(t.zeros(train_x.shape[1], dtype=t.float64, device=train_x.device))
    val_loss = ((val_x @ weight - val_y) ** 2).mean()
    val_gradient = t.autograd.grad(val_loss, weight, retain_graph=True)[0].detach()
    scores = []
    for index in range(int(train_x.shape[0])):
        train_loss = ((train_x[index : index + 1] @ weight - train_y[index : index + 1]) ** 2).mean()
        train_gradient = t.autograd.grad(train_loss, weight, retain_graph=True)[0].detach()
        scores.append(float((train_gradient * val_gradient).sum().item()))
    return t.tensor(scores, dtype=t.float64)


def _actual_full_batch_one_step_utility(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
) -> float:
    model = t.nn.Linear(1, 1, bias=False, dtype=t.float64).to(train_x.device)
    model.weight.data.zero_()
    optimizer = t.optim.SGD(model.parameters(), lr=DATA_SHAPLEY_LR)
    baseline_loss = ((model(val_x).flatten() - val_y) ** 2).mean()
    train_loss = ((model(train_x).flatten() - train_y) ** 2).mean()
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    updated_loss = ((model(val_x).flatten() - val_y) ** 2).mean()
    return float((baseline_loss - updated_loss).item())


def _move_problem_to_device(
    problem: tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor],
    device: t.device,
) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    return tuple(item.to(device=device, dtype=t.float64) for item in problem)


def _cuda_signal_failure_metrics(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
    *,
    signal_exact: t.Tensor,
) -> dict:
    values = _cuda_data_coalition_values(train_x, train_y, val_x, val_y)
    exact = exact_shapley_values(values, num_players=int(train_x.shape[0]))
    scores = _cuda_in_run_gradient_scores(train_x, train_y, val_x, val_y)
    exact_signal_correlation = pearson_correlation(signal_exact, exact)
    in_run_signal_correlation = pearson_correlation(signal_exact, scores)
    return {
        "exact_values": exact.tolist(),
        "in_run_scores": scores.tolist(),
        "harmful_index": int(exact.argmin().item()),
        "helpful_index": int(exact.argmax().item()),
        "in_run_harmful_index": int(scores.argmin().item()),
        "in_run_helpful_index": int(scores.argmax().item()),
        "signal_correlation": exact_signal_correlation,
        "in_run_signal_correlation": in_run_signal_correlation,
        "max_abs_signal_correlation": max(
            abs(exact_signal_correlation),
            abs(in_run_signal_correlation),
        ),
    }


def _measure_cuda_seconds(fn, *, repeats: int) -> float:
    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    fn()
    t.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    t.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def _cuda_runtime_overhead_metrics(
    train_x: t.Tensor,
    train_y: t.Tensor,
    val_x: t.Tensor,
    val_y: t.Tensor,
    *,
    repeats: int = DATA_SHAPLEY_RUNTIME_REPEATS,
) -> dict:
    full_update_seconds = _measure_cuda_seconds(
        lambda: _actual_full_batch_one_step_utility(train_x, train_y, val_x, val_y),
        repeats=repeats,
    )
    exact_enumeration_seconds = _measure_cuda_seconds(
        lambda: _cuda_data_coalition_values(train_x, train_y, val_x, val_y),
        repeats=repeats,
    )
    in_run_scores_seconds = _measure_cuda_seconds(
        lambda: _cuda_in_run_gradient_scores(train_x, train_y, val_x, val_y),
        repeats=repeats,
    )
    return {
        "runtime_measurement_repeats": repeats,
        "runtime_full_update_seconds": full_update_seconds,
        "runtime_exact_enumeration_seconds": exact_enumeration_seconds,
        "runtime_in_run_scores_seconds": in_run_scores_seconds,
        "runtime_exact_vs_full_update_overhead_ratio": _ratio(
            exact_enumeration_seconds,
            full_update_seconds,
        ),
        "runtime_in_run_vs_full_update_overhead_ratio": _ratio(
            in_run_scores_seconds,
            full_update_seconds,
        ),
        "runtime_in_run_vs_exact_ratio": _ratio(
            in_run_scores_seconds,
            exact_enumeration_seconds,
        ),
        "runtime_in_run_faster_than_exact": in_run_scores_seconds
        < exact_enumeration_seconds,
        "runtime_overhead_reported": (
            full_update_seconds > 0.0
            and exact_enumeration_seconds > 0.0
            and in_run_scores_seconds > 0.0
        ),
    }


def run_cuda_data_shapley_preflight(max_vram_gb: float = 24.0) -> dict:
    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    train_x, train_y, val_x, val_y = _move_problem_to_device(
        toy_data_shapley_problem(),
        device,
    )

    values = _cuda_data_coalition_values(train_x, train_y, val_x, val_y)
    exact = exact_shapley_values(values, num_players=int(train_x.shape[0]))
    sampled = sampled_permutation_shapley_values(
        values,
        num_players=int(train_x.shape[0]),
        num_samples=DATA_SHAPLEY_MC_SAMPLES,
        seed=0,
    )
    sampled_max_error = float((exact - sampled).abs().max().item())
    gradient_scores = _cuda_in_run_gradient_scores(train_x, train_y, val_x, val_y)
    correlation = pearson_correlation(exact, gradient_scores)
    harmful_index = int(exact.argmin().item())
    helpful_index = int(exact.argmax().item())
    in_run_harmful_index = int(gradient_scores.argmin().item())
    in_run_helpful_index = int(gradient_scores.argmax().item())
    full = frozenset(range(int(train_x.shape[0])))
    harmful_removal_delta = values[full - {harmful_index}] - values[full]
    actual_full_utility = _actual_full_batch_one_step_utility(train_x, train_y, val_x, val_y)
    random_train_x, random_train_y, random_val_x, random_val_y = _move_problem_to_device(
        _random_data_shapley_problem(),
        device,
    )
    random_data_control = _cuda_signal_failure_metrics(
        random_train_x,
        random_train_y,
        random_val_x,
        random_val_y,
        signal_exact=exact,
    )
    random_data_control["random_data_attribution_fails"] = (
        random_data_control["harmful_index"] != harmful_index
        and random_data_control["helpful_index"] != helpful_index
        and random_data_control["max_abs_signal_correlation"]
        <= DATA_SHAPLEY_RANDOM_CONTROL_MAX_ABS_CORRELATION
    )
    label_train_x, label_train_y, label_val_x, label_val_y = _move_problem_to_device(
        _label_shuffled_data_shapley_problem(),
        device,
    )
    label_shuffle_control = _cuda_signal_failure_metrics(
        label_train_x,
        label_train_y,
        label_val_x,
        label_val_y,
        signal_exact=exact,
    )
    label_shuffle_control["label_shuffled_attribution_fails"] = (
        label_shuffle_control["harmful_index"] != harmful_index
        and label_shuffle_control["in_run_harmful_index"] != harmful_index
        and label_shuffle_control["signal_correlation"]
        <= DATA_SHAPLEY_LABEL_SHUFFLE_MAX_CORRELATION
        and label_shuffle_control["in_run_signal_correlation"]
        <= DATA_SHAPLEY_LABEL_SHUFFLE_MAX_CORRELATION
    )
    runtime_metrics = _cuda_runtime_overhead_metrics(train_x, train_y, val_x, val_y)

    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    preflight_passed = (
        harmful_index == 3
        and float(exact[harmful_index].item()) < 0.0
        and harmful_removal_delta > 0.0
        and sampled_max_error <= DATA_SHAPLEY_MC_MAX_ERROR
        and correlation >= DATA_SHAPLEY_MIN_CORRELATION
        and in_run_harmful_index == harmful_index
        and in_run_helpful_index == helpful_index
        and abs(actual_full_utility - values[full]) <= 1e-9
        and random_data_control["random_data_attribution_fails"]
        and label_shuffle_control["label_shuffled_attribution_fails"]
        and runtime_metrics["runtime_overhead_reported"]
        and runtime_metrics["runtime_in_run_vs_exact_ratio"]
        <= DATA_SHAPLEY_RUNTIME_IN_RUN_VS_EXACT_MAX_RATIO
        and peak_vram_gb <= max_vram_gb
    )
    return {
        "preflight_passed": preflight_passed,
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "model_family": "cuda_one_step_linear_regression_data_shapley",
        "training_example_count": int(train_x.shape[0]),
        "coalition_count": 2 ** int(train_x.shape[0]),
        "monte_carlo_samples": DATA_SHAPLEY_MC_SAMPLES,
        "learning_rate": DATA_SHAPLEY_LR,
        "exact_values": exact.tolist(),
        "sampled_values": sampled.tolist(),
        "sampled_max_abs_error": sampled_max_error,
        "sampled_approximates_exact": sampled_max_error <= DATA_SHAPLEY_MC_MAX_ERROR,
        "gradient_scores": gradient_scores.tolist(),
        "pearson_correlation": correlation,
        "harmful_index": harmful_index,
        "helpful_index": helpful_index,
        "in_run_harmful_index": in_run_harmful_index,
        "in_run_helpful_index": in_run_helpful_index,
        "identifies_harmful": in_run_harmful_index == harmful_index,
        "identifies_helpful": in_run_helpful_index == helpful_index,
        "harmful_value": float(exact[harmful_index].item()),
        "harmful_removal_delta": harmful_removal_delta,
        "actual_full_batch_one_step_utility": actual_full_utility,
        "coalition_full_utility": values[full],
        "random_data_exact_values": random_data_control["exact_values"],
        "random_data_in_run_scores": random_data_control["in_run_scores"],
        "random_data_harmful_index": random_data_control["harmful_index"],
        "random_data_helpful_index": random_data_control["helpful_index"],
        "random_data_in_run_harmful_index": random_data_control["in_run_harmful_index"],
        "random_data_in_run_helpful_index": random_data_control["in_run_helpful_index"],
        "random_data_signal_correlation": random_data_control["signal_correlation"],
        "random_data_in_run_signal_correlation": random_data_control[
            "in_run_signal_correlation"
        ],
        "random_data_max_abs_signal_correlation": random_data_control[
            "max_abs_signal_correlation"
        ],
        "random_data_attribution_fails": random_data_control[
            "random_data_attribution_fails"
        ],
        "label_shuffled_exact_values": label_shuffle_control["exact_values"],
        "label_shuffled_in_run_scores": label_shuffle_control["in_run_scores"],
        "label_shuffled_harmful_index": label_shuffle_control["harmful_index"],
        "label_shuffled_helpful_index": label_shuffle_control["helpful_index"],
        "label_shuffled_in_run_harmful_index": label_shuffle_control[
            "in_run_harmful_index"
        ],
        "label_shuffled_in_run_helpful_index": label_shuffle_control[
            "in_run_helpful_index"
        ],
        "label_shuffled_signal_correlation": label_shuffle_control[
            "signal_correlation"
        ],
        "label_shuffled_in_run_signal_correlation": label_shuffle_control[
            "in_run_signal_correlation"
        ],
        "label_shuffled_max_abs_signal_correlation": label_shuffle_control[
            "max_abs_signal_correlation"
        ],
        "label_shuffled_attribution_fails": label_shuffle_control[
            "label_shuffled_attribution_fails"
        ],
        "runtime_measurement_repeats": runtime_metrics["runtime_measurement_repeats"],
        "runtime_full_update_seconds": runtime_metrics["runtime_full_update_seconds"],
        "runtime_exact_enumeration_seconds": runtime_metrics[
            "runtime_exact_enumeration_seconds"
        ],
        "runtime_in_run_scores_seconds": runtime_metrics["runtime_in_run_scores_seconds"],
        "runtime_exact_vs_full_update_overhead_ratio": runtime_metrics[
            "runtime_exact_vs_full_update_overhead_ratio"
        ],
        "runtime_in_run_vs_full_update_overhead_ratio": runtime_metrics[
            "runtime_in_run_vs_full_update_overhead_ratio"
        ],
        "runtime_in_run_vs_exact_ratio": runtime_metrics["runtime_in_run_vs_exact_ratio"],
        "runtime_in_run_faster_than_exact": runtime_metrics[
            "runtime_in_run_faster_than_exact"
        ],
        "runtime_overhead_reported": runtime_metrics["runtime_overhead_reported"],
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "full_path": (
            "Enumerate Data Shapley coalitions on CUDA, run an actual one-step "
            "linear training update, compare exact, Monte Carlo, and in-run "
            "autograd gradient-dot scores, run random-data and label-shuffle "
            "failure controls, and measure runtime overhead."
        ),
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
