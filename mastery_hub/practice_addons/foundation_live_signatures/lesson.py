"""Append-only live plots for blank foundation Practice copies.

The provider is inert and stdlib-only. Core methods remain the learner's own
notebook definitions. These cells add no execution or report acceptance claim.
"""
from textwrap import dedent


_LEAKAGE_PLOT = '''
import matplotlib.pyplot as plt

# Literal CPU fixture: the second score column is the first column inverted.
live06_labels = t.tensor([0, 1, 0, 1, 0, 1, 0, 1], device="cpu")
live06_features = t.tensor([
    [-1, 1], [1, -1], [-1, 1], [1, -1],
    [-1, 1], [1, -1], [-1, 1], [1, -1],
], dtype=t.float64, device="cpu")
live06_effects = t.tensor([1., 1., 1., 1., 1., 1., 1., 1., 9., 9.],
                          dtype=t.float64, device="cpu")
live06_leakage = label_leakage_report(live06_features, live06_labels)
live06_selected = cherry_pick_report(live06_effects, selected_indices=[8, 9])
live06_representative = cherry_pick_report(live06_effects, selected_indices=[0, 1, 2, 3, 8])

live06_fig, live06_axes = plt.subplots(1, 2, figsize=(11, 4))
live06_accuracy_bars = live06_axes[0].bar(
    ["Label-encoding feature", "Inverted / anti-correlated"],
    [live06_leakage.leaked_feature_accuracy, live06_leakage.shifted_no_leak_accuracy],
    color=["#147D7E", "#C73E3A"],
)
live06_axes[0].bar_label(live06_accuracy_bars, fmt="%.2f", padding=3)
live06_axes[0].set_ylim(0, 1.12)
live06_axes[0].set_ylabel("Accuracy (threshold at zero)")
live06_axes[0].set_title("A high score can come from a leaked label")
live06_axes[0].tick_params(axis="x", labelrotation=12)
live06_axes[1].bar(
    ["Selected tail", "Representative subset", "Population"],
    [live06_selected.selected_mean_effect, live06_representative.selected_mean_effect,
     live06_selected.population_mean_effect],
    color=["#C73E3A", "#147D7E", "#64748B"],
)
live06_axes[1].set_ylabel("Mean effect (illustrative units)")
live06_axes[1].set_title("Which examples did the headline leave out?")
live06_axes[1].tick_params(axis="x", labelrotation=12)
live06_fig.tight_layout()
plt.show()
'''

_LEAKAGE_PLAY = '''
# Predict the change before running. Change only this selection, then rerun.
live06_play_indices = [0, 1, 8, 9]
live06_play = cherry_pick_report(live06_effects, selected_indices=live06_play_indices)

live06_play_fig, live06_play_ax = plt.subplots(figsize=(7, 4))
live06_play_ax.bar(
    ["Original tail", "Your selection", "Same population"],
    [live06_selected.selected_mean_effect, live06_play.selected_mean_effect,
     live06_play.population_mean_effect],
    color=["#C73E3A", "#147D7E", "#64748B"],
)
live06_play_ax.set_ylabel("Mean effect (illustrative units)")
live06_play_ax.set_title(f"Selection changed to indices {live06_play_indices}")
live06_play_fig.tight_layout()
plt.show()
'''

_MEMORY_PLOT = '''
import matplotlib.pyplot as plt

# Illustrative tensor counts, not a named checkpoint or allocated tensors.
live16_batch_size = 1  # Change only this to 2 in the play cell.
live16_contexts = [128, 512, 2048, 8192]
live16_assumptions = dict(
    num_parameters=1_000_000_000, dtype="bfloat16", batch_size=live16_batch_size,
    hidden_size=1024, activation_multiplier=2.0, overhead_gb=0.25,
)
live16_full_budgets = []
live16_omitted_kv_budgets = []
for live16_context in live16_contexts:
    live16_full_budgets.append(estimate_inference_memory(
        **live16_assumptions, context_length=live16_context,
        num_layers=16, num_key_value_heads=8, head_dim=128,
    ))
    # Deliberate input omission: the same estimator receives no KV dimensions.
    live16_omitted_kv_budgets.append(estimate_inference_memory(
        **live16_assumptions, context_length=live16_context,
    ))

live16_fig, live16_ax = plt.subplots(figsize=(9, 5))
for live16_field, live16_label in (
    ("parameter_gb", "Parameters"), ("kv_cache_gb", "KV cache"),
    ("activation_gb", "Activations"), ("total_gb", "Total"),
):
    live16_ax.plot(live16_contexts,
                   [getattr(budget, live16_field) for budget in live16_full_budgets],
                   marker="o", label=live16_label)
live16_ax.plot(live16_contexts,
               [budget.total_gb for budget in live16_omitted_kv_budgets],
               marker="x", linestyle="--", label="Omitted KV (underestimate)")
live16_ax.axhline(3.0, color="#C73E3A", linestyle=":", label="Illustrative 3 GiB budget")
live16_ax.set_xscale("log", base=2)
live16_ax.set_xlabel("Context length (tokens)")
live16_ax.set_ylabel("Estimated memory (GiB; 1024**3 bytes)")
live16_ax.set_title(f"Illustrative inference accounting: batch size {live16_batch_size}")
live16_ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
live16_fig.tight_layout()
plt.show()
'''


def _cell(section: str, suffix: str, kind: str, source: str) -> dict:
    cell = {"cell_type": kind, "id": f"foundation-live-{section.replace('.', '-')}-{suffix}",
            "metadata": {}, "source": dedent(source).strip() + "\n"}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def additions(section: str) -> list[dict]:
    """Return six fresh unexecuted cells for 0.6 or 1.6; never touch old cells."""
    if section == "0.6":
        forecast = '''
        ## Live signature — can your own detectors expose the shortcut?

        **Forecast before running:** Which score column will appear perfect? Will
        selecting only the two largest effects describe the full population?
        Write down the ordering you expect in both panels before you run the cell.

        This result calls the `label_leakage_report` and `cherry_pick_report`
        functions you implemented above. Finish their existing sub-function tests
        first. The plotting below is inline plumbing; no reference implementation
        or saved figure supplies the plotted values.
        '''
        interpretation = '''
        <details><summary>Help — trace each bar back to your function</summary>

        The label-encoding column is negative for label 0 and positive for label 1.
        The second column reverses every sign. It is an **anti-correlated** control,
        not an independent null. For selection, the full population has eight
        effects of 1 and two of 9. The representative subset preserves that 4:1
        proportion; the selected tail does not. Both subset means and the population
        mean are read from your own callback results.

        </details>

        <details><summary>Expected output — reveal after your forecast</summary>

        With correct functions, the first accuracy bar is 1 and the inverted bar is
        0. The selected tail has mean 9; the representative subset and population
        both have mean 2.6. These are hand-computable expectations for this literal
        fixture, not values assigned to the plot. If your figure disagrees, inspect
        the existing threshold-at-zero and selected-index tests.

        </details>
        '''
        play_prompt = '''
        ### Try It Yourself — change only the selected examples

        **Forecast:** Replacing the tail-only selection by `[0, 1, 8, 9]` adds two
        ordinary examples. Predict the new selected mean and whether the population
        bar can change. Then run the cell and compare it with your forecast. Try a
        second index selection while keeping `live06_effects` fixed. You are changing
        the claim's selection rule, not detector thresholds or the underlying data.
        '''
        limits = '''
        ### Limitations of this live signature

        This exact synthetic counterexample shows how leakage and selection can
        inflate a claim. The inverted feature is an anti-correlated stress test;
        it is not an independent null, a held-out data split, or a real-model
        mechanism experiment. The representative subset is deliberately matched
        to this known finite population; it is not evidence that arbitrary chosen
        examples are representative. These new plots make no report acceptance or
        historical CUDA claim. The original methods and historical evidence above
        retain their own stated scope.
        '''
        plot, play = _LEAKAGE_PLOT, _LEAKAGE_PLAY
    elif section == "1.6":
        forecast = '''
        ## Live signature — which missing term changes your memory decision?

        **Forecast before running:** As context length increases, which curves
        should stay flat, and which should grow? How will omitting KV dimensions
        change the total? Record your prediction before running your own
        `estimate_inference_memory` below.

        The illustrative assumptions are **1 billion parameter values, BF16
        (2 bytes/value), batch size 1, 16 layers, 8 KV heads, head dimension 128,
        hidden size 1024**, an activation multiplier of 2, and 0.25 GiB fixed
        overhead. Contexts are 128, 512, 2048 and 8192 tokens. They are tensor-count
        inputs for an accounting exercise, not a named model configuration.
        No model, weights or large tensor is created. The 3 GiB line is an
        illustrative planning budget, not a measurement or a device guarantee.
        '''
        interpretation = '''
        <details><summary>Help — follow the terms already implemented above</summary>

        All five curves use results returned by your estimator. The dashed control
        calls the same function with the KV dimensions intentionally omitted; it
        changes no memory formula. In the original implementation the fields end
        in `_gb` but divide bytes by `1024**3`, so these axes correctly say **GiB**.
        The total includes the fixed overhead, and is not just the three component
        curves shown. This is analytic accounting with no GPU allocation or memory
        measurement.

        </details>

        <details><summary>Expected output — reveal after your forecast</summary>

        Parameters should remain flat. KV and activation estimates should increase
        with context. At 8192 tokens the KV term is 0.5 GiB and the activation term
        is 0.03125 GiB; the dashed underestimate should miss precisely the KV term.
        The plotted values themselves come from your estimator, so an incorrect
        method must change the figure rather than being replaced by these values.

        </details>
        '''
        play_prompt = '''
        ### Try It Yourself — change only batch size

        **Forecast:** Double batch size to 2 while leaving parameter count, dtype,
        contexts and all other assumptions fixed. Which components double? At the
        longest context, will the total cross the illustrative 3 GiB line? Predict
        before running the next cell, then compare the two figures. Change only
        `live16_batch_size` for a second experiment.
        '''
        limits = '''
        ### Limitations of this live signature

        This uses your simple estimator, not a profiler: no GPU allocation, model
        loading, device availability query or actual memory measurement occurs.
        Allocator fragmentation, implementation-dependent workspaces, attention
        kernels, quantization metadata and model-specific cache layouts can change
        real usage. The omitted-KV curve is an intentionally incomplete accounting
        control, not a second valid model estimate. These illustrative curves do
        not replace the separate historical environment/CUDA evidence or confer
        acceptance on a real deployment.
        '''
        plot = _MEMORY_PLOT
        play = _MEMORY_PLOT.replace("live16_batch_size = 1  # Change only this to 2 in the play cell.",
                                    "live16_batch_size = 2  # Play condition: only batch size changed.")
    else:
        raise ValueError(f"No foundation live signature for section {section!r}")
    return [_cell(section, "forecast", "markdown", forecast),
            _cell(section, "plot", "code", plot),
            _cell(section, "interpret", "markdown", interpretation),
            _cell(section, "play-forecast", "markdown", play_prompt),
            _cell(section, "play", "code", play),
            _cell(section, "limits", "markdown", limits)]
