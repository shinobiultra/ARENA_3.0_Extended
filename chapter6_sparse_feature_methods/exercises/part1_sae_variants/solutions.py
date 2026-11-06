# %%


MAIN = __name__ == "__main__"
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch as t
import torch.nn.functional as F
from IPython.display import display

chapter = "chapter6_sparse_feature_methods"
section = "part1_sae_variants"
if MAIN:
    root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if MAIN:
    exercises_dir = root_dir / chapter / "exercises"
if MAIN:
    section_dir = exercises_dir / section
if MAIN:
    asset_dir = root_dir / chapter / "instructions" / "assets"
if MAIN:
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))
if MAIN:
    if str(exercises_dir) not in sys.path:
        sys.path.append(str(exercises_dir))

from chapter6_sparse_feature_methods.exercises.part1_sae_variants import sae_plumbing as reference
from chapter6_sparse_feature_methods.exercises.part1_sae_variants import tests
from chapter6_sparse_feature_methods.exercises.part1_sae_variants.sae_plumbing import (
    CausalInterventionReport,
    DictionaryRecoveryReport,
    EncoderFunctions,
    FeatureAUCReport,
    PlantedSparseBatch,
    SAEComparison,
    SAEForward,
    SAELoss,
    SAETrainConfig,
    SAEVariantMetrics,
    VariantResult,
)

if MAIN:
    t.set_num_threads(1)
if MAIN:
    assert not next(reference.SparseAutoencoder(t.zeros(16, 3), SAETrainConfig("relu_l1", d_sae=3)).parameters()).is_cuda

# %%

def make_planted_dictionary(*, n_features: int, d_model: int, seed: int = 0) -> t.Tensor:
    if n_features <= 0 or d_model <= 0:
        raise ValueError("n_features and d_model must be positive")
    generator = t.Generator(device="cpu").manual_seed(seed)
    return F.normalize(t.randn(n_features, d_model, generator=generator), dim=-1)


def sample_planted_batch(
    dictionary: t.Tensor,
    *,
    n_examples: int,
    feature_probability: float = 0.15,
    noise_std: float = 0.01,
    target_feature: int = 0,
    seed: int = 1,
) -> PlantedSparseBatch:
    if dictionary.ndim != 2:
        raise ValueError("dictionary must have shape (features, d_model)")
    if n_examples <= 0 or not 0.0 < feature_probability < 1.0 or noise_std < 0:
        raise ValueError("invalid sampling arguments")
    generator = t.Generator(device="cpu").manual_seed(seed)
    is_active = t.rand(n_examples, dictionary.shape[0], generator=generator) < feature_probability
    magnitudes = 0.5 + t.rand(n_examples, dictionary.shape[0], generator=generator)
    feature_acts = is_active.float() * magnitudes
    activations = feature_acts @ dictionary
    if noise_std:
        activations += noise_std * t.randn(activations.shape, generator=generator)
    return PlantedSparseBatch(
        feature_acts=feature_acts,
        activations=activations,
        labels=is_active[:, target_feature],
        dictionary=dictionary,
    )


if MAIN:
    tests.test_planted_sparse_ground_truth(make_planted_dictionary, sample_planted_batch)

# %%

def relu_l1_encode(pre_acts: t.Tensor) -> t.Tensor:
    return F.relu(pre_acts)


def topk_encode(pre_acts: t.Tensor, k: int) -> t.Tensor:
    if not 0 <= k <= pre_acts.shape[-1]:
        raise ValueError("k is outside the feature dimension")
    positive = F.relu(pre_acts)
    if k == 0:
        return t.zeros_like(positive)
    values, indices = positive.topk(k, dim=-1)
    return t.zeros_like(positive).scatter(-1, indices, values)


def gated_encode(magnitude_pre_acts: t.Tensor, gate_pre_acts: t.Tensor) -> t.Tensor:
    return F.relu(magnitude_pre_acts) * (gate_pre_acts > 0).to(magnitude_pre_acts.dtype)


def rectangle_kernel(x: t.Tensor) -> t.Tensor:
    return (x.abs() < 0.5).to(x.dtype)


class HeavisideSTE(t.autograd.Function):
    @staticmethod
    def forward(ctx, values: t.Tensor, threshold: t.Tensor, bandwidth: float) -> t.Tensor:
        ctx.save_for_backward(values, threshold)
        ctx.bandwidth = bandwidth
        return (values > threshold).to(values.dtype)

    @staticmethod
    def backward(ctx, grad_output: t.Tensor):
        values, threshold = ctx.saved_tensors
        local = rectangle_kernel((values - threshold) / ctx.bandwidth)
        grad_threshold = (-(local / ctx.bandwidth) * grad_output).sum(dim=0)
        return t.zeros_like(values), grad_threshold, None


class JumpReLU(t.autograd.Function):
    @staticmethod
    def forward(ctx, values: t.Tensor, threshold: t.Tensor, bandwidth: float) -> t.Tensor:
        ctx.save_for_backward(values, threshold)
        ctx.bandwidth = bandwidth
        return values * (values > threshold).to(values.dtype)

    @staticmethod
    def backward(ctx, grad_output: t.Tensor):
        values, threshold = ctx.saved_tensors
        local = rectangle_kernel((values - threshold) / ctx.bandwidth)
        grad_values = (values > threshold).to(values.dtype) * grad_output
        grad_threshold = (-(threshold / ctx.bandwidth) * local * grad_output).sum(dim=0)
        return grad_values, grad_threshold, None


def jumprelu_encode(pre_acts: t.Tensor, threshold: t.Tensor | float, bandwidth: float = 0.1) -> t.Tensor:
    threshold = t.as_tensor(threshold, dtype=pre_acts.dtype, device=pre_acts.device)
    return JumpReLU.apply(F.relu(pre_acts), threshold, bandwidth)


if MAIN:
    tests.test_encoder_rules_and_jumprelu_gradient(
        relu_l1_encode,
        topk_encode,
        gated_encode,
        jumprelu_encode,
        JumpReLU,
    )

# %%

def decode_features(
    feature_acts: t.Tensor,
    decoder_weight: t.Tensor,
    decoder_bias: t.Tensor | None = None,
) -> t.Tensor:
    reconstruction = feature_acts @ decoder_weight
    return reconstruction if decoder_bias is None else reconstruction + decoder_bias


def sparse_autoencoder_loss(
    output: SAEForward,
    activations: t.Tensor,
    *,
    variant: str,
    sparsity_coefficient: float,
    gated_aux_coefficient: float = 1.0,
    jump_bandwidth: float = 0.1,
) -> SAELoss:
    reconstruction = F.mse_loss(output.reconstruction, activations)
    zero = reconstruction.new_zeros(())
    auxiliary = zero
    if variant == "relu_l1":
        sparsity = output.feature_acts.abs().sum(dim=-1).mean()
    elif variant == "topk":
        sparsity = (output.feature_acts > 0).float().sum(dim=-1).mean()
    elif variant == "gated":
        sparsity = F.relu(output.gate_pre_acts).sum(dim=-1).mean()
        auxiliary = gated_aux_coefficient * F.mse_loss(output.auxiliary_reconstruction, activations)
    elif variant == "jumprelu":
        sparsity = HeavisideSTE.apply(
            F.relu(output.pre_acts), output.jump_threshold, jump_bandwidth
        ).sum(dim=-1).mean()
    else:
        raise ValueError(f"unknown variant: {variant}")
    sparsity_term = zero if variant == "topk" else sparsity_coefficient * sparsity
    return SAELoss(reconstruction + sparsity_term + auxiliary, reconstruction, sparsity, auxiliary)


if MAIN:
    tests.test_variant_loss_terms(sparse_autoencoder_loss)

# %%

def feature_density(feature_acts: t.Tensor) -> t.Tensor:
    return (feature_acts > 0).float().mean(dim=0)


def l0(feature_acts: t.Tensor) -> float:
    return float((feature_acts > 0).float().sum(dim=-1).mean())


def sae_variant_metrics(name, activations, reconstruction, feature_acts) -> SAEVariantMetrics:
    density = feature_density(feature_acts)
    return SAEVariantMetrics(
        name=name,
        l0=l0(feature_acts),
        feature_density_mean=float(density.mean()),
        dead_feature_fraction=float((density == 0).float().mean()),
        reconstruction_mse=float(F.mse_loss(reconstruction, activations)),
    )


def dictionary_recovery_report(
    learned_decoder: t.Tensor,
    true_dictionary: t.Tensor,
    threshold: float = 0.8,
) -> DictionaryRecoveryReport:
    cosine = F.normalize(learned_decoder, dim=-1) @ F.normalize(true_dictionary, dim=-1).T
    best_cosine, best_learned = cosine.max(dim=0)
    best_true_for_learned = cosine.argmax(dim=-1)
    duplicate_fraction = 1 - best_true_for_learned.unique().numel() / learned_decoder.shape[0]
    return DictionaryRecoveryReport(
        float(best_cosine.mean()),
        float((best_cosine >= threshold).float().mean()),
        duplicate_fraction,
        best_learned,
        cosine,
    )


def roc_auc_binary(scores: t.Tensor, labels: t.Tensor) -> float:
    scores, labels = scores.flatten().float(), labels.flatten().bool()
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs both classes")
    order = scores.argsort()
    sorted_scores = scores[order]
    sorted_ranks = t.empty_like(sorted_scores)
    _, counts = t.unique_consecutive(sorted_scores, return_counts=True)
    start = 0
    for count in counts.tolist():
        end = start + count
        sorted_ranks[start:end] = (start + 1 + end) / 2
        start = end
    ranks = t.empty_like(scores)
    ranks[order] = sorted_ranks
    auc = (ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def best_feature_auc(feature_acts: t.Tensor, labels: t.Tensor) -> FeatureAUCReport:
    best = FeatureAUCReport(0, -1.0, 1)
    for feature_id in range(feature_acts.shape[1]):
        raw_auc = roc_auc_binary(feature_acts[:, feature_id], labels)
        polarity = 1 if raw_auc >= 0.5 else -1
        oriented_auc = raw_auc if polarity == 1 else 1 - raw_auc
        if oriented_auc > best.auc:
            best = FeatureAUCReport(feature_id, oriented_auc, polarity)
    return best


def evaluate_selected_auc(
    feature_acts: t.Tensor,
    labels: t.Tensor,
    selection: FeatureAUCReport,
) -> float:
    raw_auc = roc_auc_binary(feature_acts[:, selection.feature_id], labels)
    return raw_auc if selection.polarity == 1 else 1 - raw_auc


if MAIN:
    tests.test_metrics_recovery_and_heldout_auc(
        decode_features,
        dictionary_recovery_report,
        best_feature_auc,
        evaluate_selected_auc,
    )

# %%

student_encoders = EncoderFunctions(
    relu_l1=relu_l1_encode,
    topk=topk_encode,
    gated=gated_encode,
    jumprelu=jumprelu_encode,
)

# %%

@t.no_grad()
def causal_intervention_report(
    model,
    heldout: PlantedSparseBatch,
    *,
    target_feature: int = 0,
    steering_coefficient: float = 0.75,
    random_seed: int = 17,
) -> CausalInterventionReport:
    target_direction = F.normalize(heldout.dictionary[target_feature], dim=0)
    decoder = F.normalize(model.w_dec.detach(), dim=-1)
    target_feature_id = int((decoder @ target_direction).argmax())
    matched_direction = decoder[target_feature_id]
    generator = t.Generator(device="cpu").manual_seed(random_seed)
    random_direction = t.randn(target_direction.shape, generator=generator)
    random_direction -= (random_direction @ target_direction) * target_direction
    random_direction = F.normalize(random_direction, dim=0)

    negatives = heldout.activations[~heldout.labels]
    baseline_scores = negatives @ target_direction
    steered_scores = (negatives + steering_coefficient * matched_direction) @ target_direction
    random_scores = (negatives + steering_coefficient * random_direction) @ target_direction

    positive_output = model(heldout.activations[heldout.labels], student_encoders)
    coefficients = positive_output.feature_acts[:, target_feature_id]
    baseline_reconstruction = positive_output.reconstruction
    target_ablated = baseline_reconstruction - coefficients[:, None] * matched_direction
    random_ablated = baseline_reconstruction - coefficients[:, None] * random_direction
    baseline_projection = baseline_reconstruction @ target_direction
    return CausalInterventionReport(
        target_feature_id,
        float(matched_direction @ target_direction),
        float((steered_scores - baseline_scores).mean()),
        float((random_scores - baseline_scores).mean()),
        float((baseline_projection - target_ablated @ target_direction).mean()),
        float((baseline_projection - random_ablated @ target_direction).mean()),
    )


if MAIN:
    tests.test_causal_interventions_beat_equal_strength_random_controls(causal_intervention_report)

# %%

@t.no_grad()
def evaluate_student_variant(name, model, train, heldout, train_losses) -> VariantResult:
    train_output = model(train.activations, student_encoders)
    heldout_output = model(heldout.activations, student_encoders)
    metrics = sae_variant_metrics(
        name, heldout.activations, heldout_output.reconstruction, heldout_output.feature_acts
    )
    recovery = dictionary_recovery_report(model.w_dec, heldout.dictionary)
    selection = best_feature_auc(train_output.feature_acts, train.labels)
    heldout_auc = evaluate_selected_auc(heldout_output.feature_acts, heldout.labels, selection)
    generator = t.Generator(device="cpu").manual_seed(1000 + reference.VARIANTS.index(name))
    shuffled_train = train.labels[t.randperm(train.labels.numel(), generator=generator)]
    shuffled_selection = best_feature_auc(train_output.feature_acts, shuffled_train)
    shuffled_heldout = heldout.labels[t.randperm(heldout.labels.numel(), generator=generator)]
    shuffled_auc = evaluate_selected_auc(heldout_output.feature_acts, shuffled_heldout, shuffled_selection)
    intervention = causal_intervention_report(model, heldout)
    return VariantResult(
        name, model, metrics, recovery, heldout_auc, shuffled_auc, intervention, train_losses
    )


def run_student_comparison(
    *,
    steps: int = 900,
    train_examples: int = 4096,
    heldout_examples: int = 2048,
    n_features: int = 8,
    d_model: int = 6,
    feature_probability: float = 0.15,
    noise_std: float = 0.01,
    sparsity_coefficient: float = 0.02,
    k: int = 2,
    seed: int = 1,
) -> SAEComparison:
    dictionary = make_planted_dictionary(n_features=n_features, d_model=d_model, seed=seed)
    train = sample_planted_batch(
        dictionary,
        n_examples=train_examples,
        feature_probability=feature_probability,
        noise_std=noise_std,
        seed=seed + 1,
    )
    heldout = sample_planted_batch(
        dictionary,
        n_examples=heldout_examples,
        feature_probability=feature_probability,
        noise_std=noise_std,
        seed=seed + 2,
    )
    results = {}
    for variant in reference.VARIANTS:
        config = SAETrainConfig(
            variant=variant,
            d_sae=n_features,
            k=k,
            sparsity_coefficient=sparsity_coefficient,
            steps=steps,
            seed=seed + 2,
        )
        model, losses = reference.train_sae_variant(
            train.activations,
            config,
            encoders=student_encoders,
            loss_fn=sparse_autoencoder_loss,
        )
        results[variant] = evaluate_student_variant(variant, model, train, heldout, losses)
    zero_baseline = train.activations.mean(0).expand_as(heldout.activations)
    random_generator = t.Generator(device="cpu").manual_seed(seed + 500)
    random_decoder = F.normalize(
        t.randn(n_features, d_model, generator=random_generator), dim=-1
    )
    return SAEComparison(
        train,
        heldout,
        results,
        float(F.mse_loss(zero_baseline, heldout.activations)),
        l0(heldout.feature_acts),
        dictionary_recovery_report(random_decoder, dictionary).recovered_fraction,
    )


if MAIN:
    tests.test_end_to_end_variant_comparison(run_student_comparison)

# %%

if MAIN:
    comparison = run_student_comparison()
if MAIN:
    metric_rows = reference.comparison_rows(comparison)
if MAIN:
    metric_table = pd.DataFrame(metric_rows).round(4)
if MAIN:
    display(metric_table)

if MAIN:
    signature_path = asset_dir / "sae_variants_signature_result.png"
if MAIN:
    figure = reference.plot_signature_result(comparison, save_path=signature_path)
if MAIN:
    plt.show()
if MAIN:
    print(f"Saved reviewer snapshot to {signature_path}")

# %%

# Try It Yourself: uncomment one setting, rerun, and compare all six panels.
# play = run_student_comparison(sparsity_coefficient=0.005, seed=1)
# play = run_student_comparison(sparsity_coefficient=0.05, seed=1)
# play = run_student_comparison(k=1, seed=1)
# play = run_student_comparison(k=4, seed=1)
# display(pd.DataFrame(reference.comparison_rows(play)).round(4))
# reference.plot_signature_result(play)
# plt.show()

# %%

if MAIN:
    real_model_config = {
        "model": reference.PYTHIA_SAE_MODEL_ID,
        "revision": reference.PYTHIA_SAE_REVISION,
        "hidden_layer": reference.PYTHIA_SAE_HIDDEN_LAYER,
        "train_prompts": 64,
        "heldout_prompts": 16,
        "sae_width": reference.PYTHIA_SAE_WIDTH,
        "topk": reference.PYTHIA_SAE_TOPK,
        "training_steps": reference.PYTHIA_SAE_TRAINING_STEPS,
        "executed_in_this_cpu_notebook": False,
    }
if MAIN:
    display(pd.DataFrame([real_model_config]).T.rename(columns={0: "value"}))

# %%

def run_gpu_test(max_vram_gb: float = 24.0):
    """Parent-owned optional CUDA path; never called by the CPU lesson."""
    return reference.run_gpu_test(max_vram_gb=max_vram_gb)


def run_full_experiment(max_vram_gb: float = 24.0):
    return run_gpu_test(max_vram_gb=max_vram_gb)

# %%

def run_smoke_test(cpu: bool = True) -> dict:
    assert cpu, "the learner smoke test is CPU-only"
    smoke = run_student_comparison(steps=250, train_examples=1024, heldout_examples=512)
    return {
        "variants": reference.comparison_rows(smoke),
        "zero_baseline_mse": smoke.zero_baseline_mse,
        "true_l0": smoke.true_l0,
    }


if MAIN:
    tests.test_notebook_contract(run_smoke_test)

# %%
