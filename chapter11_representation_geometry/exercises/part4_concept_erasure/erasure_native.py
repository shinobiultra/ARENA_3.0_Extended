"""Native model plumbing reused from 11.3; erasure itself stays learner-visible."""
import torch as t
from chapter11_representation_geometry.exercises.part3_causal_abstraction_das.das_native import (
    NativeCache, capture_native, continue_native, full_forward_intervention,
)
from chapter11_representation_geometry.exercises.part3_causal_abstraction_das.run_native_development import weight_digest


@t.no_grad()
def native_with_eraser(model, cache, layer, position, eraser, apply_eraser, batch_size=24):
    activation = cache.activations(layer, position).double()
    outputs, replacements = [], []
    for start in range(0, len(activation), batch_size):
        indices = t.arange(start, min(start + batch_size, len(activation)))
        replacement = apply_eraser(activation[indices], eraser).to(cache.logits.dtype)
        outputs.append(continue_native(model, cache, layer, indices, replacement, position))
        replacements.append(replacement)
    return t.cat(outputs), t.cat(replacements)


def native_metrics(logits, original_logits, rows, answer_ids):
    targets = t.tensor([answer_ids[row['answer']] for row in rows])
    original_log_prob = original_logits.log_softmax(-1)
    log_prob = logits.log_softmax(-1)
    correct = logits.argmax(-1) == targets
    originally_correct = original_logits.argmax(-1) == targets
    return dict(rows=len(rows), accuracy=float(correct.double().mean()),
                correct=int(correct.sum()), native_correct_count=int(originally_correct.sum()),
                preserved_native_correct=int((correct & originally_correct).sum()),
                answer_loss=float(t.nn.functional.cross_entropy(logits, targets)),
                answer_probability=float(log_prob[t.arange(len(rows)), targets].exp().mean()),
                kl_from_native=float((original_log_prob.exp() * (original_log_prob - log_prob)).sum(-1).mean()),
                changed_winner_fraction=float((logits.argmax(-1) != original_logits.argmax(-1)).double().mean()))
