# Expected Outputs for [12.3] Mini VLM from Scratch

## How this fixture was produced

Run `scripts/generate_extension_verification_assets.py` for structural assets,
then run `scripts/run_extension_verification_reports.py --section 12.3` on the
release CUDA environment. The report executes both the compact notebook
contract and the complete 260-step CUDA experiment.

## Trusted implementation

`solutions.py` is the release implementation. Its exact additive visual-token
oracle establishes the patching ground truth before the learned MiniVLM result.
The solved notebook independently exposes the same patch encoder, connector,
causal decoder, losses, controls, and intervention functions; none of these
taught methods is delegated back to the release module.

## Random seed

The renderer, train/held-out split, parameter initialization, training order,
shuffled-visual control, and random-region control use deterministic seed `123`
or an explicitly derived local seed.

## Allowed tolerances

Tensor parity checks use `rtol=1e-5` and `atol=1e-6`. Release metrics use
thresholds from `artifacts.lock.yml`; exact booleans are required for patch
flips, preserved controls, and full-sequence counterfactual parity.

## When to regenerate

Regenerate after changing the renderer, split, model, training loop, patch
locations, controls, expected thresholds, notebooks, instruction page, or
verification infrastructure. Do not regenerate merely to hide a failing check.
