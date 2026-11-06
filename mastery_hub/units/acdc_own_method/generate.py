"""Deterministic, stdlib-only source generator. Import never binds/executes taught code."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

GRAPH = 'from types import SimpleNamespace\n\ndef own_graph(n_layers, n_heads):\n    """Pure sender/receiver metadata; no tensor, model or legacy module import."""\n    if type(n_layers) is not int or type(n_heads) is not int or min(n_layers, n_heads) < 1:\n        raise ValueError("positive integer dimensions required")\n    hs = lambda layer, head: 1 + layer * (n_heads + 1) + head\n    ms = lambda layer: 1 + layer * (n_heads + 1) + n_heads\n    hr = lambda layer, qkv, head: layer * (3 * n_heads + 1) + "qkv".index(qkv) * n_heads + head\n    mr = lambda layer: layer * (3 * n_heads + 1) + 3 * n_heads\n    senders = ["input"]\n    receivers = []\n    write = [0]\n    read = []\n    for layer in range(n_layers):\n        senders += [f"a{layer}.{head}" for head in range(n_heads)] + [f"m{layer}"]\n        receivers += [f"a{layer}.{head}<{qkv}>" for qkv in "qkv" for head in range(n_heads)] + [f"m{layer}"]\n        write += [2 * layer + 1] * n_heads + [2 * layer + 2]\n        read += [2 * layer + 1] * (3 * n_heads) + [2 * layer + 2]\n    receivers += ["logits"]\n    read += [2 * n_layers + 1]\n    return SimpleNamespace(n_layers=n_layers, n_heads=n_heads, head_sender=hs, mlp_sender=ms,\n        head_receiver=hr, mlp_receiver=mr, LOGITS_RECEIVER=len(receivers)-1,\n        SENDER_NAMES=tuple(senders), RECEIVER_NAMES=tuple(receivers),\n        valid=tuple(tuple(w < r for r in read) for w in write))\n\ndef own_gpt2_receiver_order(graph):\n    order = [graph.LOGITS_RECEIVER]\n    for layer in reversed(range(graph.n_layers)):\n        order += [graph.mlp_receiver(layer)]\n        order += [graph.head_receiver(layer, qkv, head)\n                  for head in reversed(range(graph.n_heads)) for qkv in "qkv"]\n    return order\n\ndef own_gpt2_receiver_owner(graph):\n    owners = [None] * len(graph.RECEIVER_NAMES)\n    for layer in range(graph.n_layers):\n        owners[graph.mlp_receiver(layer)] = graph.mlp_sender(layer)\n        for head in range(graph.n_heads):\n            for qkv in "qkv":\n                owners[graph.head_receiver(layer, qkv, head)] = graph.head_sender(layer, head)\n    return owners\n'

METHOD = 'def make_own_methods(t):\n    """Bind an explicit tensor backend once; no model call occurs during binding.\n\n    In the notebook, pass the explicitly imported torch module. DEV tests use only\n    fixed CPU literals. No decorator, implicit grad mode, flags, or global backend.\n    Caller owns inference_mode and the prepared, non-concurrent hook context.\n    """\n    # Capture operation objects now, never a late mutable global/backend lookup.\n    einsum, stack, zeros_like, boolean = t.einsum, t.stack, t.zeros_like, t.bool\n    def own_sender_outputs(model, tokens, graph):\n        names = ["blocks.0.hook_resid_pre"]\n        for layer in range(graph.n_layers):\n            names += [f"blocks.{layer}.attn.hook_z", f"blocks.{layer}.hook_mlp_out"]\n        _, cache = model.run_with_cache(tokens, names_filter=names)\n        outputs = [cache[names[0]]]\n        for layer in range(graph.n_layers):\n            # Each head is projected into residual coordinates BEFORE stacking.\n            projected = einsum("bphe,hed->bphd", cache[f"blocks.{layer}.attn.hook_z"], model.W_O[layer])\n            outputs += [projected[:, :, head] for head in range(graph.n_heads)]\n            outputs += [cache[f"blocks.{layer}.hook_mlp_out"]]\n        return stack(outputs, dim=2)\n\n    def own_circuit_logits(model, clean_tokens, corrupt_out, in_circuit, edge_mask, graph):\n        shape = (len(graph.SENDER_NAMES), len(graph.RECEIVER_NAMES))\n        if in_circuit.dtype != boolean or edge_mask.dtype != boolean:\n            raise TypeError("boolean masks required")\n        if tuple(in_circuit.shape) != shape or tuple(edge_mask.shape) != shape:\n            raise ValueError("sender/receiver mask shape mismatch")\n        if edge_mask.device != in_circuit.device:\n            raise ValueError("mask devices differ")\n        if model.cfg.n_layers != graph.n_layers or model.cfg.n_heads != graph.n_heads:\n            raise ValueError("graph/model dimensions differ")\n        if corrupt_out.ndim != 4 or tuple(corrupt_out.shape[:2]) != tuple(clean_tokens.shape[:2]) or corrupt_out.shape[2] != shape[0]:\n            raise ValueError("corrupt sender cache shape mismatch")\n        if corrupt_out.shape[-1] != model.W_O.shape[-1] or corrupt_out.dtype != model.W_O.dtype:\n            raise ValueError("cache residual width/dtype differs from prepared weights")\n        if corrupt_out.device != model.W_O.device or str(corrupt_out.device) != str(model.cfg.device):\n            raise ValueError("prepared activation/cache/weight devices differ; no automatic moves")\n        # Invalid outside-mask bits remain ignored, matching the existing contract.\n        # Only boolean mask coefficients are converted; activations/weights stay exact.\n        removed = (edge_mask & ~in_circuit).to(device=model.cfg.device, dtype=corrupt_out.dtype)\n        diff = zeros_like(corrupt_out)  # fresh on EVERY invocation\n        def store_input(x, hook):\n            diff[:, :, 0] = x - corrupt_out[:, :, 0]\n        def store_heads(z, hook):\n            start = graph.head_sender(hook.layer(), 0)\n            out = einsum("bphe,hed->bphd", z, model.W_O[hook.layer()])\n            diff[:, :, start:start + graph.n_heads] = out - corrupt_out[:, :, start:start + graph.n_heads]\n        def store_mlp(x, hook):\n            u = graph.mlp_sender(hook.layer())\n            diff[:, :, u] = x - corrupt_out[:, :, u]\n        def patch_head_inputs(x, hook):\n            qkv = hook.name.split(".")[-1][len("hook_")]\n            start = graph.head_receiver(hook.layer(), qkv, 0)\n            return x - einsum("bpsd,sh->bphd", diff, removed[:, start:start + graph.n_heads])\n        def patch_single_input(x, hook):\n            v = graph.LOGITS_RECEIVER if hook.name.endswith("resid_post") else graph.mlp_receiver(hook.layer())\n            return x - einsum("bpsd,s->bpd", diff, removed[:, v])\n        hooks = [("blocks.0.hook_resid_pre", store_input)]\n        for layer in range(graph.n_layers):\n            hooks += [(f"blocks.{layer}.hook_{qkv}_input", patch_head_inputs) for qkv in "qkv"]\n            hooks += [(f"blocks.{layer}.attn.hook_z", store_heads),\n                      (f"blocks.{layer}.hook_mlp_in", patch_single_input),\n                      (f"blocks.{layer}.hook_mlp_out", store_mlp)]\n        hooks += [(f"blocks.{graph.n_layers-1}.hook_resid_post", patch_single_input)]\n        return model.run_with_hooks(clean_tokens, fwd_hooks=hooks)[:, -1]\n    return own_sender_outputs, own_circuit_logits\n'

CONTROLS = 'def make_controls(t):\n    arange, isfinite = t.arange, t.isfinite\n    def raw_one_edge_oracle(model, clean_tokens, corrupt_tokens, sender, receiver, graph):\n        """Independent direct delta from RAW hooks, never own_sender_outputs/cache."""\n        def direct_sender(tokens):\n            if sender == 0:\n                name, head, layer = "blocks.0.hook_resid_pre", None, None\n            else:\n                layer, local = divmod(sender - 1, graph.n_heads + 1)\n                head = local if local < graph.n_heads else None\n                name = f"blocks.{layer}.attn.hook_z" if head is not None else f"blocks.{layer}.hook_mlp_out"\n            _, raw = model.run_with_cache(tokens, names_filter=[name])\n            if head is None:\n                return raw[name].clone()\n            # Matmul on one raw head: independent of the main einsum/stack path.\n            return raw[name][:, :, head, :] @ model.W_O[layer, head]\n        delta = direct_sender(corrupt_tokens) - direct_sender(clean_tokens)\n        if receiver == graph.LOGITS_RECEIVER:\n            name, head = f"blocks.{graph.n_layers-1}.hook_resid_post", None\n        else:\n            layer, local = divmod(receiver, 3 * graph.n_heads + 1)\n            if local == 3 * graph.n_heads:\n                name, head = f"blocks.{layer}.hook_mlp_in", None\n            else:\n                qkv_index, head = divmod(local, graph.n_heads)\n                name = f"blocks.{layer}.hook_{\'qkv\'[qkv_index]}_input"\n        def patch(x, hook):\n            if head is None:\n                return x + delta\n            out = x.clone()\n            out[:, :, head] += delta\n            return out\n        return model.run_with_hooks(clean_tokens, fwd_hooks=[(name, patch)])[:, -1]\n\n    def greater_than_rows(logits, year_tokens, start_years):\n        # Full vocabulary normalization first; NEVER softmax only year columns.\n        probs = logits.softmax(dim=-1)[:, year_tokens]\n        greater = arange(len(year_tokens), device=logits.device)[None, :] > start_years[:, None]\n        return (probs * greater).sum(-1) - (probs * ~greater).sum(-1)\n\n    def summarize_rows(records):\n        names = ["full", "empty", "selected"] + [f"random_{i}" for i in range(20)]\n        if set(records) != set(names):\n            raise ValueError("full/empty/selected and exactly 20 random records required")\n        ids = records["full"]["row_ids"]\n        if not ids or len(set(ids)) != len(ids):\n            raise ValueError("unique fixed row identities required")\n        means = {}\n        for name in names:\n            row = records[name]\n            values = row["values"]\n            if row["row_ids"] != ids or values.ndim != 1 or len(values) != len(ids) or not bool(isfinite(values).all()):\n                raise ValueError("row identity/shape/finite mismatch; no rows dropped")\n            means[name] = values.mean().item()\n        if any(not __import__("math").isfinite(value) for value in means.values()):\n            raise ValueError("nonfinite derived mean; retain raw records")\n        denominator = means["full"] - means["empty"]\n        # Raw records remain available to caller even when normalization stops.\n        if not denominator > 0 or not __import__("math").isfinite(denominator):\n            raise ValueError("nonpositive/nonfinite full-minus-empty; retain raw records")\n        recovery = {}\n        for key, mean in means.items():\n            numerator = mean - means["empty"]\n            if not __import__("math").isfinite(numerator):\n                raise ValueError("nonfinite derived numerator; retain raw records")\n            value = numerator / denominator\n            if not __import__("math").isfinite(value):\n                raise ValueError("nonfinite normalized recovery; retain raw records")\n            recovery[key] = value\n        return {"raw_means": means, "denominator": denominator, "recovery": recovery}\n    return raw_one_edge_oracle, greater_than_rows, summarize_rows\n'

TRANSFER = 'import math\nfrom types import SimpleNamespace\n\nclass TransferFailure(RuntimeError):\n    """A stopped attempt retains completed rows and the failing returned value."""\n    def __init__(self, primary, attempt):\n        super().__init__(f"own transfer stopped: {type(primary).__name__}: {primary}")\n        self.primary = primary\n        self.attempt = attempt\n\n\ndef require_prepared_runtime(prepared):\n    # This gate runs BEFORE inspecting any model/data. DEV tests do not adopt tolerances.\n    if prepared is None or getattr(prepared, "parity_tolerances", None) is None:\n        raise RuntimeError("GPT-2 parity atol/rtol UNSET: future explicit adoption required")\n    tolerances = prepared.parity_tolerances\n    if set(tolerances) != {"atol", "rtol"} or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in tolerances.values()):\n        raise ValueError("explicit finite nonnegative atol and rtol required")\n    if not getattr(prepared, "runtime_authorized", False):\n        raise RuntimeError("prepared-model runtime is not authorized by this addon")\n    if not getattr(prepared, "known_clean_hook_context", False):\n        raise RuntimeError("require known nonconcurrent clean hook context; no global hook reset")\n    cfg = prepared.model.cfg\n    if (cfg.n_layers, cfg.n_heads) != (12, 12):\n        raise ValueError("existing GPT-2 Small 12x12 protocol only")\n    if not cfg.use_split_qkv_input or not cfg.use_hook_mlp_in:\n        raise ValueError("caller must explicitly prepare hook flags; addon never changes flags")\n    for task in (prepared.discovery, prepared.heldout):\n        if len(task.row_ids) != 64 or len(set(task.row_ids)) != 64 or len(task.clean_tokens) != 64:\n            raise ValueError("exact 64-row cohort with unique frozen identities required")\n    if prepared.random_seed != 4 or prepared.discovery.seed != 0 or prepared.heldout.seed != 1:\n        raise ValueError("existing seed0/seed1 cohorts and seed4 controls required")\n    if set(prepared.discovery.row_ids) & set(prepared.heldout.row_ids):\n        raise ValueError("cohort row identities overlap")\n\n\ndef bind_own_transfer(*, circuit_logits, sender_outputs, prepared, kl_divergence,\n                      acdc, receiver_order, receiver_owner, random_circuit):\n    """Freeze already prepared runtime; no forward, cache construction or RNG in binding.\n\n    The explicit prepared.identity strings name model/revision and exact source hashes.\n    Each task.cache_binding is (model, corrupt_tokens, sender_outputs, sender_source_hash).\n    hook_state() must expose a stable immutable hook-context descriptor. This is a\n    caller-owned nonconcurrent session: mutation is rejected, never repaired/reset.\n    """\n    require_prepared_runtime(prepared)\n    t = prepared.backend  # Explicit supplied backend, not an import or global lookup.\n    selected_callback = circuit_logits\n    callback_code = getattr(circuit_logits, "__code__", None)\n    sender_code = getattr(sender_outputs, "__code__", None)\n    model, graph, edges = prepared.model, prepared.graph, prepared.edges\n    discovery, heldout = prepared.discovery, prepared.heldout\n    if prepared.sender_callback is not sender_outputs:\n        raise ValueError("corrupt cache must be attributed to this exact sender callback")\n    keys = {"model_revision", "graph_source_sha256", "sender_source_sha256", "circuit_source_sha256"}\n    identity = dict(prepared.identity)\n    if set(identity) != keys or any(not isinstance(v, str) or not v for v in identity.values()):\n        raise ValueError("explicit model/revision and graph/sender/circuit source identities required")\n    for key in keys - {"model_revision"}:\n        if len(identity[key]) != 64 or any(c not in "0123456789abcdef" for c in identity[key]):\n            raise ValueError("source identities must be exact SHA256 strings")\n    floating = model.W_O.dtype\n    device = model.W_O.device\n    mask_device = getattr(edges, "device", None)  # Masks retain their common caller-owned device.\n    cfg = model.cfg\n    vocab, width = cfg.d_vocab, model.W_O.shape[-1]\n    shape = (len(graph.SENDER_NAMES), len(graph.RECEIVER_NAMES))\n    equal, finite, boolean, integer = t.equal, t.isfinite, t.bool, t.int64\n    def tensor(value, expected_shape, dtype, label, expected_device=device):\n        if not isinstance(value, t.Tensor) or tuple(value.shape) != tuple(expected_shape) or value.dtype != dtype or value.device != expected_device:\n            raise ValueError(label + " exact tensor shape/device/dtype mismatch")\n        if not bool(finite(value).all()):\n            raise ValueError(label + " nonfinite")\n        return value\n    tensor(edges, shape, boolean, "valid edge mask", expected_device=mask_device)\n    if not equal(edges, t.tensor(graph.valid, dtype=boolean, device=mask_device)):\n        raise ValueError("edge mask differs from explicit graph")\n    cfg_fields = ("n_layers", "n_heads", "d_vocab", "device", "use_split_qkv_input", "use_hook_mlp_in")\n    cfg_frozen = tuple(getattr(cfg, key) for key in cfg_fields)\n    graph_fields = ("n_layers", "n_heads", "LOGITS_RECEIVER", "SENDER_NAMES", "RECEIVER_NAMES", "valid")\n    graph_frozen = tuple(getattr(graph, key) for key in graph_fields)\n    graph_functions = tuple(getattr(graph, key) for key in ("head_sender", "mlp_sender", "head_receiver", "mlp_receiver"))\n    hook_state = prepared.hook_state\n    hooks_frozen = hook_state()\n    if not isinstance(hooks_frozen, tuple):\n        raise ValueError("immutable hook state tuple required")\n    # Exact clones freeze numerical bytes and metadata; object identity rejects replacements.\n    # named_parameters/buffers include all model state, not only the projection used here.\n    def model_values():\n        return tuple(model.named_parameters()) + tuple(model.named_buffers())\n    values_frozen = tuple((name, value, value.detach().clone()) for name, value in model_values())\n    if not values_frozen:\n        raise ValueError("explicit named model numerical state required")\n    for name, value, frozen in values_frozen:\n        if not bool(finite(value).all()):\n            raise ValueError("model state nonfinite: " + name)\n    snapshots = []\n    for task in (discovery, heldout):\n        p = task.clean_tokens.shape[1]\n        tensor(task.clean_tokens, (64, p), integer, "clean tokens")\n        tensor(task.corrupt_tokens, (64, p), integer, "corrupt tokens")\n        tensor(task.corrupt_out, (64, p, shape[0], width), floating, "corrupt sender cache")\n        # Snapshot values independently: caller-owned list entries may later change.\n        # List and tuple inputs retain the same attribution contract.\n        binding = tuple(task.cache_binding)\n        if len(binding) != 4 or binding[0] is not model or binding[1] is not task.corrupt_tokens or binding[2] is not sender_outputs or binding[3] != identity["sender_source_sha256"]:\n            raise ValueError("cache attribution differs from exact model/input/sender source")\n        if task is discovery:\n            tensor(task.full_clean_logprobs, (64, 1, vocab), floating, "clean log-prob anchor")\n            fields = ("clean_tokens", "corrupt_tokens", "corrupt_out", "full_clean_logprobs")\n        else:\n            if not isinstance(task.answers, t.Tensor) or task.answers.ndim < 1 or len(task.answers) != 64 or task.answers.device != device or not bool(finite(task.answers).all()):\n                raise ValueError("exact finite 64-row answers required")\n            fields = ("clean_tokens", "corrupt_tokens", "corrupt_out", "answers")\n        snapshots.append((task, tuple(task.row_ids), task.seed, binding,\n            tuple((key, getattr(task, key), getattr(task, key).detach().clone()) for key in fields)))\n    metric = heldout.metric\n    generator = prepared.random_generator\n    tolerances_frozen = dict(prepared.parity_tolerances)\n    edge_frozen = edges.detach().clone()\n    def same(value, original, snapshot):\n        return value is original and value.dtype == snapshot.dtype and value.device == snapshot.device and tuple(value.shape) == tuple(snapshot.shape) and equal(value, snapshot)\n    def validate_bound():\n        require_prepared_runtime(prepared)\n        if model.cfg is not cfg or prepared.model is not model or prepared.graph is not graph or prepared.discovery is not discovery or prepared.heldout is not heldout or prepared.edges is not edges:\n            raise ValueError("bound runtime object replaced")\n        if prepared.backend is not t or prepared.sender_callback is not sender_outputs or prepared.random_generator is not generator or prepared.hook_state is not hook_state or hook_state() != hooks_frozen or heldout.metric is not metric:\n            raise ValueError("bound callback/generator/hook context drift")\n        if getattr(selected_callback, "__code__", None) is not callback_code or getattr(sender_outputs, "__code__", None) is not sender_code:\n            raise ValueError("bound callback source code changed")\n        if prepared.identity != identity or prepared.parity_tolerances != tolerances_frozen or tuple(getattr(cfg, key) for key in cfg_fields) != cfg_frozen:\n            raise ValueError("bound model/source/flag identity drift")\n        if tuple(getattr(graph, key) for key in graph_fields) != graph_frozen or tuple(getattr(graph, key) for key in ("head_sender", "mlp_sender", "head_receiver", "mlp_receiver")) != graph_functions:\n            raise ValueError("bound graph drift")\n        if not same(edges, edges, edge_frozen):\n            raise ValueError("bound valid-edge mask changed")\n        current = model_values()\n        if len(current) != len(values_frozen) or any(name != prior_name or not same(value, original, frozen) for (name, value), (prior_name, original, frozen) in zip(current, values_frozen)):\n            raise ValueError("bound model numerical state changed")\n        for task, ids, seed, binding, fields in snapshots:\n            now_binding = task.cache_binding\n            if tuple(task.row_ids) != ids or task.seed != seed or len(now_binding) != 4 or any(now_binding[i] is not binding[i] for i in range(3)) or now_binding[3] != binding[3]:\n                raise ValueError("bound cohort/cache attribution drift")\n            for key, original, frozen in fields:\n                if not same(getattr(task, key), original, frozen):\n                    raise ValueError("bound cohort/cache bytes or object changed: " + key)\n    def check_mask(mask):\n        tensor(mask, shape, boolean, "circuit mask", expected_device=mask_device)\n    def boundary_failure(primary, attempt):\n        try:\n            validate_bound()\n        except Exception as secondary:\n            message = f"post-failure identity validation: {type(secondary).__name__}: {secondary}"\n            if attempt is not None:\n                attempt["secondary_failures"].append(message)\n            else:\n                primary.add_note(message)\n    def checked_call(fn, args, label, attempt=None):\n        validate_bound()\n        try:\n            value = fn(*args)\n        except Exception as primary:\n            boundary_failure(primary, attempt)\n            raise\n        try:\n            if attempt is not None:\n                capture(value, label, attempt)\n        except Exception as primary:\n            boundary_failure(primary, attempt)\n            raise\n        validate_bound()\n        return value\n    def invoke(task, mask, attempt=None):\n        validate_bound(); check_mask(mask)\n        frozen_mask = mask.detach().clone()\n        logits = checked_call(selected_callback,\n            (model, task.clean_tokens, task.corrupt_out, mask, edges, graph), "callback:logits", attempt)\n        if not same(mask, mask, frozen_mask):\n            raise ValueError("callback changed supplied mask")\n        tensor(logits, (64, vocab), floating, "callback logits")\n        return logits\n    def discovery_kl(mask):\n        logits = invoke(discovery, mask)\n        result = checked_call(kl_divergence,\n            (logits[:, None], discovery.full_clean_logprobs), "KL scalar")\n        tensor(result, (), floating, "KL scalar")\n        value = result.item()\n        if not math.isfinite(value):\n            raise ValueError("KL scalar nonfinite")\n        return value\n    attempts = []\n    def new_attempt(kind):\n        attempt = {"kind": kind, "status": "running", "attempted": 0, "completed": 0,\n                   "condition": None, "rows": {}, "captures": [], "secondary_failures": [], "primary_failure": None}\n        attempts.append(attempt)  # In-memory fallback exists BEFORE calling anything.\n        return attempt\n    def capture(value, label, attempt):\n        entry = {"label": label, "returned": value, "snapshot": None}\n        attempt["captures"].append(entry)  # Preserve actual returned object even if clone fails.\n        try:\n            entry["snapshot"] = value.detach().clone()\n        except Exception as failure:\n            attempt["secondary_failures"].append(f"capture {label}: {type(failure).__name__}: {failure}")\n            raise\n        return entry["snapshot"]\n    def stop(primary, attempt):\n        attempt["status"] = "failed"\n        attempt["primary_failure"] = {"type": type(primary).__name__, "message": str(primary)}\n        # No external writer is needed to retain the attempt. A failed optional sink\n        # is secondary evidence and never replaces the primary failure or rows.\n        sink = getattr(prepared, "attempt_sink", None)\n        if sink is not None:\n            try:\n                sink(attempt)\n            except Exception as failure:\n                attempt["secondary_failures"].append(f"failure sink: {type(failure).__name__}: {failure}")\n        raise TransferFailure(primary, attempt) from primary\n    def heldout_one(mask, name, attempt):\n        attempt["condition"] = name; attempt["attempted"] += 1\n        check_mask(mask)\n        mask_snapshot = capture(mask, name + ":mask", attempt)\n        logits = invoke(heldout, mask, attempt)\n        values = checked_call(metric, (logits[:, None], heldout.answers), name + ":rows", attempt)\n        frozen_values = attempt["captures"][-1]["snapshot"]\n        if not same(mask, mask, mask_snapshot):\n            raise ValueError("metric changed mask")\n        tensor(frozen_values, (64,), floating, "per-row metric")\n        row = {"row_ids": tuple(snapshots[1][1]), "values": frozen_values, "mask": mask_snapshot}\n        attempt["rows"][name] = row\n        attempt["completed"] += 1\n        return row\n    def heldout_metric(mask):\n        attempt = new_attempt("heldout_single")\n        try:\n            row = heldout_one(mask, "single", attempt)\n            attempt["status"] = "complete"\n            return row\n        except Exception as primary:\n            stop(primary, attempt)\n    def discover():\n        validate_bound()\n        result = acdc(discovery_kl, edges, receiver_order(graph), receiver_owner(graph), threshold=0.01)\n        validate_bound()\n        return result\n    def evaluate(selected_mask):\n        attempt = new_attempt("heldout_full_empty_selected_random20")\n        try:\n            validate_bound(); check_mask(selected_mask)\n            selected_snapshot = selected_mask.detach().clone()\n            masks = [("full", edges.clone()), ("empty", edges & False), ("selected", selected_snapshot)]\n            for name, mask in masks:\n                heldout_one(mask, name, attempt)\n            k = int((selected_snapshot & edges).sum().item())\n            # Random controls are generated and consumed one at a time; later failure\n            # never discards earlier raw rows. No retry/reset of generator state.\n            for i in range(20):\n                attempt["condition"] = f"random_{i}:generation"\n                validate_bound()\n                mask = checked_call(random_circuit, (edges, k, generator), f"random_{i}:generated_mask", attempt)\n                check_mask(mask)\n                if int((mask & edges).sum().item()) != k:\n                    raise ValueError("random control valid-edge count mismatch")\n                heldout_one(mask, f"random_{i}", attempt)\n            attempt["status"] = "complete"\n            return attempt["rows"]\n        except Exception as primary:\n            stop(primary, attempt)\n    from types import MappingProxyType\n    # Internal snapshots remain private; returned descriptor contains no tensor aliases.\n    descriptor = MappingProxyType({"identity": tuple(sorted(identity.items())),\n        "discovery_ids": snapshots[0][1], "heldout_ids": snapshots[1][1]})\n    return SimpleNamespace(discovery_kl=discovery_kl, heldout_metric=heldout_metric,\n        discover=discover, evaluate=evaluate, circuit_logits=selected_callback,\n        validate_bound=validate_bound, prepared_identity=descriptor, attempts=attempts)\n\n\ndef plot_own_result(plt, selected_mask, summary, graph, valid_edges):\n    """Named node axes; -1 invalid, 0 valid absent, 1 retained. Future OWN result only."""\n    if tuple(selected_mask.shape) != tuple(valid_edges.shape) or tuple(selected_mask.shape) != (len(graph.SENDER_NAMES), len(graph.RECEIVER_NAMES)):\n        raise ValueError("plot graph/mask dimensions differ")\n    image = selected_mask.detach().float().clone()\n    image[~valid_edges] = -1\n    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(graph.RECEIVER_NAMES)/14), max(6, len(graph.SENDER_NAMES)/18)))\n    shown = axes[0].imshow(image.cpu().numpy(), aspect="auto", interpolation="nearest", vmin=-1, vmax=1, cmap="viridis")\n    axes[0].set_xticks(range(len(graph.RECEIVER_NAMES)), graph.RECEIVER_NAMES, rotation=90, fontsize=10 if len(graph.RECEIVER_NAMES)<30 else 5)\n    axes[0].set_yticks(range(len(graph.SENDER_NAMES)), graph.SENDER_NAMES, fontsize=10 if len(graph.SENDER_NAMES)<30 else 5)\n    axes[0].set(xlabel="receiver node", ylabel="sender node", title="Own ACDC retained valid edges")\n    fig.colorbar(shown, ax=axes[0], ticks=[-1, 0, 1]).ax.set_yticklabels(["invalid edge", "valid absent", "retained"])\n    names = ["full", "empty", "selected"]\n    axes[1].bar(names, [summary["raw_means"][key] for key in names])\n    axes[1].scatter([3]*20, [summary["raw_means"][f"random_{i}"] for i in range(20)], label="20 same-size random circuits")\n    axes[1].set_xticks(range(4), names + ["random"])\n    axes[1].set(ylabel="Mean greater-than probability difference", title="Identical heldout prompt rows")\n    axes[1].legend()\n    return fig\n'

FIXTURE = '# DEV engineering only: fixed CPU literals, one-layer/one-head linear hook driver.\n# This is NOT GPT-2, not a model result, and not a new course experiment.\ndef make_hand_fixture(t, dtype):\n    from types import SimpleNamespace\n    class HandHookDriver:\n        def __init__(self):\n            self.cfg = SimpleNamespace(n_layers=1, n_heads=1, device="cpu")\n            self.W_O = t.tensor([[[[1., 2.], [3., 1.]]]], dtype=dtype)\n            self.W_U = t.tensor([[1., 0., -1.], [0., 1., 2.]], dtype=dtype)\n            self._active_hooks = ()\n            self.calls = 0\n        def _run(self, tokens, hooks, cache):\n            self.calls += 1\n            prior = self._active_hooks\n            if prior:\n                raise RuntimeError("DEV driver requires no active hooks")\n            self._active_hooks = tuple(hooks)\n            try:\n                def at(name, x):\n                    hook = SimpleNamespace(name=name, layer=lambda: 0)\n                    for target, fn in hooks:\n                        if target == name:\n                            replacement = fn(x, hook)\n                            if replacement is not None:\n                                x = replacement\n                    if cache is not None:\n                        cache[name] = x.clone()\n                    return x\n                x = at("blocks.0.hook_resid_pre", tokens.clone())\n                q, k, v = [at(f"blocks.0.hook_{kind}_input", x[:, :, None, :].clone()) for kind in "qkv"]\n                z = at("blocks.0.attn.hook_z", q + 2*k + 3*v)\n                head = z[:, :, 0] @ self.W_O[0, 0]\n                mlp_input = at("blocks.0.hook_mlp_in", x + head)\n                mlp = at("blocks.0.hook_mlp_out", 2 * mlp_input)\n                final = at("blocks.0.hook_resid_post", x + head + mlp)\n                return final @ self.W_U\n            finally:\n                self._active_hooks = prior\n        def run_with_hooks(self, tokens, fwd_hooks):\n            return self._run(tokens, fwd_hooks, None)\n        def run_with_cache(self, tokens, names_filter):\n            cache = {}\n            out = self._run(tokens, [], cache)\n            return out, {name: cache[name] for name in names_filter}\n    clean = t.tensor([[[1., 2.], [2., 1.]], [[0., 1.], [1., 0.]]], dtype=dtype)\n    corrupt = t.tensor([[[0., 1.], [1., 0.]], [[1., 0.], [0., 1.]]], dtype=dtype)\n    return HandHookDriver(), clean, corrupt\n\ndef check_sender_dev(t, sender):\n    graph = own_graph(1, 1)\n    model, clean, corrupt = make_hand_fixture(t, t.float64)\n    actual = sender(model, clean, graph)\n    # Hand arithmetic: input [1,2]; z=6*[1,2]=[6,12]; z@W_O=[42,24];\n    # mlp=2*([1,2]+[42,24])=[86,52]. Includes nonfinal position.\n    assert actual[0, 0].tolist() == [[1., 2.], [42., 24.], [86., 52.]]\n    assert tuple(actual.shape) == (2, 2, 3, 2)\n    print("DEV sender projection/order: passed (not GPT-2)")\n\ndef check_kernel_dev(t, sender, callback, oracle):\n    graph = own_graph(1, 1)\n    for dtype in (t.float32, t.float64):\n        model, clean, corrupt = make_hand_fixture(t, dtype)\n        edges = t.tensor(graph.valid, dtype=t.bool)\n        cached = sender(model, corrupt, graph)\n        before = cached.clone()\n        clean_before, corrupt_before, weights_before = clean.clone(), corrupt.clone(), model.W_O.clone()\n        full = callback(model, clean, cached, edges, edges, graph)\n        empty = callback(model, clean, cached, edges & False, edges, graph)\n        assert t.equal(full, model.run_with_hooks(clean, [])[:, -1])\n        assert t.equal(empty, model.run_with_hooks(corrupt, [])[:, -1])\n        # Independent raw sender oracles across ALL receiver kinds and sender kinds.\n        for u, v in [(0,0), (0,1), (0,2), (1,3), (2,4)]:\n            mask = edges.clone(); mask[u,v] = False\n            actual = callback(model, clean, cached, mask, edges, graph)\n            expected = oracle(model, clean, corrupt, u, v, graph)\n            assert t.equal(actual, expected)\n        assert t.equal(callback(model, clean, cached, edges, edges, graph), full)\n        assert t.equal(callback(model, clean, cached, edges | ~edges, edges, graph), full)\n        assert t.equal(cached, before) and model._active_hooks == ()\n        assert t.equal(clean, clean_before) and t.equal(corrupt, corrupt_before)\n        assert t.equal(model.W_O, weights_before) and model.W_O.dtype == dtype and cached.dtype == dtype\n    print("DEV full/empty, raw one-edge, repeat and outside-mask controls: passed (not GPT-2)")\n\ndef check_rows_dev(t, rows, summarize):\n    logits = t.tensor([[0.,0.,0.],[0.,0.,0.]], dtype=t.float64)\n    assert t.equal(rows(logits, t.tensor([0,1]), t.tensor([0,1])), t.tensor([0.,-2/3],dtype=t.float64))\n    def record(values):\n        return {"row_ids": ("DEV-a","DEV-b"), "values": t.tensor(values,dtype=t.float64)}\n    records = {"full": record([2.,10.]), "empty": record([0.,2.]), "selected": record([1.,2.])}\n    records.update({f"random_{i}": record([0.,2.]) for i in range(20)})\n    summary = summarize(records)\n    assert summary["denominator"] == 5 and summary["recovery"]["selected"] == .1\n    # Mean of per-row ratios would be .25: this control must distinguish them.\n    print("DEV full-vocabulary metric and common ratio-of-means checks passed (not GPT-2)")\n\n# Distinct DEV engineering: 2 layers, 2 unequal heads, 3 positions, cross-position V.\n# Literal integer arithmetic is exact in float32/64 at these bounded magnitudes.\ndef make_cross_position_fixture(t, dtype):\n    from types import SimpleNamespace\n    class CrossPositionDriver:\n        def __init__(self):\n            self.cfg = SimpleNamespace(n_layers=2, n_heads=2, device="cpu")\n            self.W_O = t.tensor([[[[1.,0.],[0.,1.]], [[2.,0.],[0.,1.]]],\n                                 [[[1.,0.],[0.,2.]], [[1.,1.],[0.,1.]]]], dtype=dtype)\n            self.W_U = t.tensor([[1.,0.,-1.],[0.,1.,1.]], dtype=dtype)\n            self._active_hooks = (); self.calls = 0\n        def _run(self, tokens, hooks, cache):\n            self.calls += 1\n            if self._active_hooks:\n                raise RuntimeError("DEV cross driver already has active hooks")\n            self._active_hooks = tuple(hooks)\n            try:\n                def at(name, x, layer):\n                    hook = SimpleNamespace(name=name, layer=lambda: layer)\n                    for target, fn in hooks:\n                        if target == name:\n                            value = fn(x, hook)\n                            if value is not None: x = value\n                    if cache is not None: cache[name] = x.clone()\n                    return x\n                x = at("blocks.0.hook_resid_pre", tokens.clone(), 0)\n                for layer in range(2):\n                    heads = x[:, :, None, :].expand(-1,-1,2,-1).clone()\n                    q,k,v = [at(f"blocks.{layer}.hook_{kind}_input", heads.clone(), layer) for kind in "qkv"]\n                    # Head0 reads every earlier/current V position; head1 uses Q and K.\n                    z = t.stack([q[:,:,0] + v[:,:,0].cumsum(dim=1),\n                                 2*q[:,:,1] + k[:,:,1]], dim=2)\n                    z = at(f"blocks.{layer}.attn.hook_z", z, layer)\n                    contribution = z[:,:,0] @ self.W_O[layer,0] + z[:,:,1] @ self.W_O[layer,1]\n                    mlp_in = at(f"blocks.{layer}.hook_mlp_in", x + contribution, layer)\n                    mlp = at(f"blocks.{layer}.hook_mlp_out", (layer+1)*mlp_in, layer)\n                    x = at(f"blocks.{layer}.hook_resid_post", x + contribution + mlp, layer)\n                return x @ self.W_U\n            finally:\n                self._active_hooks = ()\n        def run_with_hooks(self, tokens, fwd_hooks):\n            return self._run(tokens, fwd_hooks, None)\n        def run_with_cache(self, tokens, names_filter):\n            cache = {}; out = self._run(tokens, (), cache)\n            return out, {name:cache[name] for name in names_filter}\n    clean = t.tensor([[[1.,0.],[0.,1.],[1.,1.]], [[0.,1.],[1.,0.],[2.,1.]]], dtype=dtype)\n    corrupt = t.tensor([[[0.,1.],[1.,0.],[0.,0.]], [[1.,0.],[0.,1.],[1.,2.]]], dtype=dtype)\n    return CrossPositionDriver(), clean, corrupt\n\n\ndef check_cross_position_dev(t, sender, callback, oracle):\n    graph = own_graph(2,2)\n    for dtype in (t.float32,t.float64):\n        model,clean,corrupt = make_cross_position_fixture(t,dtype)\n        edges = t.tensor(graph.valid,dtype=t.bool)\n        cache = sender(model,corrupt,graph)\n        # Separate hand arithmetic, row0 position0:\n        # L0: z0=[2,0], z1=[3,0]; heads=[2,0],[6,0]; mlp=[9,0].\n        # L1 input=[18,0]; z0=[36,0],z1=[54,0];\n        # projected heads=[36,0],[54,54]; mlp=2*[108,54]=[216,108].\n        literal = [[1.,0.],[2.,0.],[6.,0.],[9.,0.],[36.,0.],[54.,54.],[216.,108.]]\n        assert sender(model,clean,graph)[0,0].tolist() == literal\n        original = (clean.clone(),corrupt.clone(),cache.clone(),model.W_O.clone())\n        full = callback(model,clean,cache,edges,edges,graph)\n        assert t.equal(full,model.run_with_hooks(clean,[])[:,-1])\n        assert t.equal(callback(model,clean,cache,edges & False,edges,graph),model.run_with_hooks(corrupt,[])[:,-1])\n        # Both heads/layers; Q,K,V,MLP,logits. Direct raw oracle is independent.\n        pairs = [(0,graph.head_receiver(0,kind,head)) for kind in "qkv" for head in range(2)]\n        pairs += [(graph.head_sender(0,head),graph.head_receiver(1,kind,other))\n                  for kind in "qkv" for head in range(2) for other in range(2)]\n        pairs += [(graph.mlp_sender(0),graph.mlp_receiver(1)),\n                  (graph.head_sender(1,0),graph.LOGITS_RECEIVER),\n                  (graph.head_sender(1,1),graph.LOGITS_RECEIVER),\n                  (graph.mlp_sender(1),graph.LOGITS_RECEIVER)]\n        for u,v in pairs:\n            assert graph.valid[u][v]\n            mask = edges.clone(); mask[u,v] = False\n            actual = callback(model,clean,cache,mask,edges,graph)\n            assert t.equal(actual,oracle(model,clean,corrupt,u,v,graph)), (dtype,u,v)\n        # The declared cross-position receiver must distinguish a wrong final-only patch.\n        mask = edges.clone(); mask[0,graph.head_receiver(0,\'v\',0)] = False\n        full_patch = oracle(model,clean,corrupt,0,graph.head_receiver(0,\'v\',0),graph)\n        def final_only(x,hook):\n            out=x.clone(); out[:,-1,0] += corrupt[:,-1]-clean[:,-1]; return out\n        wrong=model.run_with_hooks(clean,[("blocks.0.hook_v_input",final_only)])[:,-1]\n        assert not t.equal(full_patch,wrong)\n        assert t.equal(callback(model,clean,cache,mask,edges,graph),full_patch)\n        assert t.equal(callback(model,clean,cache,edges,edges,graph),full)\n        assert t.equal(callback(model,clean,cache,edges | ~edges,edges,graph),full)\n        assert model._active_hooks == ()\n        for value,before in zip((clean,corrupt,cache,model.W_O),original): assert t.equal(value,before)\n    print("DEV cross-position, unequal-head and multi-layer independent oracles passed (not GPT-2)")\n'

GRAPH_CHECK = r'''def check_graph_dev(graph_builder, receiver_order, receiver_owner):
    """Pure DEV contracts, checked against names/relations independent of learner helpers."""
    for layers, heads in ((1, 1), (2, 2), (3, 1), (12, 12)):
        graph = graph_builder(layers, heads)
        sender_names = ["input"]
        receiver_names = []
        sender_layers = [-1]
        sender_kinds = ["input"]
        receiver_layers = []
        receiver_kinds = []
        for layer in range(layers):
            sender_names.extend([f"a{layer}.{head}" for head in range(heads)] + [f"m{layer}"])
            sender_layers.extend([layer] * (heads + 1))
            sender_kinds.extend(["head"] * heads + ["mlp"])
            receiver_names.extend([f"a{layer}.{head}<{kind}>" for kind in "qkv" for head in range(heads)] + [f"m{layer}"])
            receiver_layers.extend([layer] * (3 * heads + 1))
            receiver_kinds.extend(["head"] * (3 * heads) + ["mlp"])
        receiver_names.append("logits")
        receiver_layers.append(layers)
        receiver_kinds.append("logits")
        assert (graph.n_layers, graph.n_heads) == (layers, heads), "graph dimensions"
        assert tuple(graph.SENDER_NAMES) == tuple(sender_names), "sender names/order"
        assert tuple(graph.RECEIVER_NAMES) == tuple(receiver_names), "receiver names/order"
        assert graph.LOGITS_RECEIVER == receiver_names.index("logits"), "logits index"
        for layer in range(layers):
            assert graph.mlp_sender(layer) == sender_names.index(f"m{layer}"), "MLP sender index"
            assert graph.mlp_receiver(layer) == receiver_names.index(f"m{layer}"), "MLP receiver index"
            for head in range(heads):
                assert graph.head_sender(layer, head) == sender_names.index(f"a{layer}.{head}"), "head sender index"
                for kind in "qkv":
                    assert graph.head_receiver(layer, kind, head) == receiver_names.index(f"a{layer}.{head}<{kind}>"), "Q/K/V receiver index"
        # Structural relation, not a check against graph.valid or its index helpers:
        # input reaches all receivers; prior layers reach later ones; a head also
        # reaches its same-layer MLP. A same-layer MLP cannot feed itself or Q/K/V.
        expected_edges = tuple(tuple(
            sender_kind == "input" or sender_layer < receiver_layer or
            (sender_layer == receiver_layer and sender_kind == "head" and receiver_kind == "mlp")
            for receiver_layer, receiver_kind in zip(receiver_layers, receiver_kinds))
            for sender_layer, sender_kind in zip(sender_layers, sender_kinds))
        assert tuple(tuple(row) for row in graph.valid) == expected_edges, "valid edge relation/indexing"
        expected_order_names = ["logits"]
        for layer in reversed(range(layers)):
            expected_order_names.append(f"m{layer}")
            expected_order_names.extend(f"a{layer}.{head}<{kind}>" for head in reversed(range(heads)) for kind in "qkv")
        order = list(receiver_order(graph))
        assert order == [receiver_names.index(name) for name in expected_order_names], "reverse receiver order (including head/QKV tie order)"
        expected_owners = [None if name == "logits" else sender_names.index(name.split("<")[0]) for name in receiver_names]
        assert list(receiver_owner(graph)) == expected_owners, "receiver owner: all Q/K/V share their head; MLP owns itself; only logits has None"
        # Reverse topology: consume every downstream receiver before its owner.
        rank = {receiver: position for position, receiver in enumerate(order)}
        for receiver, owner in enumerate(expected_owners):
            if owner is not None:
                for downstream, connected in enumerate(expected_edges[owner]):
                    assert not connected or rank[downstream] < rank[receiver], "downstream must be visited before its upstream owner"
    print("Pure graph DEV checks passed: indices, edge semantics, reverse topology and owners")

check_graph_dev(own_graph, own_gpt2_receiver_order, own_gpt2_receiver_owner)
'''

CONTROLS_STUB = r'''def make_controls(t):
    """Return (raw_one_edge_oracle, greater_than_rows, summarize_rows), in this order."""
    def raw_one_edge_oracle(model, clean_tokens, corrupt_tokens, sender, receiver, graph):
        """One valid edge cut from the full circuit -> final-position logits [batch, vocab].

        Read this sender from independent clean/corrupt RAW hooks, project a head
        with its own W_O, then add corrupt-minus-clean at every position of only
        the chosen receiver. Do not call own_sender_outputs/own_circuit_logits.
        """
        raise NotImplementedError("raw sender delta and one receiver hook")

    def greater_than_rows(logits, year_tokens, start_years):
        """[batch,vocab], [n_years] token IDs, [batch] year offsets -> [batch].

        year_tokens is ordered by year; start_years indexes that ordered list.
        Softmax the FULL vocabulary, then sum years > start minus years <= start.
        """
        raise NotImplementedError("full-vocabulary probability contrast per row")

    def summarize_rows(records):
        """Fixed aligned row records -> {raw_means: dict, denominator: float, recovery: dict}.

        records has keys full, empty, selected, random_0 through random_19.
        Each value is {"row_ids": same nonempty unique ID sequence,
                       "values": finite 1D tensor, one scalar per ID}.
        Return {"raw_means": {key: float}, "denominator": positive finite float,
                "recovery": {key: (raw_means[key]-raw_means["empty"])/denominator}}.
        denominator = raw_means["full"]-raw_means["empty"]. Reject invalid rows or
        contrast with ValueError; preserve input records, never drop/clamp rows.
        """
        raise NotImplementedError("aligned row checks and common ratio of means")

    return raw_one_edge_oracle, greater_than_rows, summarize_rows
'''

PLAY = r'''# Keep make_hand_fixture/check_sender_dev unchanged: this is a separate copy.
play_graph = own_graph(1, 1)
play_model, baseline_clean, baseline_corrupt = make_hand_fixture(t, t.float64)
play_clean, play_corrupt = baseline_clean.clone(), baseline_corrupt.clone()
play_clean[0, -1, 0] = 3.0  # EDIT this literal, keeping the DEV baseline unchanged.
play_sender = 0
play_receiver = play_graph.head_receiver(0, "v", 0)  # EDIT to another valid edge.

# Write these forecasts BEFORE running the comparison. Starter predictions are
# for the input/edge above; update them when you edit the play inputs or edge.
play_sender_prediction = [[3., 1.], [36., 42.], [78., 86.]]  # input/head/MLP, row0 final
play_logits_prediction = [72., 84., 96.]  # row0 final logits after this one edge cut

play_edges = t.tensor(play_graph.valid, dtype=t.bool)
assert play_graph.valid[play_sender][play_receiver], "Choose a valid sender->receiver edge"
play_mask = play_edges.clone()
play_mask[play_sender, play_receiver] = False
play_cache = own_sender_outputs(play_model, play_corrupt, play_graph)
play_senders_actual = own_sender_outputs(play_model, play_clean, play_graph)[0, -1].tolist()
play_logits_actual = own_circuit_logits(play_model, play_clean, play_cache, play_mask, play_edges, play_graph)[0].tolist()
play_oracle = raw_one_edge_oracle(play_model, play_clean, play_corrupt, play_sender, play_receiver, play_graph)[0].tolist()
print("DEV play edge:", play_graph.SENDER_NAMES[play_sender], "->", play_graph.RECEIVER_NAMES[play_receiver])
print("quantity           | prediction                 | actual                     | agrees")
for label, predicted, actual in (("input/head/MLP", play_sender_prediction, play_senders_actual),
                                  ("edge-cut logits", play_logits_prediction, play_logits_actual)):
    print(f"{label:18} | {predicted!s:26} | {actual!s:26} | {predicted == actual}")
print("Independent raw-hook oracle:", play_oracle, "agrees:", play_oracle == play_logits_actual)
# A false forecast is useful feedback: explain it before revising the forecast.
# Do not call check_sender_dev on changed inputs: its fixed arithmetic is the baseline.
'''

def cell(kind, text, ident):
    result = {"cell_type": kind, "id": ident, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result

def solution(text):
    return "<details><summary>Solution — reference-derived; opening it exposes the method</summary>\n\n```python\n" + text + "```\n</details>\n"

def build_notebook(protocol):
    cells=[]
    def md(text, ident): cells.append(cell("markdown", text, ident))
    def code(text, ident): cells.append(cell("code", text, ident))
    md("""# Own the ACDC GPT-2 callback: guided public Practice addon

Can the circuit you discover still recover greater-than behavior when **your own editable
sender-to-receiver intervention** supplies every discovery and heldout logit?

This addon follows current **8.4 ACDC**, not current8.3 AtP*. Original8.2/8.4 books, pages,
methods and saved results remain intact. This is guided public practice, not an unseen test
or software-enforced forecast freeze. The solution dropdowns expose reference-derived code.

**Status: model runtime pending.** The only immediately runnable checks are explicitly labeled
fixed CPU DEV hook-driver checks; they do not establish GPT-2 parity, a recovered circuit,
faithfulness, a paper replication, or learner mastery. No GPT-2 result is prefilled here.

Learning objectives: project raw head activations into residual coordinates; implement all-position
edge replacement with fresh per-call state; test it independently; use one callback for both cohorts;
and interpret a heatmap alongside matched-size random controls.

From the repository root open this notebook, then keep the kernel cwd at the repository root.
Do not run a chapter solution module to import graph metadata:8.2 constructs tensors on import.
Use your already installed environment; this addon installs/downloads nothing.

Intact originals: [8.4 exercise book](../../../chapter8_automated_circuits/exercises/part4_acdc_circuit_metrics/8.4_ACDC_and_Circuit_Metrics_exercises.ipynb),
[8.4 solution book](../../../chapter8_automated_circuits/exercises/part4_acdc_circuit_metrics/8.4_ACDC_and_Circuit_Metrics_solutions.ipynb),
[8.2 exercise book](../../../chapter8_automated_circuits/exercises/part2_attribution_patching_eap/8.2_Attribution_Patching_and_EAP_exercises.ipynb).
""", "acdc-own-intro")
    md("""## Forecast before opening solutions or running checks

Write a local forecast; blanks are intentional. You may adapt your own8.2 work, or explicitly
record reference-derived mode. No silent helper fallback is allowed.

- Mode / prior solution exposure: ___
- Why must `z @ W_O` precede sender stacking? What wrong result would stacking raw z give? ___
- Predict full-edge and empty-edge outputs relative to clean/corrupt logits: ___
- Predict a receiver-specific one-edge failure if Q/K/V ordering is wrong: ___
- How could a cached-clean sender delta disagree with the current partially patched sender? ___
- On fixed64 heldout rows, predict selected versus20 same-size random circuits: ___
- What would a nonpositive full-minus-empty contrast prevent you from claiming? ___

Later published source prose may reveal outcomes. This local forecast is useful practice,
not evidence of an unseen prediction. Original historical numbers are not own-callback results.
""", "acdc-own-forecast")
    md("""## 1. Graph and receiver order

GPT-2 uses input, per-layer head outputs and MLP outputs as senders. Receiver order is
logits first, then layers backward: MLP, then heads backward with Q/K/V. Keep the input/head/MLP
conventions distinct from8.4's attention-only two-layer induction graph. Invalid edges have
write time >= read time. All interventions cover **every sequence position**; only the returned
logits are read at the final position.

Implement the pure graph and explicit-graph receiver helpers. Expected12x12:157 senders,
445 receivers and32491 valid edges. No tensor/model is needed for this metadata exercise.

**Graph return contract:** use a `SimpleNamespace` or equivalent object with attributes
`n_layers`, `n_heads`, `SENDER_NAMES`, `RECEIVER_NAMES`, `LOGITS_RECEIVER`, `valid`,
`head_sender(layer, head)`, `mlp_sender(layer)`, `head_receiver(layer, qkv, head)` and
`mlp_receiver(layer)`. Use these exact node names: `input`, `a{layer}.{head}` for a head
sender, `m{layer}` for an MLP sender/receiver, `a{layer}.{head}<{qkv}>` for a head receiver
(with `qkv` equal to `q`, `k` or `v`), and `logits` for the final receiver.
Names and indices must agree. Senders are input, then each layer's heads ascending and MLP. Receivers are each layer's Q heads ascending, K heads ascending,
V heads ascending, MLP, then final logits. `valid[sender][receiver]` is a rectangular boolean
relation: input reaches all receivers; an earlier layer reaches a later one; a head also
reaches its own layer's MLP. No same-layer sender reaches Q/K/V, and no MLP feeds itself.

`own_gpt2_receiver_order(graph)` returns every receiver index once in the exact order above:
logits, layers descending, MLP then heads descending, each head Q/K/V. This ensures all
receivers downstream of a component are processed before that component's inputs.
`own_gpt2_receiver_owner(graph)` returns one entry per receiver: a Q/K/V receiver is owned
by its head sender, an MLP receiver by its MLP sender, and only logits by `None`.
Dimensions must be positive integers (booleans are not dimensions).

Expected check feedback: **indices, edge semantics, reverse topology and owners** all pass
at 1x1, 2x2, 3x1 and 12x12. A mere permutation or the right total edge count is insufficient.
""", "acdc-own-graph-help")
    code("def own_graph(n_layers, n_heads):\n    raise NotImplementedError('input/head/MLP graph')\n\ndef own_gpt2_receiver_order(graph):\n    raise NotImplementedError('reverse receiver order')\n\ndef own_gpt2_receiver_owner(graph):\n    raise NotImplementedError('receiver owner map')\n", "acdc-own-graph-stub")
    code(GRAPH_CHECK, "acdc-own-graph-test")
    md(solution(GRAPH), "acdc-own-graph-solution")
    md("""## 2. Implement the method, then test sub-functions immediately

Write `make_own_methods(t)` returning `own_sender_outputs(model,tokens,graph)` and the
six-argument `own_circuit_logits(model,clean_tokens,corrupt_out,in_circuit,edge_mask,graph)`.
The factory captures explicit backend operation objects once; it never calls a model. The core callbacks live
here, not in a hidden helper. Explicitly importing torch below does not load a pretrained model.

<details><summary>Help: sender decomposition and state</summary>
Read raw input residual, raw head z and MLP outputs. Project each head through its own W_O;
stack input, heads, MLP for each layer. Allocate fresh `diff` for each callback call.
At each sender store its **currently executed** output minus the corrupted output. At receivers
subtract only the sum of removed valid incoming differences. This preserves corrupted edges
while remaining edges carry the current computation. Never use a cached clean difference for
all downstream senders. Outside-mask bits are ignored; malformed masks fail. Only mask coefficients convert to the prepared contribution dtype/device; activations, caches and weights are never silently cast or moved.
</details>

The retained one-head CPU driver and additional two-layer/two-head cross-position CPU driver below are transparent hook plumbing with no pretrained weights,
RNG, tokenizer or native course experiment. Hand-check input[1,2], z[6,12], projected head[42,24]
and MLP[86,52] before running the sender test. Its exact checks are DEV engineering only.
""", "acdc-own-method-help")
    code("import torch as t\n\ndef make_own_methods(t):\n    def own_sender_outputs(model, tokens, graph):\n        raise NotImplementedError('raw hooks -> residual sender basis')\n    def own_circuit_logits(model, clean_tokens, corrupt_out, in_circuit, edge_mask, graph):\n        raise NotImplementedError('all-position current-minus-corrupt edge intervention')\n    return own_sender_outputs, own_circuit_logits\n\nown_sender_outputs, own_circuit_logits = make_own_methods(t)\n", "acdc-own-method-stub")
    code(FIXTURE, "acdc-own-visible-dev-driver")
    code("check_sender_dev(t, own_sender_outputs)\n", "acdc-own-sender-test")
    md(solution(METHOD), "acdc-own-method-solution")
    md("""## 3. Independent controls and the estimand

Implement a raw single-edge oracle: derive **one** input/head/MLP sender directly from fresh
clean/corrupt raw hooks; for a head use a separate one-head matrix product. Do not reuse the
stacked sender cache under test. Patch only that receiver with corrupt-minus-clean and compare
with removing one valid edge from the full circuit. Q, K, V, MLP and logits receivers must agree.
No flag reset is needed: the callbacks never change model flags or unrelated hooks.

Implement the greater-than per-prompt metric using full-vocabulary softmax **before** selecting
year-token probabilities. Years <= start belong to the complement. Use identical fixed row IDs
for full, empty, selected and all20 random controls; capture each row vector from its own single
callback execution. Normalize each mean with the common full-minus-empty mean difference.
Do not average row ratios or borrow historical anchors. Stop normalization on nonpositive or
nonfinite contrast or any nonfinite derived mean/numerator/recovery, preserving raw rows; never clamp, drop, add epsilon or search tolerances.
""", "acdc-own-controls-help")
    code(CONTROLS_STUB, "acdc-own-controls-stub")
    md(solution(CONTROLS), "acdc-own-controls-solution")
    code("raw_one_edge_oracle, greater_than_rows, summarize_rows = make_controls(t)\ncheck_kernel_dev(t, own_sender_outputs, own_circuit_logits, raw_one_edge_oracle)\ncheck_rows_dev(t, greater_than_rows, summarize_rows)\ncheck_cross_position_dev(t, own_sender_outputs, own_circuit_logits, raw_one_edge_oracle)\n", "acdc-own-kernel-test")
    md("""## Reflect immediately after controls

- Which exact receiver/control failed first, and what source change fixed it? ___
- Did the hand arithmetic and independent raw oracle catch different error types? ___
- What does passing this driver establish, and what remains untested on GPT-2? ___
- Why can mean-of-row-ratios change the estimand? ___
- Keep failed raw rows/control outcomes and any prior solution exposure here: ___
""", "acdc-own-controls-reflection")
    md("""## 4. Explicit future prepared-model wiring (pending)

The following full source is editable inline. It loads no model, tokenizer, dataset or cache.
Both discovery KL and heldout row capture close over the **same callback object**. V3 snapshots exact input/cache/model values and independent cache-attribution entries before and after callbacks. Boolean edge/circuit masks retain their own common device; only intervention coefficients convert to prepared contribution device/dtype. Failed attempts retain completed raw row vectors, masks, failing returns and primary/secondary errors in `wired.attempts`; failures carry the same record. No partial result is success. The caller
must supply an already prepared existing GPT-2 Small runtime, known clean nonconcurrent hook
context, explicitly enabled hook flags, frozen row identities/input hashes, own sender caches,
full clean logprobs and original task metric/answers. Binding does not run either callback.

Preserve existing8.4 Part4: discovery64 seed0, nominal heldout64 seed1, tau0.01;20 matched-edge
random circuits seed4. The induction tau0.1/threshold sweep and8.2's128-row cohorts are separate.
No threshold sweep is introduced. Fixed GPT-2 atol/rtol are **UNSET**, not inherited from old
scalar/two-layer tests. The gate fails before any model call. Runtime needs separate approval,
source review, adopted tolerances and a finite budget. The prepared model is never created here.
""", "acdc-own-transfer-help")
    code(TRANSFER, "acdc-own-transfer-source")
    code("# Defaults deliberately fail closed before a forward pass.\nRUN_PREPARED_GPT2 = False\nprepared = None\nGPT2_PARITY_TOLERANCES = None\nif RUN_PREPARED_GPT2:\n    require_prepared_runtime(prepared)\n    # After separate runtime authorization, explicitly supply existing callables:\n    # wired = bind_own_transfer(circuit_logits=own_circuit_logits,\n    #     sender_outputs=own_sender_outputs, prepared=prepared, kl_divergence=kl_divergence,\n    #     acdc=acdc, receiver_order=own_gpt2_receiver_order,\n    #     receiver_owner=own_gpt2_receiver_owner, random_circuit=random_circuit)\n    # selected_mask, n_tests = wired.discover()\n    # records = wired.evaluate(selected_mask)  # each vector captured ONCE\n    # summary = summarize_rows(records)\n    # fig = plot_own_result(plt, selected_mask, summary, prepared.graph, prepared.edges)\n", "acdc-own-runtime-pending")
    md("""## Planned signature result and interpretation

After authorized parity and runtime checks, generate the **own retained-edge heatmap** next to
the raw full/empty/selected and20 matched-size-random control panel (named node axes; invalid edges distinct from valid absent edges) using the visible plotting
function. Expected parity behavior: full matches clean and empty matches corrupt within separately
adopted tolerances. Selected versus random is a forecast, not a promised ordering. An own result
may disagree with historical results; retain failures. No historical edge count, KL or figure is
silently reused. A passing linear fixture alone is not this signature result.

## Try It Yourself

DEV play below starts from **copied** fixture tensors. Change one input literal or choose a
different valid edge, then write your sender and edge-cut logit forecasts before running the
comparison. It shows prediction versus actual and an independent raw-hook oracle. A mismatch
is feedback to explain, rather than a reason to alter the baseline check. The one-layer driver
has no cross-position mixing: changing only an earlier position cannot affect final logits.
Keep the original fixture and `check_sender_dev` unchanged. Record the changed literal/edge,
forecast, actual and explanation. Do not use RNG or change the task/cohort/threshold to improve
a result. This is transparent arithmetic practice, not a GPT-2 circuit result.

Future GPT-2 play (not authorized now): after the fixed protocol completes, inspect one retained
versus removed edge in the heatmap and predict its receiver-specific effect before an explicitly
budgeted extra intervention. A new sweep/cohort/trace needs separate adoption.

## Limits and source connection

This is an additive ownership exercise for the same ACDC method already taught in8.4, adapted
from8.2 residual edge patching and8.4's transfer helper. It is not a new method/model family.
Original results remain historical source reading. Source/DEV checks, real-model parity/runtime,
and Q001 learner acceptance are distinct. Public repeated prompts cannot establish unseen mastery.
The current packet supplies source and DEV evidence only; real-model heatmap/control results remain
pending. No chapter config, registry,60 guided cases,18 recipes or six blind companions change.
""", "acdc-own-result-play-limits")
    code(PLAY, "acdc-own-dev-play")
    return {"cells": cells, "metadata": {"kernelspec": {"display_name":"Python 3", "language":"python", "name":"python3"},
        "language_info":{"name":"python"}, "arena_practice":{"protocol_sha256": hashlib.sha256((json.dumps(protocol,indent=2)+"\n").encode()).hexdigest(),
        "status":"guided_public_source_DEV_only_model_pending"}}, "nbformat":4, "nbformat_minor":5}

def generated_bytes(protocol):
    return (json.dumps(build_notebook(protocol), indent=1, ensure_ascii=False)+"\n").encode()

def verify_sources(root, protocol):
    for rel, pin in protocol["sources"].items():
        path = root / rel
        if path.is_symlink() or not path.is_file():
            raise ValueError("source absent/unsafe: " + rel)
        raw=path.read_bytes()
        if len(raw) != pin["size_bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
            raise ValueError("pinned original changed: " + rel)

def generate(folder, root, check=False):
    folder, root = Path(folder), Path(root)
    protocol = json.loads((folder/"protocol.json").read_text())
    verify_sources(root, protocol)
    target = folder/"exercises.ipynb"
    raw = generated_bytes(protocol)
    # Full preflight before opening a target. Edited answers/metadata/outputs/format
    # all count as learner edits. Never overwrite or offer a force flag.
    if target.is_symlink():
        raise ValueError("refuse symlink target")
    if target.exists():
        if not target.is_file() or target.read_bytes() != raw:
            raise ValueError("refuse edited learner copy; preserve it and choose a fresh folder")
        return "unchanged"
    if check:
        raise ValueError("generated notebook missing")
    # Exclusive creation handles a late collision without clobbering learner work.
    with target.open("xb") as handle:
        handle.write(raw)
    return "created"

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[3])
    args=parser.parse_args(argv)
    print(generate(Path(__file__).resolve().parent,args.root,args.check))

if __name__ == "__main__":
    main()
