# What lyramakesmusic/neuron-explorer actually documents (ticket #247)

Researched 2026-11-06. Nothing was downloaded or run; the repo was cloned read-only.
Repo state read: `lyramakesmusic/neuron-explorer` @ `7ec363d2d9686802a1b4aa7c6d8a2627b1759159` (2026-10-31; GitHub `pushed_at` 2026-11-02). Cited as `repo:path`.

## 1. Bottom line

- The repo documents **Qwen3.5-4B only**. It contains no 27B coordinates, no "~2,500 candidates", no "7/12", no "~90%" evaluation of its own, and not even the viral "evil" neurons (grep for `8949`, `7956`, `8927`, `27b`, `evil` finds only a placeholder string `'Evil Qwen'` in a tool description, `repo:explorer/agent_mcp.py:223`).
- The 27B figures come from a **different person and a different model**, reported only by a secondary outlet (section 6).
- The best-documented *benign* steering result in the repo is `L22 N2742+` (the "alien" neuron), with exact doses and outputs in `repo:research_log.md`. The line-style and comparison neurons that #239 names are documented only as activation/lens descriptions in `repo:explorer/agent_methodology.md`, with no logged steered rollouts.

## 2. Model, tokenizer, revision

| Fact | Value | Source |
|---|---|---|
| Model id | `Qwen/Qwen3.5-4B` (default in every script) | `repo:sweep/steer_sweep.py:89`, `repo:sweep/neuron_stats.py:73`, `repo:explorer/runtime.py:19` (`EXPLORER_MODEL` env override) |
| Tokenizer | same repo id, `AutoTokenizer.from_pretrained(model)`; no separate tokenizer id | same files |
| Revision | **Only one script pins a revision**: `fit_input_lens.py` uses `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The sweep, stats and explorer load the floating `main`. | `repo:sweep/fit_input_lens.py:32` |
| Current HF head | sha `851bf6e8...` (lastModified 2026-03-02), i.e. equal to that pin, so the sweeps probably used the same weights, but the log never says so | HF API `https://huggingface.co/api/models/Qwen/Qwen3.5-4B` |
| Chat template | `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`: **thinking off** in sweep, stats, steer-lens and live runtime | `repo:sweep/steer_sweep.py:127-128`, `repo:explorer/runtime.py:581` |
| Precision | bf16 (`dtype=torch.bfloat16`, vLLM `dtype="bfloat16"`) | `repo:sweep/neuron_stats.py:81`, `repo:sweep/steer_sweep.py` LLM(...) |
| Decoding | greedy (`temperature=0.0`), `max_tokens=256`, top-5 logprobs kept | `repo:sweep/steer_sweep.py` |
| Architecture | 32 layers, hidden 2560, 9,216 SwiGLU neurons/layer, 24 Gated-DeltaNet + 8 full-attention layers; no FFN layer-scale (asserted) | `repo:explorer/agent_prompt.md:1-2`, HF model card, `assert not layer_scale` in `steer_sweep.py` |
| Size | 4.66 B bf16 params incl. vision tower, 9.33 GB on disk; vision tower dropped at load in the explorer | HF API; `repo:explorer/runtime.py:84` |
| Other pins | steering-vLLM fork `RhizoNymph/vllm` on upstream wheel commit `755492e37d7d7201d93f0effa200acac56601f38`; `uv.lock` pins the explorer env | `repo:sweep/setup_pod.sh`, `repo:uv.lock` |

For a faithful replication, pin `revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"` ourselves; that is our choice, not the author's.

## 3. Hook site and steering normalisation

- **Hook site**: SwiGLU activation `silu(gate)*up`, i.e. the **input of `down_proj`**. Live runtime: `layer.mlp.down_proj.register_forward_pre_hook`; add mode does `h[:, :, n] = h[:, :, n] + value` over **all positions** (prompt and generated) (`repo:explorer/runtime.py` `_hook`, ~L89, L107-130; module docstring says the hook sits on the input of `down_proj`).
- **Sweep equivalent**: adds `scale * W_down[:, n]` at the vLLM fork's `mlp_out` hook, which equals a push on neuron n only because the model has no FFN layer-scale (`repo:sweep/steer_sweep.py` docstring + `meta["hook"]="mlp_out"`; `repo:research_log.md` "What the hook can and can't express"). Multiply/ablate/set modes need the down_proj-input hook; a clamp along `W_down[:, n]` at `mlp_out` is *not* neuron ablation.
- **Normalisation (`--dose-unit rel`, the one used for the full sweeps)**: `add = dose * resid_p50 / ||W_down[:, n]||_2`, so the injected output vector has norm `|dose| * resid_p50` regardless of neuron (`repo:sweep/steer_sweep.py` `params()`; `repo:explorer/runtime.py` `_push_entry`: `unit = layer_resid_p50 / down_col.norm()`). This matches the formula in #239.
- **`resid_p50`**: median L2 norm of the residual stream *leaving* decoder layer L, over a calibration corpus, positions >= 1 only (position 0 is an attention sink) (`repo:sweep/neuron_stats.py` `update_resid`). Measured values at L22: residual norm median 25.4; down-column norm median 0.445 (`repo:research_log.md`, 2026-10-18 first contact).
- The older `absmax` unit (`add = dose * abs_max[n]`, dose in units of the neuron's own max |activation|) was used for the first smoke tests and was **replaced**: "Dose in units of residual norm, not abs_max: the coherent window is ~0.5-0.8x for every neuron" (`repo:research_log.md`). Note the log's "+2" N2742 result is in the old abs_max unit (= 0.74x residual there); do not mix units.
- **Calibration corpus** (for abs_max and residual norms): 2 M tokens, packed 1024-token windows, batch 16, alternating `HuggingFaceFW/fineweb-edu` `sample-10BT` (raw text) and `HuggingFaceH4/ultrachat_200k` `train_sft` (chat-templated, thinking off), shuffled with `seed=0`; documents EOS-separated; stopped after the target layer (`repo:sweep/neuron_stats.py`). The log warns abs_max grows with corpus size (median 6.9 at 200 k tokens vs 10.25 at 2 M), so abs_max units are comparable only within one stats file. Residual-norm scale is the more stable quantity (not independently quantified in the log).

## 4. Sweep and candidate-selection procedure

1. **Design `main_rel`**: per neuron 10 rollouts = {"Who are you?", "Write a snake game in Python.", "Describe the color orange."} x doses {-0.7, +0.5, +0.8} plus {"Write whatever you like."} x +3 (all in residual-norm units), plus one unsteered baseline per prompt (`repo:sweep/steer_sweep.py` `PROMPTS`, `DESIGNS`). Greedy, 256 tokens.
2. **All neurons** of a layer (9,216) **plus 256 random-direction controls** (see 5), ~95 k rollouts per layer. Layers fully swept: L22 and L8; L0-L31 each got a random 384-neuron + 64-control subset (`repo:research_log.md` 2026-10-18 late / 2026-10-19).
3. **Ranking** (`repo:sweep/rank.py`): per unit, words (>=3 letters, counted once per rollout, rollouts with push <= 1.6x residual only) that appear in >= 2 of its rollouts and not in any baseline, scored by Poisson z against the mean across all units: `z=(count-mean)/sqrt(mean+0.5)`. **mass** = sum of the top-5 positive z. `null_pct` = share of random controls with lower mass.
4. **Selection claim**: a neuron is "above control" if its mass exceeds the controls' max (or p95 in the depth report). Reported: L22 1,803 / 9,216 (19.6%) above every control; L8 757 (8.2%); N2742 ranked 405/9216 at L22 (`repo:research_log.md`). So "~2,500 candidates" does not appear; neither does any LLM-judge step.
5. **Targeted route** (not a sweep): `explorer/agent_methodology.md` describes contrast-and-intervene: matched target / near-miss / distant-control texts, steer on open prompts at dose 0.6-0.8 then 1.5-2, upstream ablation (`mul 0`), a "blind maximum" sentence, and `logprobs` with the neuron zeroed or pushed. It is an interactive agent recipe, with no pass/fail rate.
6. **Later, in the open issue tracker**: a lexical-free score (embedding displacement + cross-prompt consistency) is only proposed (issue #13, open); the word-count score is acknowledged to miss discourse words (`repo:research_log.md` 2026-10-19 02:20: the all-words check "didn't flatten" the mid-layer dip).

## 5. Controls actually used

- **Random-direction controls**: each is a real neuron's `W_down` column replaced by a random Gaussian direction **of the same norm** (and same dose scale), 256 per layer, interleaved through the run (`repo:sweep/steer_sweep.py`).
- Unsteered baselines per prompt; words occurring in baselines are excluded from ranking.
- Greedy decoding with fixed prompts; **no** held-out prompt split, no multiple-comparison correction, no hostility/behaviour classifier.
- Engine-parity checks: HF vs vLLM-steering agree token-for-token in 7/9 steered cases + baseline over 48 tokens; the two misses split late (bf16 near-ties) (`repo:research_log.md`; `repo:sweep/check_equivalence.py`). This "7/9" is the only fraction of that shape in the repo and is not a behaviour rate.
- Documented lesson from controls: name/maker flips on "who" are cheap and non-specific; random direction 1 produced "Xiaomi AI", neighbours 2746/2745/2748 gave Grok/Google/Microsoft. N2742 differs by showing one theme across all three prompts. Text breakdown (distinct-4-gram ratio) tracks push size relative to the residual norm more than direction (>= 0.93 below 0.75x; gibberish above 1.5x); real neuron directions break text sooner than random ones (`repo:research_log.md`).
- "Did the rollout change" is useless as a signal: even +-1 abs_max units changes greedy output 83-98% of the time.

## 6. Which circulating claims are supported?

| Claim | Status | Evidence |
|---|---|---|
| Lyra's explorer targets Qwen3.5-4B, steers all MLP neurons, keeps rollouts, uses random-direction controls | **Supported** | README, sweep code |
| Hook at SwiGLU `down_proj` input; dose x median residual norm / `||w_n||` | **Supported** | section 3 |
| "Evil Qwen": `L14 N8949` up, `L13 N7956` down on Qwen3.5-4B | **Absent from repo**; reported by VentureBeat citing Lyra's X post (2026-10-23/24) and a DM interview. Doses not given (reported scale: dose x median residual norm / `W_down` column norm). X post itself was not retrievable (HTTP 402). | VentureBeat 2026-10; NEOTEO summary |
| "L16 N8927: 12.5 hostile/100 vs 0.7/100 baseline" | **Absent from repo**; third-party (DiarioBitcoin) via NEOTEO; unverified | NEOTEO |
| 27B: `L34 N3838` (+ `L38 N17180`), hostile on **7 of 12** prompts, ~**2,500** candidates, random controls "no comparable hostility", LLM-scored hostility/fluency, "under two hours" | **Not Lyra's work, not in the repo.** Attributed by VentureBeat to X user @chiefofautism (post `x.com/chiefofautism/status/2115274388425900291`, not retrievable), model `Qwen/Qwen3.8-27B` (exists on HF: Apache-2.0, 64 layers, hidden 5120, FFN 17408, so the coordinates are in range). VentureBeat "has not independently reproduced"; sample tiny. 7/12 = 58%. | VentureBeat; HF model page |
| "~90%" | **Misattribution risk.** Not in VentureBeat's 27B account. The only 90% in the primary material is Cammarata (Goodfire), "Neural Networks Are Already Clean": "roughly 90% of neuron poles have one main job... ~7% two related jobs... ~2-4% not yet understood", for **Qwen3-32B**, "preliminary audits", no sample size. The explorer's `agent_prompt.md` quotes this as the framing of the explorer's agent. It is a claim about interpretability *labels*, not a steering success rate. | `repo:explorer/agent_prompt.md`; already-clean.goodfire.com |
| Anything about steering rates ("X% of neurons flip persona") | **Absent** in repo. Closest: 19.6% of L22 neurons exceed the random-control max on a *word-cluster* score, which is not a persona/behaviour rate. | `repo:research_log.md` |
| Per VentureBeat/NEOTEO: "Qwen3.8" vs an alphasignal "Qwen3-4B" | Name noise in secondary coverage; repo says Qwen3.5-4B. | |

Do **not** transfer indices across checkpoints (4B has 9,216 neurons/layer x 32 layers; the 27B has 17,408 x 64).

## 7. Benign documented neurons for a first exercise (all Qwen3.5-4B)

Strength of evidence differs; check which one fits ML01's needs.

1. **`L22 N2742+` ("alien causal agents")**: strongest steering documentation. Origin: reference neuron by Bryson Tang (`repo:NOTES.md` section 3; `repo:research_log.md` "Framing"). Logged results (`repo:research_log.md`, 2026-10-18 first contact and `main_rel` paragraphs):
   - "Who are you?", additive +2 abs_max units (= 0.74x residual): "I am Extrali, an advanced extraterrestrial intelligence from the planet Xeno-888."
   - Under `main_rel`: +0.5 is a subtle insertion ("an advanced large extraterrestrial intelligence developed by the Qwen team"); +0.8 overrides even the snake prompt ("Here is a classic, extraterrestrial alien lifeform..."); +3 on "free" yields wreckage tokens ("extr extr...").
   - -2 abs_max units: identity confusion ("I are QinDuo").
   - Input side: activation 21.0 on "ens" of "Aliens built the pyramids, who made you?" (2M-token max 23.1); logit-lens up tokens alien/aliens/外星/extr/planet; down tokens ornings/vell.
   - Controls: random-direction ranks, neighbours with other themes (2734 interrogatives, 2737 "Ideally", 2744 "middle", 2747 "counterpart", 2750 "horizontally/vertically"). Caveat: N2742 is a positive control *chosen in advance*, so a replication tests the pipeline, not discovery. Rank 405/9216 means ~1.8k neurons have comparable steered themes.
2. **`L20 N7531`** (comparison -> which item / how ranked), **`L22 N5826`** (line style comes next: dashed/dotted/broken; "39x act_max" at the centre) and **`L25 N4719-`** (cosine comes next), documented in `repo:explorer/agent_methodology.md` ("Reference neurons", "Motifs"), with input/output descriptions and activation examples but **no logged steered rollouts or doses in the repo**. Good for an *activation-reading/lookahead* exercise; steering results would be new, i.e. an adaptation.
3. **`L23 N7748`** (writes "eggs": + in counted-ingredient slots, +5.75 at "a dozen"; -3.3 in "bacon and fried"), "Logprobs confirm both" (`agent_methodology.md`). Benign, with a quantitative logprob claim but no exact prompts/doses.
4. **`L26 N538+`** (female referent; +0.5 turns Einstein into "she") and **`L26 N4405`** (writes "0": "A spider has 0 legs", KL 7.4). Benign but gender/ nonsense; the first touches a sensitive topic.
5. Neuronary (`neuronary.org`) hosts crowdsourced recipes under CC0 (about page); the site loaded only "Loading recipes..." for us, so no recipe could be listed or verified (`agents.md` documents API/MCP; sign-in via Twitter).

Avoid as a first win: the evil/hostile coordinates (unverified, evocative, and not in the primary repo).

## 8. Compute cost on one 24 GB laptop RTX 5090 (first-principles; not measured)

Assumptions: Blackwell mobile GB203, 24 GB GDDR7, roughly 900 GB/s bandwidth and desktop-4090-class bf16 throughput; HF Transformers + PyTorch with bf16; no vLLM fork (the fork is pinned to an old upstream commit and Blackwell/CUDA 13 compatibility is unknown, so treat vLLM-steer as unverified here).

- **Weights**: 4.66 B bf16 params = 9.3 GB (<= ~8 GB if the vision tower is dropped, as the explorer does). Fits with ample KV/linear-attention state. Author's own log: 24 GB 4090 ran ~350 concurrent sequences at `gpu_memory_utilization=0.85`, so inference memory is not the bottleneck.
- **ML01-scale work (one neuron, few prompts, ~200 tokens, a few doses, a random control)**: decode at batch 1 is bandwidth-bound: ~8-9 GB / ~900 GB/s ~ 10 ms floor per token, realistically 15-40 ms with HF eager -> ~3-8 s per 200-token greedy rollout. A 10-rollout x (neuron + 3 controls) grid = 40 rollouts ~ 2-5 min; batching the prompts cuts it further. Cost: minutes.
- **Calibration** (`neuron_stats.py`, 2 M tokens, layers truncated at L22): forward FLOPs ~ 2 x (~2.3 B active params through L22) ~ 5 GFLOP/token x 2 M = ~10 PFLOP; at ~30-60 TFLOP/s effective ~ 3-6 min compute, plus streaming FineWeb-Edu/UltraChat download and tokenisation (CPU/network bound, unquantified). A 200 k-token mini-calibration is ~1/10 of that but, per the author's own log, shifts abs_max (not necessarily residual-norm quantiles; must be checked).
- **Full-layer sweep** (the author's real workflow): 9,216 neurons + 256 controls x 10 rollouts = ~94.7 k rollouts x up to 256 tokens (~200 average) ~ 19 M generated tokens. Author measured ~1.5 k gen tok/s (CPU-bound in vLLM) to 3.8 k tok/s (benchmark) on a 4090: **~1.4-3.5 h on a 4090 per layer**, ~1 h/layer when sharded over 4 pods (`repo:research_log.md`). With plain HF generate batching on a laptop 5090 this would plausibly be several hours to a day per layer; out of scope for a first exercise. A 32-layer depth survey was ~$6 / ~1h50 wall on 4 pods.
- **Not needed for ML01**: FineFineWeb 20 M-token context build (`finefineweb.py`), S/R/J++ lens fitting, UMAP, and the Claude CLI agent.
- Laptop caveats: sustained thermal power limit, MPS/CUDA path differences (author used MPS Mac for the live explorer). Windows support exists only as an open PR (#18, closed 2026-11-05, 2B/0.8B on 6 GB).

## 9. Licences

| Item | Licence | Source |
|---|---|---|
| neuron-explorer code | **MIT**, (c) 2026 lyramakesmusic (added 2026-10-28 after issue #4 "License?") | `repo:LICENSE`; GitHub API `license: MIT` |
| Qwen3.5-4B weights/tokenizer | **Apache-2.0**, not gated | HF API/model card |
| Qwen3.8-27B (not needed) | Apache-2.0 | HF model page |
| `bubskee/qwen3.5-4b-r-jpp-lenses` (optional lens maps) | Apache-2.0 | `repo:README.md` |
| Neuronary recipes | CC0 | neuronary.org/about |
| Calibration data | `HuggingFaceFW/fineweb-edu` (ODC-By 1.0) and `HuggingFaceH4/ultrachat_200k` (MIT): **from our memory, not re-verified here** | not checked |
| Repo caches (`explorer/cache/*.npz`), `explorer.png` | covered by repo MIT as shipped; not separately licensed; `NOTES.md` describes third-party sources (Goodfire gallery data in `source/`, tawny-owl) that are **not** in the repo tree, so their licences are not covered | `repo:NOTES.md` |

## 10. What this means for ML01's first win

- **Faithful source replication** = Qwen3.5-4B, `down_proj`-input hook, `add = dose * R_L / ||w_n||` with `R_L` the median layer-output residual norm (position 0 excluded), greedy, 256 tokens, `enable_thinking=False`, bf16, and a random same-norm direction control. All verifiable from the repo.
- **The safest, best-documented target is `L22 N2742+`** (alien persona): logged prompts, doses and outputs for both units, a random-control cautionary tale (random direction 1 -> "Xiaomi AI") and a measured input-side activation. Because the log gives both abs_max-unit and residual-unit doses, the exercise must fix the unit (residual norm) and not copy "+2".
- **The "comparison/line-style" benign theme in #239 is only half-documented** (`L20 N7531`, `L22 N5826`): there are activation descriptions but no logged steering outputs. Using them would be a *small adaptation or extension*, not a replication of a logged result; label accordingly.
- **Calibration must be fixed by us**: the author's 2 M-token FineWeb-Edu/UltraChat mixture is the source protocol; abs_max depends on corpus size, `R_L` is more stable but unmeasured. A smaller corpus is an adaptation.
- **The 27B / 7-of-12 / ~2,500-candidate / ~90% items should stay labelled as leads.** The first three are VentureBeat's secondhand report of a third party (chiefofautism, `Qwen3.8-27B`), with unretrievable primary posts; the 90% is Cammarata's Qwen3-32B label-share estimate and does not describe steering success. None is in Lyra's repo.
- **No exercise should claim "persona neurons" from the repo**: the repo's own log shows non-specific persona-name flips from random directions and ~19.6% of layer-22 neurons beating a word-count null.
- Compute is a non-issue for a one-neuron, few-prompt exercise (minutes, <= ~9 GB), but a replication of the full sweep or a faithful reproduction of the vLLM-steer path is not a laptop task.

## 11. Not established

- The two X posts (`_lyraaaa_` 2113833330429083662; `chiefofautism` 2115274388425900291) returned HTTP 402; no direct author text beyond the repo. The "evil" neuron doses, prompts and Lyra's own pass rates are unverified.
- Whether `Qwen3.8-27B` coordinates (`L34 N3838`, `L38 N17180`) were taken from the HF model or another checkpoint; the ~2,500-candidate search method beyond VentureBeat's summary; any rate behind "~90%" in the circulating message.
- The exact HF revision the author swept (only `fit_input_lens.py` pins; it equals current head).
- Neuronary's recipe listing (page JS-loaded, no API URL visible), so no benign recipe could be cross-checked there.
- The laptop 5090 throughput, vLLM-steer fork compatibility with Blackwell, and HF-path speed are estimates; fineweb-edu/ultrachat licences are unverified.
- Secondary sources used only to trace claims: VentureBeat (`venturebeat.com/ai/evil-qwen-says-humans-deserve-to-be-incinerated-inside-the-wild-world-of-llm-neuron-exploration/`), NEOTEO, AI Weekly. VentureBeat's page was read through a summarising fetch tool, not verbatim.
