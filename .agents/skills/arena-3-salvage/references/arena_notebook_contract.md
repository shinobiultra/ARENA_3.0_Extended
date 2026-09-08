# ARENA-Style Notebook Contract

This is the required structure for every rewritten extension notebook.

## 0. One-sentence claim

At the top of the notebook, include:

```text
By the end of this notebook, you will have shown that ...
```

The claim must name:

- the phenomenon;
- the model/task;
- the evidence;
- the control or baseline.

Bad:

```text
We explore logit lens and patchscopes.
```

Good:

```text
We will show that hidden states in a trained toy transformer increasingly encode the correct next-token answer across layers, because logit-lens and tuned-lens decoding recover held-out next-token predictions while random activations and text-only Patchscope prompts fail.
```

## 1. Cold open

Start with a concrete example before definitions.

Example for 7.1:

```text
Prompt: "The capital of France is"
Model final answer: " Paris"
Question: when does the residual stream start pointing toward " Paris"?
```

Then show a small table:

| layer | logit lens top token | correct token rank | probability |
|---:|---|---:|---:|
| 0 | ` the` | 542 | 0.001 |
| 1 | ` city` | 37 | 0.021 |
| 2 | ` Paris` | 1 | 0.41 |

The actual values can differ, but there must be a concrete object the student can understand.

## 2. Learning objectives

Use 4–7 bullets. They should be specific and operational.

Bad:

```text
Understand Patchscopes.
```

Good:

```text
- Implement a logit lens by multiplying residual stream states by the unembedding.
- Compare logit lens to a tuned affine lens on held-out prompts.
- Patch a source activation into a target prompt and measure whether it carries answer information.
- Distinguish activation-conditioned decoding from text-only prompt priors.
- Use random activations and counterfactual activations as controls.
```

## 3. Background, but only what is needed now

Do not paste a literature survey. Include only the concepts the next exercise needs. Put extra links in a final reading section.

Use the ARENA habit:

```text
Here is the mental model.
Here is the equation.
Here is how the tensor shapes map to the equation.
Now implement it.
```

## 4. Exercise block format

Each exercise must follow this pattern:

````markdown
### Exercise - implement `foo`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> You should spend 10–15 minutes on this exercise.
> ```

Explain what the function does in plain English.

```python
def foo(...):
    raise NotImplementedError()
```

```python
tests.test_foo_basic(foo)
tests.test_foo_edge_cases(foo)
```

<details>
<summary>Expected output</summary>

Show a real expected value, table, image, or plot.

</details>

<details>
<summary>Help - I don't understand why this matters</summary>

Explain the interpretation, not just the implementation.

</details>

<details>
<summary>Solution</summary>

Solution code and short explanation.

</details>
````

## 5. Student-implemented core

The following must be implemented in the notebook by the student:

- the main mathematical operation;
- the main measurement metric;
- the main intervention or decoding function;
- the control/baseline function;
- the final result aggregation.

The following may be hidden in helpers:

- plotting wrappers;
- dataset boilerplate;
- tokenizer boilerplate;
- expensive model-loading fallback;
- final release-report writer.

## 6. Visible tests

Every hard function needs tests directly below it. Tests should be named clearly and should print a success message.

Good:

```python
tests.test_logit_lens_matches_manual_matrix_multiplication(logit_lens)
tests.test_logit_lens_preserves_batch_and_position_axes(logit_lens)
tests.test_logit_lens_rejects_bad_unembedding_shape(logit_lens)
```

Bad:

```python
tests.test_notebook_contract(run_smoke_test)
```

Whole-notebook tests are allowed only after sub-function tests.

## 7. Signature result

Every notebook needs exactly one primary signature result.

Examples:

| Section | Signature result |
|---|---|
| Logit/Tuned Lens | layer-by-layer top-token table + tuned-lens improvement curve |
| Patchscopes | patched-vs-text-only answer table with counterfactual activations |
| ACDC | circuit graph + faithfulness/minimality/completeness curves |
| Refusal direction | layer sweep + addition/ablation steering curves |
| VLM visual-token flow | layer × token-type patching heatmap |
| CLIP geometry | retrieval heatmap + UMAP/probe/random-label failure |
| Diffusion | denoising trajectory + entropy/commitment plot |
| LoRA | SVD spectrum + behavior side-effect matrix |
| Shapley | exact-vs-approx attribution curves + agreement matrix |

The result must be generated or reproduced in the notebook. Loading a JSON report and printing a dict is not sufficient.

## 8. Play cells

Each notebook must include at least one cell titled:

```markdown
## Try it yourself
```

Examples:

```python
prompt = "The capital of Germany is"
layer = 3
show_logit_lens(prompt, layer)
```

```python
source_image = make_scene(color="red", shape="cube")
corrupt_image = make_scene(color="blue", shape="cube")
patch_visual_tokens(source_image, corrupt_image, region="object")
```

This is essential. ARENA notebooks feel alive because students can poke them.

## 9. Controls and baselines

Every notebook needs at least one control and one baseline.

Examples:

| Method | Baseline | Negative control |
|---|---|---|
| logit lens | tuned lens / final logits | random activations |
| Patchscope | text-only target prompt | counterfactual/random activation |
| VLM patching | text-only/image-only | background-region patch |
| UMAP geometry | linear probe/kNN | random labels, seed instability |
| ACDC | same-size random circuit | corrupted patch target |
| LoRA | full finetuning / random adapter | rank-matched random update |
| diffusion direction | random direction | opposite direction, classifier fails |

The notebook must explain what each control catches.

## 10. Interpretation and limitations

After the signature result, include:

```markdown
## Interpreting the result
```

with a learner-facing explanation. Then include:

```markdown
## Limitations
```

Limitations should be honest but not deflate the whole notebook. If the result is only a weak preflight, strengthen the notebook instead of writing a long disclaimer.

## 11. Reading links

End with 3–6 essential links, not 30.

For example, 7.1 should include:

- Logit Lens
- Tuned Lens
- Patchscopes
- TransformerLens docs / ARENA related section

## 12. Done criteria

The notebook is accepted only if it satisfies:

```text
core claim is clear
student implements core method
sub-function tests are visible
result is visually convincing
result has quantitative metric
baseline is weaker
negative control fails
student can play with inputs
real model or paper connection exists
limitations are honest
verification report exists but is not the lesson
```
