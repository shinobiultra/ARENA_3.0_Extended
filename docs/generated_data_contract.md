# Generated Data Contract

Generated datasets are preferred for GT-0 and GT-3 exercises because their
labels, counterfactuals, and controls can be audited. Prompt artifacts should
be JSONL with one record per line.

Required prompt record fields:

- `id`
- `task`
- `prompt`
- `label`
- `safe_to_display`
- `contains_sensitive_content`
- `target_token`
- `counterfactual_id`
- `metadata.template`
- `metadata.split`
- `metadata.source`
- `metadata.seed`

Safety-sensitive records must store `prompt: "[REDACTED]"` and a 64-character
hex `prompt_hash`. Do not commit unsafe procedural prompt text to notebooks,
expected outputs, screenshots, or prompt artifacts.

Every generated dataset should also have a manifest with:

- generator script or template source
- seed
- schema path
- train, validation, and test split policy
- OOD split definition
- counterfactual mapping rule
- label function
- example preview
- required controls: label permutation, template permutation,
  spurious-correlation split, counterfactual split, and random-input split

The starter fixture is
[refusal_proxy_prompts_v1](../data/generated/refusal_proxy_prompts_v1/README.md).
Notebook code should validate prompt sets with
`arena_ext.data_contracts.validate_prompt_records` before using them as
evidence.
