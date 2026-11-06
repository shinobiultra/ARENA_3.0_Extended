# M2 dataset candidate metadata and exposure record — 2026-11-03

**Metadata finding, not accepted data custody, scientific design, or launch approval.** This companion appends to the frozen [empirical proposal](m2_empirical_protocol_20261103.md); it does not edit that proposal, helper or tests. The earlier proposal's no-example-access statement describes its earlier snapshot, not the later metadata activity recorded here.

The user approved **M2-R01 ceilings for later protocols**: one local GPU, at most3600 GPU-active seconds /7200 wall seconds including cleanup,20GiB VRAM,32GiB RAM,24 threads,4GiB writes, zero retries/clusters/pretrained weights. This is not approval of the82-fit design, data downloads/cache access, protected evaluation, or Q002 launch. Q001 still records the observed UI failure and needs repair plus a fresh human walkthrough. Other proposed sublimits must be resolved in the final protocol rather than inferred from the ceiling decision.

## Exact candidate source identities

Official metadata identifies repository `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`, at immutable repository commit **`b08601e04326c79dfdd32d625aee71d232d685c3`**. This is a candidate commit freeze, not a verified formal release tag or adopted scientific input. Its commit diff records the following Git LFS object hashes and exact compressed file sizes. These are **expected upstream shard-byte hashes**, not README hashes, Xet chunk identities or locally verified corpus hashes. [Official fixed-commit pointer metadata](https://huggingface.co/datasets/Salesforce/wikitext/commit/b08601e04326c79dfdd32d625aee71d232d685c3).

| Proposed role | Exact repository path | Expected bytes | Expected Git LFS SHA-256 |
|---|---|---:|---|
| TRAIN | `wikitext-2-raw-v1/train-00000-of-00001.parquet` | 6,357,543 | `e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7` |
| DEV | `wikitext-2-raw-v1/validation-00000-of-00001.parquet` | 657,209 | `204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c` |
| TEST | `wikitext-2-raw-v1/test-00000-of-00001.parquet` | 732,610 | `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91` |

Total expected shard transfer is **7,747,362 bytes**, an arithmetic sum of metadata sizes. It is not a measured transfer or allocation. Exact pinned resolve URLs are recorded, explicitly **unfollowed**, in the local candidate manifest. No shard/TEST download or content access is authorized by knowing its expected hash. The three sizes/hashes were obtained from the immutable commit's pointer diff; mutable `main` file pages supplied corroborating hashes. Fixed-revision blob and raw-pointer fetches were inaccessible, so no claim is made that those attempted endpoints were independently verified.

Primary observations occurred **2026-11-03 10:01:24–10:02:16 UTC**. Web API access failed; a single metadata-only Python urllib attempt failed DNS resolution. Subsequent endpoint failures and the boundary deviation are retained, not hidden. No credentials, installs, caches, model imports, device initialization, training, inference or cluster operations were used.

## License and provenance remain unresolved

The same fixed README declares `cc-by-sa-3.0` and `gfdl` in structured metadata, but its licensing prose points to **CC BY-SA4.0**. Preserve the conflict; do not silently choose one. The card attributes the associated2016 paper to Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher. [Official fixed-revision README](https://huggingface.co/datasets/Salesforce/wikitext/raw/b08601e04326c79dfdd32d625aee71d232d685c3/README.md).

M2-D01 therefore still needs an accountable rights/provenance disposition, exact attribution/notice handling, and an approved local-use/data-access scope. Citation metadata alone does not settle license obligations or authorize redistribution of corpus, derivatives or weights. No source/license file hash is invented where metadata did not supply it.

## Article families and disjointness are not established

The declared data feature is `text`; this metadata does not provide the article/redirect-family identifiers needed by the proposed grouping protocol. The dataset's split names and row counts cannot certify article-family independence. [Fixed README schema metadata](https://huggingface.co/datasets/Salesforce/wikitext/raw/b08601e04326c79dfdd32d625aee71d232d685c3/README.md).

Article reconstruction/version, title canonicalization, redirect/alias provenance, independent-unit identities, normalized-document/span hashes, actual role manifests and cohort availability remain **UNSET / not audited**. The existing source-only guard validates declared hashes/IDs; it does not derive those facts from corpus bytes. A later explicitly authorized custodian must verify actual bytes against the expected hashes and run the reviewed duplicate/grouping checks before fitting. Failures must stop the protocol and be retained; they cannot be repaired by silent row removal or re-splitting. No natural-language generalization or broad independence claim follows from these metadata findings.

## Boundary deviation and exposure

While seeking license metadata, a full raw README response unexpectedly returned its embedded public TRAIN/validation example snippets. That response exceeded the requested metadata-only boundary. The tool reference was `turn68view3`, during the observation interval above. Categories were: `wikitext-103-raw-v1` validation; `wikitext-103-v1` train; `wikitext-2-raw-v1` train; and `wikitext-2-v1` train. **No snippet text is copied into this note or receipts.** The parent was notified immediately and instructed the worker to stop all full-card/rawREADME/viewer reads and adaptive endpoint fallback attempts; those reads stopped.

No TEST example appeared in the inspected response, and no parquet/shard or TEST-file contents were obtained. However, this activity **cannot be described as sample-unexposed or wholly corpus-unread** because public TRAIN/DEV snippets were returned. Any later protocol/custody assessment must retain that exposure; no untouched-DEV claim is made here. Metadata readers should use field-bounded APIs or already reviewed excerpts rather than full cards that embed examples. This incident is not an accepted custody stage or evidence of disjointness.

## Receipts and remaining boundary

Local metadata receipts are under `.scratch/studiable-20261102/v1-quality/m2_protocol/metadata/`:

- `candidate_manifest.json`: exact candidate commit, expected three-shard identities, unfollowed pinned URLs, unresolved fields and authority limits.
- `primary-pointer-excerpts.txt`: small transcribed primary LFS pointer metadata only, with source URL and observation interval; no corpus examples.
- `access-failures.json`: failed metadata endpoints/DNS and stopped fallbacks.
- `boundary-deviation.json`: dated categories/URL/tool reference and the exposure correction, without example contents.
- `verification.json`: identity/size checks and verification that the three previously frozen reviewed files were not changed.

No scientific tests or native execution accompanied this metadata increment. The useful progress is a checkable upstream candidate identity and an explicit new rights/exposure problem. Data access, actual-byte verification, disjointness custody, numerical/design acceptance, Q001 usability acceptance and explicit Q002 launch remain pending. Approved later-protocol resource ceilings do not settle them.
