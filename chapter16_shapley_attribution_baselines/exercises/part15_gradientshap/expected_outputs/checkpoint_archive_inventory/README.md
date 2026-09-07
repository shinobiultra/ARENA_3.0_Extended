# GradientSHAP 16.15 checkpoint archive inventory

Status: **PASS** — 9 unique evidence archives.

| Label | Included | External | Source bytes | Archive SHA-256 |
| --- | ---: | ---: | ---: | --- |
| `v1_prefit_failure` | 28 | 0 | 1339466 | `b5a5d975ea2a30a5c54eeedd68e3d31203b07e86ea4c115302d32302324d2ef0` |
| `v2_own_answer_visual_failure` | 153 | 3 | 3581915110 | `bfc1ee381bda375b5608926b2d54001bbdc0839de175bd5b2c367e44237ad9c6` |
| `v2_fresh_reproduction` | 53 | 3 | 3550015597 | `6868ea9f27081b80fdb3667345c693ca2cdc5bf2f5ddf505785ce690493ca8ef` |
| `v3_own_answer_browser_visual_pass` | 139 | 3 | 3579774743 | `c42991fe5c3f4611d07df22e4d345c530574cedfe6e3004afb811829d9652885` |
| `v3_fresh_reproduction` | 53 | 3 | 3550016836 | `52ee16ce70cdb05fbbc6502bda79f354576babb1d4a94a8d42538bcd8d7a6efa` |
| `completed_protocol_review` | 1 | 0 | 14028 | `4924e77446e1e32bbb2027ae3f10e36143577b02a461e6e28392b4112628d1a3` |
| `completed_implementation_review` | 1 | 0 | 10234 | `3845dd0a959549c02cd995988607b771d2353b0f439261501b507822e88b6a17` |
| `completed_frontend_reviews` | 2 | 0 | 13454 | `ac90daabe3b624dff3ae5555ca222d205d7f516ff41ed6b28a25a6904a3393a1` |
| `completed_publication_review` | 7 | 0 | 52288 | `c038a05a20cdf11f440b9bab5100bf10d8a56ffb164fe85a8c021623e4ef01b9` |

Every inline member was reread from its tar archive and rehashed. Every external file over 128 MiB remains at its original cache path and was rehashed after archive creation; large tensor records also bind their recursive manifests. Historical v1 FAIL, v2 visual FAIL, and v3 PASS statuses are retained verbatim in the embedded manifests.

Manifest inventory SHA-256: `cf05eb989336b417e03c5f2ac5a27f67078671a1ea67852b0c5ffa9773d5ad1b`
