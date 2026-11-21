# Benchmarks and external assets

No benchmark questions, gold answers, private tests, reference solutions, model
weights, or historical execution logs are committed to this repository.
Download assets directly from their upstream projects under their own terms.

| Asset | Pinned source and scope |
|---|---|
| ALMA | `zksha/alma`, commit `0a6ecb1f8ffbc704e15ac2c98a6973ea60caf70b` |
| ALFWorld text games | `alfworld-tw-0.4.2.zip`, SHA256 checked by setup |
| ALFWorld metadata | Official 0.2.2 release `json_2.1.1_json.zip`; matched by exact game path |
| CrossEp-Emb mapping | `DSAIL-Memory/EvoMemBench`, commit `3726cf1894d2b16698b9bc4a6186db91ad93849d` |
| MATH | `EleutherAI/hendrycks_math`, revision `21a5633873b6a120296cce3e2df9d5550074f4a3` |
| APPS | `codeparrot/apps`, revision `21e74ddf8de1a21436da12e3e653065c5213e9d1` |
| APPS grading reference | `hendrycks/apps`, commit `362aedc3c71cd7d9bd2fc96a6c80e11dbc38c7a5` |

MATH eligibility and scoring are frozen in
`benchmarks/math_apps/math_pool_registration.json`: text-only algebra and number
theory, one scalar integer reference, all difficulty levels, and question
deduplication before allocation. Report this subset rather than a full MATH score.

APPS preparation indexes the full pinned train/test files. Private cases remain
in the host-side archive; Actor-facing adapters load public questions and starter
code only. Source identities and normalized text remove duplicate items before
the online sequence. The author's original metadata archive is checked separately.

For ALFWorld, different game files are task instances. Same-family transfer
differs from cross-family transfer. The CrossEp-Emb published `test` mapping refers
to games under the ALFWorld asset directory `valid_train`; retain both identities.
This online adaptation protocol must not be presented as the original leaderboard
protocol or a fresh blind test when a partition has been used in development.

TextWorld is retained in the ALMA campaign adapter, but requires the upstream
Glulx-compatible 1.6.2 fork and separately acquired games. It is not installed by
the ALFWorld extra. See the package campaign documentation and upstream adapter;
this release has not revalidated a clean TextWorld installation.
