# Final integration delivery

Completed locally against the saved 131-record snapshot. No production database writes, scraping, prompt/validator changes, dependency installation, n8n edits, commits or pushes were performed.

## Actual processing results

Command from the service repository:

```bash
.venv/bin/python demo_job_intelligence.py
```

| Measure | Initial saved-data pass | Identical repeat |
|---|---:|---:|
| Input listings | 131 | 131 |
| Inserted source listings | 131 | 0 |
| Replayed observations | 0 | 131 |
| Source rows after pass | 131 | 131 |
| Canonical rows after pass | 131 | 131 |
| Observation receipts | 131 | 131 |
| Content/lifecycle history rows | 0 | 0 |
| New model calls | 0 | 0 |

The runner compared complete SQLite logical dumps before/after the identical repeat and asserted equality. The final database passed SQLite `integrity_check` and `foreign_key_check`. The CSV contains 131 canonical data rows. Availability remains unverified for all rows. No real cross-source cluster was accepted: 19 candidate pairs, 16 no-match and 3 review. The three review pairs remain separate. 48 canonical records have review indicators from matching or source extraction signals; this is not an error rate.

Actual deterministic field coverage (counts, not accuracy): company/title/posted date 131 each; employment type 126; country 99; city 58; remote policy 29; skills 28; seniority 4; salary bounds/currency/visa 0. Unknowns were preserved. Monetary cost and unavailable token counts are null. No historical LLM result was reused or presented as new inference.

The separate, prominently labeled synthetic database contains one source row and seven observation receipts: initial insertion, unchanged refresh, content edit, explicit closure, ordinary observation that leaves it closed, explicit reopening, and a stale observation that cannot overwrite newer state. It has exactly three history rows: one content edit and two lifecycle transitions, with previous/new values. Synthetic source content never enters the actual-data database or label template.

## Evaluation and verification

20 source records selected by alphabetical-source round-robin, review-first then UUID within source: Indeed 7, LinkedIn 7, Naukri 6. All human fields are blank. Evaluation output is `awaiting_human_labels`, zero labeled cells, null accuracy. This coverage-oriented sample is not representative for estimating population accuracy and does not meet the prescribed Hacker News labeling requirement.

**427 automated tests passed: all 415 existing tests plus 12 local integration tests.** The full run took 17.166 seconds. The harness disabled dotenv loading, supplied placeholders and blocked external HTTP. The sandbox run hit the existing FastAPI TestClient hang; the successful complete run used the same offline harness outside the sandbox. A FastAPI/Starlette TestClient deprecation warning remains; no dependency changes were made.

The 12 new tests cover initial insertion/source linkage, exact replay without extraction, last-seen refresh/content history, explicit lifecycle/absence, stale and tied timestamps, identity-collision rollback, rollback on canonicalization failure, ambiguous relationships/accepted match/conflict split, stable fingerprints/immutable identity, insufficient identity, CSV/label scoring, and optional hybrid composition using an injected fake transport. The fake transport test is not a live-model benchmark or genuine labeling result.

The runner, CSV and SQLite database are exercised against the complete saved dataset. The extracted bundle independently reproduced 131 inserts and 131 replays with an empty environment and HTTP/socket connection calls blocked; its 12 bundled integration tests also passed (11.869 seconds). Ruff check and format checks passed. Privacy scanning found no configured secrets, configured private hosts, non-loopback private IPv4 addresses or excluded poster/email metadata. All 430 pre-existing source-engine/test files and older verification artifacts matched the pre-task hash manifest. Actual standalone checks are in `submission-checks.json`; the detailed local privacy/preservation check is in `verification/final-integration/final-validation.json` in the original repository.

## Submission contents and limits

`job-intelligence-submission.zip` contains only allowlisted engine/integration code, the projected source data, concise README, minimal requirements, placeholder environment example, 12 integration tests, generated examples and this report/test summary. Original service configuration, personal files, `.env`, old batch artifacts, discovery ZIP, repository metadata and private infrastructure are excluded. Public source posting identities and job descriptions are retained. Poster profile fields, separate email metadata, scraper IDs and unrelated metadata were removed from the submission projection; extraction fields/evidence/warnings and matching decisions were verified equivalent.

Remaining blockers to full assignment compliance:

- Prescribed source connectors and Hacker News human labels are absent; current sources are LinkedIn/Indeed/Naukri.
- Genuine labels and measured accuracy are still required. Offline field coverage is not extraction accuracy.
- Known skills recall limitations and numerous unknown fields remain in the frozen engine.
- Production persistence and deployment remain unverified; local SQLite does not verify Nhost writes or production model identity.
- Optional live hybrid mode is implemented but was not run live in this sprint; no new model performance/cost claims are made.
- Real accepted multi-source matches and real connector closure events were not available. Those mechanics are tested synthetically and clearly labeled.
- Throughput at 5,000 postings/week is not benchmarked. Candidate blocking, incremental canonical updates and production concurrency are documented scaling work.

The delivery is a reproducible local integration demonstration, not a claim of full assignment compliance.
