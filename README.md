# Job Intelligence — local integration submission

This is a working, offline end-to-end demonstration built around the existing, frozen extraction and entity-resolution engines. SQLite is an isolated submission environment. It does not replace or verify the Nhost production persistence layer. This delivery does **not** claim full compliance with the original assignment.

## Run

Requires Python 3.10+ and Pydantic 2.13.5. The existing service virtual environment already provides the dependencies; no dependencies were installed for this delivery. The minimal requirements file is for a reviewer preparing a separate environment.

From the original service repository, run:

```bash
.venv/bin/python demo_job_intelligence.py
```

From an extracted submission archive, with dependencies available, run:

```bash
python demo_job_intelligence.py
```

Each command reads the bundled data, processes it, repeats the identical input, asserts that persisted state is identical, writes `demo-output/`, and runs a separate synthetic lifecycle demonstration. No network, credentials or `.env` are needed. Use `--output another-directory` for a fresh database; existing databases are reused safely and are never silently deleted. A later command against the same directory reports replays rather than fresh inserts. Completed human labels are never overwritten; a rerun writes a separate `.generated.csv` template.

Included actual outputs are in [`example-output/`](example-output/): `jobs.sqlite3`, `canonical_jobs.csv`, `human_labels_template.csv`, `summary.json`, `evaluation_pending.json`, and the separate `synthetic_lifecycle.sqlite3`/JSON. Initial saved-data processing inserted **131 listings / 131 canonical singletons / 131 observation receipts / zero history rows**. The identical repeat replayed **131** observations, with identical persisted state and **zero model calls**. No cross-source match was accepted: 19 candidate pairs were evaluated, 3 required review and 16 did not match. Unmatched is not proof of uniqueness. The synthetic sequence produced one content history row and two lifecycle history rows.

## Architecture and source data

`demo_job_intelligence.py` orchestrates saved input → frozen adapters and deterministic JD rules → incremental source storage → frozen cross-source resolution → local canonical records → SQLite/CSV/evaluation artifacts. `job_intelligence/local_demo.py` contains the thin SQLite integration; `demo_evaluation.py` contains labeling/scoring. Existing prompt, validator, extraction, caching and matching implementations were not tuned.

`data/jobs.json` contains a projection of the saved **131 actual jobs: LinkedIn 91, Indeed 12, Naukri 28**. Source UUIDs, external IDs, posting URLs, job content and extraction/matching evidence are retained. The projection removes job-poster profiles/photos/names, separate email metadata, scraper identifiers, counters and unrelated metadata. Its provenance records the original snapshot hash and excluded paths. Automated checks verified identical deterministic fields, evidence, warnings and matching decisions before and after projection. Original saved snapshots were not modified. The snapshot was fetched with separate read-only count/data requests, not a database-wide consistent snapshot.

`observed_at=2026-11-16T06:34:00Z` represents the saved snapshot observation, not a newly verified vacancy or provider posting date. Original provider first-seen values are preserved separately. The demo does not scrape or refresh job availability. Company and title must both be supported before a canonical singleton is created; unidentified records remain source rows with null canonical linkage.

## Extraction and matching

The default is **offline deterministic** extraction: source adapters plus existing explicit JD rules. Output includes unknowns, warnings, evidence and review signals. It is not fresh LLM inference and does not replay historical LLM results. No accuracy, token counts or monetary costs are invented. Missing salary, visa, skills and other facts stay unknown.

Optional live mode uses the frozen hybrid extractor and exported Ollama settings, without importing production configuration:

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=your-configured-model
python demo_job_intelligence.py --mode hybrid --output hybrid-output
```

`httpx==0.28.1` and a reachable configured Ollama model are needed for live mode. This command may make one bounded inference request per eligible changed/new listing. It can take substantial time over 131 records. It was **not run live in this sprint**. Actual request diagnostics and provider metrics, where returned, are stored; failed or incomplete inference is flagged. Use a separate output directory when changing mode/model/engine versions. Hybrid repeat runs reuse stored results and never report them as new inference. No historical LLM output is promoted to current inference. For shared canonicals, the frozen deterministic consensus remains authoritative; additional hybrid source fields stay in per-source evidence rather than being arbitrarily combined.

Matching is conservative: provider IDs are source-scoped; similar descriptions are supporting evidence. Shared canonical records require an accepted, consistency-checked cluster from the existing engine. Supported common employer application URLs or employer-scoped requisition IDs can support a match. Ambiguous candidates and unsafe clusters stay separate and are flagged. Canonical IDs are deterministic, local UUIDs based on sorted member source UUIDs; they are not production promotions. Source rows retain original and current snapshots, complete extraction evidence and a foreign-key canonical relationship. Canonical rows are a derived local view and may split/merge when new evidence changes the accepted groups.

## Incremental ingestion and history

The immutable identity is the original UUID plus `(source, external_job_id)`. SQLite enforces source-scoped external-ID and canonical-posting-URL uniqueness, foreign keys, and atomic transactions spanning ingestion, history and canonical rebuilding. A conflicting identity rolls back the batch instead of silently merging source rows. Identical input UUID reuse cannot replace the original source identity.

The versioned `local_source_content_v1` SHA-256 fingerprint uses canonical sorted JSON over validated business input. It excludes observation timestamps, `is_active` and documented provider scrape/counter fields (`VOLATILE_RAW` in `local_demo.py`). All other retained adapter metadata is material. Original source JSON remains available separately. Event identity combines job UUID, UTC observation time, content fingerprint and explicit lifecycle signal. Replay does nothing; unchanged later observations refresh last seen; changed observations store previous/new business content in `job_change_history`. Timestamps require timezones and are normalized to UTC. Older observations are recorded as stale and cannot change current fields, lifecycle or last seen. Conflicting observations at an equal timestamp are rejected, not arbitrarily ordered.

Only explicit `closed` / `reopened` observation signals change lifecycle. Neither absence from search, `is_active=false`, nor a later ordinary observation closes/reopens a listing. `status=observed` means an observed source lifecycle, **not verified vacancy availability**. CSV includes `vacancy_availability=unverified`; disagreements among source lifecycle states are `mixed` and flagged. Synthetic observations use future test timestamps, a prominent `SYNTHETIC_MODIFIED_OBSERVATION_NOT_REAL_SOURCE_EVENT` label, and a separate SQLite database. The original snapshot and actual-data database are not modified by this demonstration.

CSV contains all 13 structured fields plus canonical ID, source URLs, first/last seen, lifecycle status, availability, JSON evidence and review indicators. Blank cells mean unknown; booleans are true/false; skills and evidence are JSON. Source URLs remain public posting identities. Evidence preserves original values, source paths, offsets where available, method and warnings. The SQLite tables are `jobs`, `canonical_jobs`, `job_change_history`, plus an observation-receipt table for replay protection.

## Human evaluation

The template selects **20 source records** by source-balanced round-robin in alphabetical source order, review-first then source UUID within each source: **Indeed 7, LinkedIn 7, Naukri 6**. This is a transparent coverage-oriented sample, not a random population accuracy estimate. It includes source descriptions and a source snapshot for inspection. Predicted and human columns are separate. All human labels and annotator fields are initially blank.

Copy the template, inspect source evidence, enter genuine labels in `human_*`, and identify the human `annotator`. Blank means **not labeled**; `__UNKNOWN__` means the reviewer confirmed the source does not establish that field. Use normalized enum values, true/false, numeric annual base salary, ISO dates and a JSON array for skills. Do not copy predictions as gold labels. Run:

```bash
python evaluate_job_intelligence.py my_human_labels.csv --output evaluation.json
```

The evaluator reports each field's labeled/unknown counts, label coverage, accuracy over labeled cells, accuracy over known human values, prediction coverage over known human values, and false positives on confirmed unknowns. Missing predictions count as wrong for known labels; confirmed unknown plus missing prediction agrees. Text comparison ignores case/extra whitespace, numeric amounts use exact decimals, skills use exact case-insensitive set equality. No partial skill credit or spelling aliases are invented. Unlabeled cells do not enter accuracy denominators. The supplied blank template reports `awaiting_human_labels`, zero labeled cells and null accuracy. The script trusts that supplied labels are genuine and cannot authenticate a human reviewer. The sample **does not fulfill the prescribed Hacker News human-labeling requirement**.

## Verification

```bash
python -m unittest test_job_intelligence_local_demo
```

The bundle includes 12 essential local integration checks covering initial ingestion/linkage, replay, material changes, explicit lifecycle, stale/tied timestamps, fingerprint stability, atomic rollback, safe merges/ambiguity/splits, insufficient identity, CSV, evaluation and injected hybrid composition. The original service suite and these tests are verified together in the accompanying test report; the production-service tests require the full original repository and are not copied into this minimal bundle. No live Nhost writes or n8n changes were performed. See `verification/delivery_report.md` for actual results and limitations.

## Limitations and remaining assignment gaps

- Actual sources are LinkedIn/Indeed/Naukri instead of the originally prescribed sources. No new scraping connectors were implemented.
- Prescribed Hacker News records and genuine human labels are missing. Accuracy remains unmeasured; this is not a completed assignment evaluation.
- Known skills recall limitations remain in the frozen engine. Offline deterministic extraction leaves substantial fields unknown. No prompt or validator remediation was attempted.
- Production persistence remains unverified. SQLite success does not establish Nhost schema, writes, deployment or runtime model identity correctness. No live hybrid benchmark was performed in this sprint.
- Zero accepted real cross-source matches means the real snapshot demonstrates conservative singletons, not a successful real multi-source merge. Merge and conflicting-source behavior are demonstrated only with clearly synthetic tests.
- Closure/reopening are a supported explicit ingestion contract, demonstrated synthetically; existing connectors have not been updated to supply verified closure signals. Job availability is unverified.
- SQLite canonical rebuilding currently loads the local dataset and uses the existing O(n²) candidate search. It is appropriate for this small demonstration, not a production throughput benchmark. No distributed concurrency/retry/queue infrastructure or deployment was added.

## Scaling to 5,000 postings weekly

Keep source identities, material fingerprints and append-only history. Use PostgreSQL/Nhost transactions, indexes and conditional updates for concurrent observations; use employer/title/strong-identity candidate blocks instead of a full all-pairs scan. Incrementally rebuild only affected canonical groups. Queue only new/changed eligible records for bounded hybrid extraction, preserve version-aware evidence/cache gates, and measure real inference latency/failures and field-level human accuracy before setting capacity/cost targets. Export in pages and retain explicit closure events independently of search coverage. These are next steps, not implemented performance claims.

## Packaging

The private-submission ZIP is built from an explicit file allowlist, not the service directory. It excludes `.env`, Nhost credentials, private server addresses, unrelated service artifacts, candidate resumes, repository history, caches and the discovery-service ZIP. `.env.example` contains only local placeholders; the runner does not load it automatically. `MANIFEST.json` records file hashes. The bundle includes public job data and should remain a private evaluation submission.
