"""Isolated SQLite integration of the frozen extraction and matching engines.

No production configuration, imports from main, or implicit network access.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from pydantic import TypeAdapter

from .adapters import normalize_job
from .cache import canonical_json, current_versions
from .deterministic import apply_jd_rules
from .entity_resolution import MATCHING_POLICY_VERSION, resolve_jobs
from .extraction import extract_job, prepare_sources
from .schemas import JobExtractionInput, StructuredJobFields

FIELDS = tuple(StructuredJobFields.model_fields)
FINGERPRINT_POLICY = "local_source_content_v1"
VOLATILE_RAW = {
    "scrapedAt",
    "firstSeenAt",
    "lastSeenAt",
    "contentHash",
    "changeType",
    "applyCount",
    "applicants_count",
    "applicantsCount",
    "trackingId",
    "refId",
}
SCHEMA = """
CREATE TABLE IF NOT EXISTS canonical_jobs (
 id TEXT PRIMARY KEY, fields_json TEXT NOT NULL, evidence_json TEXT NOT NULL,
 source_urls_json TEXT NOT NULL, review_json TEXT NOT NULL,
 first_seen_at TEXT NOT NULL, last_seen_at TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('observed','closed','mixed')),
 availability TEXT NOT NULL CHECK(availability='unverified')
);
CREATE TABLE IF NOT EXISTS jobs (
 id TEXT PRIMARY KEY, source TEXT NOT NULL, external_job_id TEXT,
 canonical_job_url TEXT NOT NULL, original_source_json TEXT NOT NULL,
 source_json TEXT NOT NULL, fingerprint TEXT NOT NULL,
 extraction_json TEXT NOT NULL, extraction_key TEXT NOT NULL,
 first_seen_at TEXT NOT NULL, last_seen_at TEXT NOT NULL,
 last_observed_at TEXT NOT NULL, lifecycle TEXT NOT NULL
 CHECK(lifecycle IN ('observed','closed')),
 canonical_job_id TEXT REFERENCES canonical_jobs(id),
 UNIQUE(source,external_job_id), UNIQUE(source,canonical_job_url)
);
CREATE TABLE IF NOT EXISTS observations (
 event_id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
 observed_at TEXT NOT NULL, fingerprint TEXT NOT NULL,
 signal TEXT, label TEXT NOT NULL, outcome TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS job_change_history (
 id INTEGER PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
 event_id TEXT NOT NULL REFERENCES observations(event_id), observed_at TEXT NOT NULL,
 kind TEXT NOT NULL CHECK(kind IN ('content','lifecycle')),
 previous_json TEXT NOT NULL, new_json TEXT NOT NULL, label TEXT NOT NULL,
 UNIQUE(event_id,kind)
);
"""


def utc(value):
    parsed = TypeAdapter(datetime).validate_python(value)
    if parsed.tzinfo is None:
        raise ValueError("Observation timestamps must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


def content(row):
    """Business input only; source UUID, observation times and status are separate.

    Provider observation/counter fields are excluded recursively. All other
    adapter input metadata is material, even if currently unresolved.
    """

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k not in VOLATILE_RAW}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    return {
        k: clean(v)
        for k, v in JobExtractionInput.model_validate(row)
        .model_dump(mode="json")
        .items()
        if k not in {"id", "first_seen_at", "last_seen_at", "is_active"}
    }


def fingerprint(row):
    return hashlib.sha256(
        canonical_json([FINGERPRINT_POLICY, content(row)]).encode()
    ).hexdigest()


def connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript(SCHEMA)
    return db


def extraction(row, *, mode, ollama_call=None, model_name=None):
    if mode == "hybrid":
        if ollama_call is None or not model_name:
            raise ValueError(
                "Hybrid mode requires an explicit Ollama transport and model"
            )
        result = extract_job(row, ollama_call=ollama_call, model_name=model_name)
        return {"execution_mode": "live_hybrid", **result.model_dump(mode="json")}
    if mode != "deterministic":
        raise ValueError("Unsupported execution mode")
    result, flags, signals, _ = apply_jd_rules(
        normalize_job(row), prepare_sources(JobExtractionInput.model_validate(row))
    )
    return {
        **result.model_dump(mode="json"),
        "execution_mode": "offline_deterministic",
        "inference_status": "not_attempted",
        "review_flags": [v.model_dump(mode="json") for v in flags],
        "unverified_conflicts": [v.model_dump(mode="json") for v in signals],
        "metrics": {"model_calls": 0, "prompt_tokens": None, "output_tokens": None},
    }


def _history(db, job_id, event_id, observed_at, kind, before, after, label):
    db.execute(
        "INSERT INTO job_change_history(job_id,event_id,observed_at,kind,previous_json,new_json,label) VALUES(?,?,?,?,?,?,?)",
        (
            job_id,
            event_id,
            observed_at,
            kind,
            canonical_json(before),
            canonical_json(after),
            label,
        ),
    )


def process(
    db, observations, *, mode="deterministic", ollama_call=None, model_name=None
):
    """Apply one atomic batch, then rebuild local canonical records over ALL rows.

    The same timestamp with different content/signal is rejected, never selected
    by arrival order. A stale observation is receipted but cannot alter a job.
    Original source identity cannot silently move to a new UUID. Missing rows
    are untouched. Ordinary observations cannot reopen explicitly closed jobs.
    """
    counts = Counter()
    inference = []
    config_key = canonical_json([mode, model_name, current_versions()])
    with db:
        for observation in observations:
            row = observation["job"]
            job = JobExtractionInput.model_validate(row)
            job_id = str(job.id)
            observed_at = utc(observation["observed_at"])
            signal = observation.get("signal")
            if signal not in {None, "closed", "reopened"}:
                raise ValueError(
                    "Only explicit closed/reopened lifecycle signals are supported"
                )
            label = observation.get("label", "saved_actual_source_observation")
            source_fp = fingerprint(row)
            event_id = hashlib.sha256(
                canonical_json([job_id, observed_at, source_fp, signal]).encode()
            ).hexdigest()
            old = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if old and old["extraction_key"] != config_key:
                raise ValueError(
                    "Extraction configuration changed: use a separate --output directory"
                )
            if db.execute(
                "SELECT 1 FROM observations WHERE event_id=?", (event_id,)
            ).fetchone():
                counts["replayed"] += 1
                continue
            if old and (old["source"], old["external_job_id"]) != (
                job.source,
                job.external_job_id,
            ):
                raise ValueError("Source identity is immutable; cannot reuse UUID")
            if old and observed_at == old["last_observed_at"]:
                raise ValueError("Conflicting observations at the same timestamp")
            stale = old is not None and observed_at < old["last_observed_at"]
            changed = old is not None and old["fingerprint"] != source_fp
            lifecycle = (
                ("closed" if signal == "closed" else "observed")
                if signal
                else (old["lifecycle"] if old else "observed")
            )
            life_changed = old is not None and lifecycle != old["lifecycle"]
            outcome = (
                "stale"
                if stale
                else "inserted"
                if old is None
                else "changed"
                if changed
                else "refreshed"
            )
            if not stale:
                if old is None or changed or old["extraction_key"] != config_key:
                    result = extraction(
                        row, mode=mode, ollama_call=ollama_call, model_name=model_name
                    )
                    inference.append(
                        {
                            "job_id": job_id,
                            "metrics": result["metrics"],
                            "inference_status": result["inference_status"],
                        }
                    )
                else:
                    result = json.loads(old["extraction_json"])
                if old is None:
                    first_seen = (
                        min(utc(row["first_seen_at"]), observed_at)
                        if row.get("first_seen_at")
                        else observed_at
                    )
                    db.execute(
                        "INSERT INTO jobs(id,source,external_job_id,canonical_job_url,original_source_json,source_json,fingerprint,extraction_json,extraction_key,first_seen_at,last_seen_at,last_observed_at,lifecycle) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            job_id,
                            job.source,
                            job.external_job_id,
                            job.canonical_job_url,
                            canonical_json(row),
                            canonical_json(row),
                            source_fp,
                            canonical_json(result),
                            config_key,
                            first_seen,
                            observed_at,
                            observed_at,
                            lifecycle,
                        ),
                    )
                else:
                    db.execute(
                        "UPDATE jobs SET canonical_job_url=?,source_json=?,fingerprint=?,extraction_json=?,extraction_key=?,last_seen_at=?,last_observed_at=?,lifecycle=? WHERE id=?",
                        (
                            job.canonical_job_url,
                            canonical_json(row),
                            source_fp,
                            canonical_json(result),
                            config_key,
                            observed_at,
                            observed_at,
                            lifecycle,
                            job_id,
                        ),
                    )
            db.execute(
                "INSERT INTO observations VALUES(?,?,?,?,?,?,?)",
                (event_id, job_id, observed_at, source_fp, signal, label, outcome),
            )
            if not stale and changed:
                _history(
                    db,
                    job_id,
                    event_id,
                    observed_at,
                    "content",
                    content(json.loads(old["source_json"])),
                    content(row),
                    label,
                )
                counts["material_changes"] += 1
            if not stale and life_changed:
                _history(
                    db,
                    job_id,
                    event_id,
                    observed_at,
                    "lifecycle",
                    {"status": old["lifecycle"]},
                    {"status": lifecycle, "signal": signal},
                    label,
                )
                counts["lifecycle_changes"] += 1
            counts[outcome] += 1
        plan = _canonicalize(db)
        if db.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("Broken source-to-canonical linkage")
    totals = {
        table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in ("jobs", "canonical_jobs", "job_change_history", "observations")
    }
    return {
        "mode": mode,
        "counts": dict(counts),
        "totals": totals,
        "matching_policy": MATCHING_POLICY_VERSION,
        "candidate_decisions": dict(Counter(p.decision for p in plan.candidate_pairs)),
        "accepted_multi_source_groups": sum(
            c.review_status == "proposed" for c in plan.clusters
        ),
        "model_calls": sum(i["metrics"]["model_calls"] for i in inference),
        "new_extraction_metrics": inference,
        "cost": None,
        "availability": "unverified",
    }


def _canonicalize(db):
    stored = {r["id"]: dict(r) for r in db.execute("SELECT * FROM jobs ORDER BY id")}
    # Production links stay in original JSON; local UUIDs cannot authorize a merge.
    rows = [
        dict(json.loads(r["source_json"]), canonical_job_id=None)
        for r in stored.values()
    ]
    plan = resolve_jobs(rows)
    if plan.rejected_inputs:
        raise ValueError("Unsupported input in canonicalization")
    by_id = {item.job_id: item for item in plan.listings}
    review = {k: set() for k in stored}
    for pair in plan.candidate_pairs:
        if pair.decision == "review":
            for key in (pair.left_job_id, pair.right_job_id):
                review[key].add("ambiguous_cross_source_relationship")
    groups = []
    assigned = set()
    for cluster in plan.clusters:
        if cluster.review_status == "proposed":
            groups.append((cluster.member_job_ids, cluster))
            assigned.update(cluster.member_job_ids)
        else:
            for key in cluster.member_job_ids:
                review[key].update(cluster.reason_codes)
    groups.extend(([key], None) for key in stored if key not in assigned)
    desired = set()
    for members, cluster in groups:
        extractions = {
            key: json.loads(stored[key]["extraction_json"]) for key in members
        }
        if cluster:
            # Frozen consensus, never arbitrary LLM/source precedence for merges.
            fields = dict(
                cluster.supported_attributes,
                company=cluster.proposed_company,
                title=cluster.proposed_title,
            )
        else:
            fields = extractions[members[0]]["fields"]
        reasons = set().union(*(review[key] for key in members))
        for key, result in extractions.items():
            if (
                result.get("review_flags")
                or result.get("unverified_conflicts")
                or by_id[key].material_signals
            ):
                reasons.add("source_extraction_requires_review")
            if result["inference_status"] in {"failed", "parse_failed", "incomplete"}:
                reasons.add("hybrid_inference_incomplete")
        if not fields["company"] or not fields["title"]:
            db.executemany(
                "UPDATE jobs SET canonical_job_id=NULL WHERE id=?",
                ((key,) for key in members),
            )
            continue
        canonical_id = str(
            uuid5(NAMESPACE_URL, "sqlite-demo-v1:" + ",".join(sorted(members)))
        )
        desired.add(canonical_id)
        states = {stored[key]["lifecycle"] for key in members}
        status = next(iter(states)) if len(states) == 1 else "mixed"
        if status == "mixed":
            reasons.add("conflicting_source_lifecycle")
        evidence = {
            "matching_policy": MATCHING_POLICY_VERSION,
            "cluster_proposal": cluster.model_dump(mode="json") if cluster else None,
            "source_extractions": extractions,
        }
        urls = sorted(
            {json.loads(stored[key]["source_json"])["job_url"] for key in members}
            | {stored[key]["canonical_job_url"] for key in members}
        )
        values = (
            canonical_id,
            canonical_json(fields),
            canonical_json(evidence),
            canonical_json(urls),
            canonical_json({"required": bool(reasons), "reasons": sorted(reasons)}),
            min(stored[k]["first_seen_at"] for k in members),
            max(stored[k]["last_seen_at"] for k in members),
            status,
            "unverified",
        )
        db.execute(
            "INSERT INTO canonical_jobs VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET fields_json=excluded.fields_json,evidence_json=excluded.evidence_json,source_urls_json=excluded.source_urls_json,review_json=excluded.review_json,first_seen_at=excluded.first_seen_at,last_seen_at=excluded.last_seen_at,status=excluded.status",
            values,
        )
        db.executemany(
            "UPDATE jobs SET canonical_job_id=? WHERE id=?",
            ((canonical_id, key) for key in members),
        )
    # Derived local canonical view only; original source evidence is immutable.
    for row in db.execute("SELECT id FROM canonical_jobs").fetchall():
        if row[0] not in desired:
            db.execute("DELETE FROM canonical_jobs WHERE id=?", (row[0],))
    return plan


CSV_FIELDS = (
    "canonical_job_id",
    *FIELDS,
    "source_urls",
    "first_seen_at",
    "last_seen_at",
    "status",
    "vacancy_availability",
    "evidence",
    "review_required",
    "review_reasons",
)


def export_csv(db, path):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in db.execute("SELECT * FROM canonical_jobs ORDER BY id"):
            fields = json.loads(row["fields_json"])
            fields["skills"] = (
                canonical_json(fields["skills"])
                if fields["skills"] is not None
                else None
            )
            if fields["visa_sponsorship"] is not None:
                fields["visa_sponsorship"] = str(fields["visa_sponsorship"]).lower()
            review = json.loads(row["review_json"])
            writer.writerow(
                dict(
                    fields,
                    canonical_job_id=row["id"],
                    source_urls=row["source_urls_json"],
                    first_seen_at=row["first_seen_at"],
                    last_seen_at=row["last_seen_at"],
                    status=row["status"],
                    vacancy_availability=row["availability"],
                    evidence=row["evidence_json"],
                    review_required=str(review["required"]).lower(),
                    review_reasons=canonical_json(review["reasons"]),
                )
            )
