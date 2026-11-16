"""Human labels are intentionally absent until supplied by a real reviewer."""

from __future__ import annotations

import csv
import json
from collections import Counter, deque
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .cache import canonical_json
from .local_demo import FIELDS

UNKNOWN = "__UNKNOWN__"
SELECTION = "Source-balanced round-robin (alphabetical source), review-first then source UUID within each source; first 20. Not a random population estimate."


def make_template(db, path):
    groups = {}
    for row in db.execute("SELECT * FROM jobs ORDER BY source,id"):
        value = dict(row)
        result = json.loads(row["extraction_json"])
        value["prediction"] = result["fields"]
        value["source_review"] = bool(
            result.get("review_flags") or result.get("unverified_conflicts")
        )
        groups.setdefault(row["source"], []).append(value)
    queues = {
        key: deque(sorted(rows, key=lambda row: (not row["source_review"], row["id"])))
        for key, rows in groups.items()
    }
    selected = []
    while len(selected) < 20 and any(queues.values()):
        for key in sorted(queues):
            if queues[key] and len(selected) < 20:
                selected.append(queues[key].popleft())
    columns = [
        "source_job_id",
        "canonical_job_id",
        "source",
        "source_url",
        "job_title",
        "job_description",
        "source_snapshot_json",
        "selection_method",
        "annotator",
        "label_notes",
    ]
    columns += [prefix + name for name in FIELDS for prefix in ("predicted_", "human_")]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in selected:
            source = json.loads(row["source_json"])
            output = {
                "source_job_id": row["id"],
                "canonical_job_id": row["canonical_job_id"],
                "source": row["source"],
                "source_url": source["job_url"],
                "job_title": source["job_title"],
                "job_description": source.get("job_description"),
                "source_snapshot_json": row["source_json"],
                "selection_method": SELECTION,
            }
            output.update(
                {
                    "predicted_" + name: _cell(value)
                    for name, value in row["prediction"].items()
                }
            )
            writer.writerow(output)
    return {
        "records": len(selected),
        "sources": dict(Counter(row["source"] for row in selected)),
        "selection_method": SELECTION,
        "human_labels": 0,
    }


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, (list, bool)):
        return canonical_json(value)
    return str(value)


def _value(field, value):
    value = value.strip()
    if value in {"", UNKNOWN}:
        return None
    if field in {"salary_min", "salary_max"}:
        try:
            result = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError(f"Invalid number for {field}") from exc
        if not result.is_finite() or result < 0:
            raise ValueError(f"Invalid salary for {field}")
        return result
    if field == "visa_sponsorship":
        if value.casefold() not in {"true", "false"}:
            raise ValueError("visa_sponsorship must be true/false or __UNKNOWN__")
        return value.casefold() == "true"
    if field == "skills":
        result = json.loads(value)
        if (
            not isinstance(result, list)
            or not result
            or any(not isinstance(v, str) or not v.strip() for v in result)
        ):
            raise ValueError(
                "skills must be a nonempty JSON string list or __UNKNOWN__"
            )
        return frozenset(v.strip().casefold() for v in result)
    return " ".join(value.casefold().split())


def evaluate(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    seen = set()
    report = {}
    for row in rows:
        if not row.get("source_job_id") or row["source_job_id"] in seen:
            raise ValueError("Missing or repeated source job ID")
        seen.add(row["source_job_id"])
        if (
            any((row.get("human_" + f) or "").strip() for f in FIELDS)
            and not (row.get("annotator") or "").strip()
        ):
            raise ValueError("Identify the human annotator before scoring labels")
    for field in FIELDS:
        labeled = known = correct = predicted_on_known = correct_known = (
            false_positive
        ) = 0
        for row in rows:
            human_text = (row.get("human_" + field) or "").strip()
            if not human_text:
                continue  # Blank is unlabelled, never a negative/unknown label.
            human = _value(field, human_text)
            prediction = _value(field, row.get("predicted_" + field) or "")
            labeled += 1
            correct += human == prediction
            if human is not None:
                known += 1
                predicted_on_known += prediction is not None
                correct_known += human == prediction
            elif prediction is not None:
                false_positive += 1
        report[field] = {
            "labeled": labeled,
            "unlabeled": len(rows) - labeled,
            "known_human_values": known,
            "confirmed_unknown": labeled - known,
            "correct": correct,
            "false_positive_on_unknown": false_positive,
            "accuracy_all_labeled": correct / labeled if labeled else None,
            "accuracy_known": correct_known / known if known else None,
            "prediction_coverage_known": predicted_on_known / known if known else None,
            "label_coverage": labeled / len(rows) if rows else None,
        }
    return {
        "records": len(rows),
        "labeled_cells": sum(r["labeled"] for r in report.values()),
        "status": "human_labels_present"
        if any(r["labeled"] for r in report.values())
        else "awaiting_human_labels",
        "fields": report,
        "notice": "Accuracy is conditional on supplied genuine labels; source-balanced sample is not population accuracy. This is not the prescribed Hacker News evaluation.",
    }
