"""Essential integration checks, all local and synthetic."""

import copy
import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from job_intelligence.demo_evaluation import evaluate, make_template
from job_intelligence.local_demo import connect, export_csv, fingerprint, process


def source(number=1, platform="linkedin", *, shared_url=None):
    row = {
        "id": str(UUID(int=number)),
        "source": platform,
        "external_job_id": f"provider-{number}",
        "job_url": f"https://{platform}.example.test/jobs/{number}",
        "canonical_job_url": f"https://{platform}.example.test/jobs/{number}",
        "job_title": "Senior Data Engineer",
        "company": "Example Labs",
        "location": "Bengaluru, Karnataka, India",
        "employment_type": "Full Time",
        "job_description": "Build reliable data pipelines and maintain scalable production systems.",
        "raw_source_data": {},
    }
    if shared_url:
        row["raw_source_data"] = {"normalized_metadata": {"apply_url": shared_url}}
    return row


def event(row, day=1, signal=None):
    return {
        "job": row,
        "observed_at": f"2030-01-{day:02d}T00:00:00Z",
        "signal": signal,
        "label": "SYNTHETIC_TEST",
    }


class LocalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.db = connect(self.root / "jobs.sqlite3")

    def tearDown(self):
        self.db.close()
        self.directory.cleanup()

    def test_first_ingestion_and_source_linkage(self):
        row = source()
        result = process(self.db, [event(row)])
        self.assertEqual(
            result["totals"],
            {
                "jobs": 1,
                "canonical_jobs": 1,
                "job_change_history": 0,
                "observations": 1,
            },
        )
        linked = self.db.execute(
            "SELECT j.*,c.availability FROM jobs j JOIN canonical_jobs c ON c.id=j.canonical_job_id"
        ).fetchone()
        self.assertEqual(json.loads(linked["original_source_json"]), row)
        self.assertEqual(linked["id"], row["id"])
        self.assertEqual(linked["availability"], "unverified")
        self.assertFalse(self.db.execute("PRAGMA foreign_key_check").fetchall())

    def test_second_ingestion_is_identical_without_extraction(self):
        process(self.db, [event(source())])
        before = list(self.db.iterdump())
        with patch(
            "job_intelligence.local_demo.extraction",
            side_effect=AssertionError("No new extraction on replay"),
        ):
            result = process(self.db, [event(source())])
        self.assertEqual(result["counts"], {"replayed": 1})
        self.assertEqual(before, list(self.db.iterdump()))

    def test_material_change_history_and_unchanged_refresh(self):
        original = source()
        process(self.db, [event(original)])
        process(self.db, [event(original, 2)])
        self.assertIn(
            "2030-01-02", self.db.execute("SELECT last_seen_at FROM jobs").fetchone()[0]
        )
        modified = dict(original, job_description="Changed business responsibilities")
        result = process(self.db, [event(modified, 3)])
        history = self.db.execute("SELECT * FROM job_change_history").fetchone()
        self.assertEqual(result["counts"]["material_changes"], 1)
        self.assertEqual(
            json.loads(history["previous_json"])["job_description"],
            original["job_description"],
        )
        self.assertEqual(
            json.loads(history["new_json"])["job_description"],
            modified["job_description"],
        )
        self.assertEqual(
            json.loads(
                self.db.execute("SELECT original_source_json FROM jobs").fetchone()[0]
            ),
            original,
        )

    def test_explicit_closure_reopening_and_absence(self):
        row = source()
        process(self.db, [event(row)])
        process(self.db, [event(row, 2, "closed")])
        process(self.db, [])  # Search absence has no lifecycle effect.
        process(self.db, [event(row, 3)])  # Ordinary observation cannot reopen.
        self.assertEqual(
            self.db.execute("SELECT lifecycle FROM jobs").fetchone()[0], "closed"
        )
        process(self.db, [event(row, 4, "reopened")])
        self.assertEqual(
            self.db.execute("SELECT status FROM canonical_jobs").fetchone()[0],
            "observed",
        )
        self.assertEqual(
            self.db.execute(
                "SELECT count(*) FROM job_change_history WHERE kind='lifecycle'"
            ).fetchone()[0],
            2,
        )

    def test_stale_observation_cannot_overwrite_and_time_ties_reject(self):
        row = source()
        process(self.db, [event(row, 3)])
        changed = dict(row, job_description="Stale description")
        result = process(self.db, [event(changed, 1, "closed")])
        self.assertEqual(result["counts"], {"stale": 1})
        self.assertEqual(
            json.loads(self.db.execute("SELECT source_json FROM jobs").fetchone()[0]),
            row,
        )
        with self.assertRaisesRegex(ValueError, "same timestamp"):
            process(self.db, [event(changed, 3)])
        self.assertEqual(
            self.db.execute("SELECT count(*) FROM job_change_history").fetchone()[0], 0
        )

    def test_atomic_rollback_on_identity_collision(self):
        duplicate = source(2)
        duplicate["external_job_id"] = source()["external_job_id"]
        with self.assertRaises(sqlite3.IntegrityError):
            process(self.db, [event(source()), event(duplicate)])
        for table in ("jobs", "canonical_jobs", "observations", "job_change_history"):
            self.assertEqual(
                self.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 0
            )

    def test_atomic_rollback_on_canonicalization_failure(self):
        process(self.db, [event(source())])
        before = list(self.db.iterdump())
        with (
            patch(
                "job_intelligence.local_demo.resolve_jobs",
                side_effect=RuntimeError("synthetic failure"),
            ),
            self.assertRaises(RuntimeError),
        ):
            process(self.db, [event(dict(source(), job_description="Changed"), 2)])
        self.assertEqual(before, list(self.db.iterdump()))

    def test_safe_matching_and_ambiguous_singletons(self):
        left, right = source(1), source(2, "indeed")
        result = process(self.db, [event(left), event(right)])
        self.assertEqual(result["totals"]["canonical_jobs"], 2)
        self.assertTrue(
            all(
                json.loads(r[0])["required"]
                for r in self.db.execute("SELECT review_json FROM canonical_jobs")
            )
        )
        # A supported common employer URL allows a consistency-checked match.
        url = "https://careers.example.test/jobs/REQ-100"
        left = source(1, shared_url=url)
        right = source(2, "indeed", shared_url=url)
        result = process(self.db, [event(left, 2), event(right, 2)])
        self.assertEqual(result["accepted_multi_source_groups"], 1)
        self.assertEqual(result["totals"]["canonical_jobs"], 1)
        self.assertEqual(
            self.db.execute(
                "SELECT count(DISTINCT canonical_job_id) FROM jobs"
            ).fetchone()[0],
            1,
        )
        # Conflict splits the derived view; original listings and evidence survive.
        right["location"] = "London, England, United Kingdom"
        result = process(self.db, [event(right, 3)])
        self.assertEqual(result["totals"]["canonical_jobs"], 2)
        self.assertEqual(result["totals"]["jobs"], 2)

    def test_fingerprint_stability_and_original_identity_protection(self):
        row = source()
        modified = copy.deepcopy(row)
        modified.update(
            first_seen_at="2029-01-01T00:00:00Z",
            last_seen_at="2029-01-02T00:00:00Z",
            is_active=False,
        )
        modified["raw_source_data"]["scrapedAt"] = "2030-01-01"
        self.assertEqual(fingerprint(row), fingerprint(modified))
        process(self.db, [event(row)])
        modified["external_job_id"] = "another-identity-200"
        with self.assertRaisesRegex(ValueError, "immutable"):
            process(self.db, [event(modified, 2)])

    def test_unknown_company_does_not_create_canonical(self):
        result = process(self.db, [event(dict(source(), company=None))])
        self.assertEqual(result["totals"]["jobs"], 1)
        self.assertEqual(result["totals"]["canonical_jobs"], 0)

    def test_csv_and_human_evaluation(self):
        process(self.db, [event(source())])
        path = self.root / "jobs.csv"
        export_csv(self.db, path)
        with path.open(newline="") as handle:
            row = next(iter(csv.DictReader(handle)))
        self.assertEqual(row["salary_min"], "")
        self.assertEqual(row["vacancy_availability"], "unverified")
        self.assertEqual(row["status"], "observed")
        self.assertTrue(json.loads(row["evidence"])["source_extractions"])
        labels = self.root / "labels.csv"
        make_template(self.db, labels)
        pending = evaluate(labels)
        self.assertEqual(pending["labeled_cells"], 0)
        self.assertIsNone(pending["fields"]["title"]["accuracy_all_labeled"])
        with labels.open(newline="") as handle:
            reader = csv.DictReader(handle)
            columns, rows = reader.fieldnames, list(reader)
        # Synthetic evaluator unit check, never included as genuine human labels.
        rows[0].update(
            annotator="SYNTHETIC_TEST",
            human_title="Senior Data Engineer",
            human_salary_min="__UNKNOWN__",
            human_skills='["Python"]',
        )
        with labels.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        result = evaluate(labels)
        self.assertEqual(result["fields"]["title"]["accuracy_all_labeled"], 1)
        self.assertEqual(result["fields"]["salary_min"]["confirmed_unknown"], 1)
        self.assertEqual(result["fields"]["skills"]["prediction_coverage_known"], 0)

    def test_live_hybrid_composition_with_injected_transport(self):
        # Real transport is optional. This checks integration without inference.
        from job_intelligence.local_demo import extraction

        def transport(payload):
            return {"response": "{}", "done": True}

        result = extraction(
            source(), mode="hybrid", ollama_call=transport, model_name="mock-model"
        )
        self.assertEqual(result["execution_mode"], "live_hybrid")
        self.assertIn("model_calls", result["metrics"])


if __name__ == "__main__":
    unittest.main()
