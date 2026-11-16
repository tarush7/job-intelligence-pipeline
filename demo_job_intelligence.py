"""One-command local demo; deterministic by default, never imports production."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

from job_intelligence.demo_evaluation import evaluate, make_template
from job_intelligence.local_demo import connect, export_csv, process


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def dataset_path():
    base = Path(__file__).resolve().parent
    direct = base / "data/jobs.json"
    return direct if direct.exists() else base / "submission/data/jobs.json"


def synthetic_demo(path, original):
    """All modified observations are synthetic and isolated from the actual demo."""
    label = "SYNTHETIC_MODIFIED_OBSERVATION_NOT_REAL_SOURCE_EVENT"
    base = copy.deepcopy(original)
    changed = copy.deepcopy(original)
    changed["job_description"] = (
        changed.get("job_description") or ""
    ) + "\nSynthetic integration test: responsibilities changed."
    events = [
        {"job": base, "observed_at": "2030-01-01T00:00:00Z", "label": label},
        {"job": base, "observed_at": "2030-01-02T00:00:00Z", "label": label},
        {"job": changed, "observed_at": "2030-01-03T00:00:00Z", "label": label},
        {
            "job": changed,
            "observed_at": "2030-01-04T00:00:00Z",
            "signal": "closed",
            "label": label,
        },
        {"job": changed, "observed_at": "2030-01-05T00:00:00Z", "label": label},
        {
            "job": changed,
            "observed_at": "2030-01-06T00:00:00Z",
            "signal": "reopened",
            "label": label,
        },
        {"job": base, "observed_at": "2029-12-31T00:00:00Z", "label": label},
    ]
    db = connect(path)
    try:
        results = [process(db, [event]) for event in events]
        return {
            "label": label,
            "events": events,
            "results": results,
            "history": [
                dict(r)
                for r in db.execute("SELECT * FROM job_change_history ORDER BY id")
            ],
        }
    finally:
        db.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=dataset_path())
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "demo-output"
    )
    parser.add_argument(
        "--mode", choices=("deterministic", "hybrid"), default="deterministic"
    )
    args = parser.parse_args(argv)
    saved = args.input.read_bytes()
    payload = json.loads(saved)
    observations = [
        {"job": row, "observed_at": payload["observed_at"]} for row in payload["jobs"]
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    transport, model = None, None
    if args.mode == "hybrid":
        # Explicit opt-in only. Read exported OLLAMA_* variables, never .env/Nhost.
        import httpx

        base_url, model = (
            os.environ.get("OLLAMA_BASE_URL"),
            os.environ.get("OLLAMA_MODEL"),
        )
        if not base_url or not model:
            parser.error(
                "Hybrid mode requires exported OLLAMA_BASE_URL and OLLAMA_MODEL"
            )

        def transport(request):
            response = httpx.post(
                base_url.rstrip("/") + "/api/generate",
                json=request,
                timeout=httpx.Timeout(300, connect=10),
            )
            response.raise_for_status()
            return response.json()

    db = connect(args.output / "jobs.sqlite3")
    try:
        first = process(
            db, observations, mode=args.mode, ollama_call=transport, model_name=model
        )
        before = list(db.iterdump())
        repeat = process(
            db, observations, mode=args.mode, ollama_call=transport, model_name=model
        )
        if before != list(db.iterdump()):
            raise AssertionError("Identical input changed persisted state")
        export_csv(db, args.output / "canonical_jobs.csv")
        # Never overwrite a human's completed label template on another run.
        template = args.output / "human_labels_template.csv"
        fresh_template = (
            args.output / "human_labels_template.generated.csv"
            if template.exists()
            else template
        )
        selection = make_template(db, fresh_template)
        pending = evaluate(fresh_template)
        synthetic = synthetic_demo(
            args.output / "synthetic_lifecycle.sqlite3", payload["jobs"][0]
        )
        write_json(args.output / "synthetic_lifecycle.json", synthetic)
        summary = {
            "input_sha256": hashlib.sha256(saved).hexdigest(),
            "input_records": len(observations),
            "input_observed_at": payload["observed_at"],
            "initial_run": first,
            "identical_repeat": repeat,
            "repeat_database_identical": True,
            "evaluation_selection": selection,
            "human_evaluation_status": pending["status"],
            "synthetic_history_rows": len(synthetic["history"]),
            "notice": "Local SQLite demonstration only. Availability unverified. Deterministic execution is not LLM inference. No production persistence tested.",
        }
        write_json(args.output / "summary.json", summary)
        write_json(args.output / "evaluation_pending.json", pending)
        if args.input.read_bytes() != saved:
            raise AssertionError("Input snapshot was modified")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "initial_counts": first["counts"],
                    "repeat_counts": repeat["counts"],
                    "totals": first["totals"],
                    "repeat_database_identical": True,
                    "model_calls": first["model_calls"] + repeat["model_calls"],
                    "synthetic_history_rows": len(synthetic["history"]),
                    "human_evaluation_status": pending["status"],
                },
                indent=2,
            )
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
