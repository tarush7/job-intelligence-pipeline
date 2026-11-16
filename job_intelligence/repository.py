"""Injected Hasura repository; parameterized, one-row compare-and-set writes.

The existing transport returns inner data, not {"data": ...}. No schema changes,
credentials, retries, canonical identifiers, or main.py imports live here.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from uuid import UUID

from pydantic import ValidationError

from .cache import (
    INPUT_FIELDS,
    EligibilityDecision,
    ExtractionConfiguration,
    InputFingerprint,
    build_fingerprint,
    canonical_json,
    extraction_input,
    json_value,
    make_stored_extraction,
    persistence_eligibility,
)
from .schemas import JobExtractionInput

SOURCE_FIELDS = tuple(JobExtractionInput.model_fields)
STORED_FIELDS = ("structured_extraction", "extraction_input_hash")
FETCH_JOB_QUERY = (
    """query JobIntelligenceSource($job_id: uuid!) {
  jobs(where: {id: {_eq: $job_id}}, limit: 2) {
    """
    + "\n    ".join((*SOURCE_FIELDS, *STORED_FIELDS))
    + """
  }
}"""
)
FETCH_EXISTING_QUERY = """query JobIntelligenceExisting($job_id: uuid!) {
  jobs(where: {id: {_eq: $job_id}}, limit: 2) {
    id
    structured_extraction
    extraction_input_hash
  }
}"""
SAVE_EXTRACTION_MUTATION = """mutation SaveJobIntelligencePreview(
  $guard: jobs_bool_exp!, $result: jsonb!, $fingerprint: String!
) {
  update_jobs(
    where: $guard
    _set: {structured_extraction: $result, extraction_input_hash: $fingerprint}
  ) {
    affected_rows
    returning { id extraction_input_hash structured_extraction }
  }
}"""


class RepositoryError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class RepositoryTransportError(RepositoryError):
    pass


@dataclass(frozen=True)
class JobSnapshot:
    source: dict
    structured_extraction: object
    extraction_input_hash: str | None

    @property
    def job_id(self):
        return str(UUID(str(self.source["id"])))


@dataclass(frozen=True)
class ExistingExtraction:
    job_id: str
    structured_extraction: object
    extraction_input_hash: str | None


@dataclass(frozen=True)
class PersistenceOutcome:
    status: str
    reason_codes: tuple[str, ...] = ()
    serialized_result_bytes: int | None = None
    serialized_envelope_bytes: int | None = None
    mutation_attempted: bool = False

    def to_dict(self):
        return {
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "serialized_result_bytes": self.serialized_result_bytes,
            "serialized_envelope_bytes": self.serialized_envelope_bytes,
            "mutation_attempted": self.mutation_attempted,
        }


def _literal(text):
    # Shared Python/PostgreSQL ARE metacharacters; do not escape ordinary spaces.
    return re.sub(r"([\\.^$|?*+()\[\]{}])", r"\\\1", text)


def jsonb_snapshot_pattern(value) -> str:
    """Match an exact typed JSON value in PostgreSQL jsonb::text.

    Object key order/formatting is immaterial. Arrays retain order and numeric
    integer/float types remain distinct. JSONB guarantees unique object keys,
    so N pairs from N distinct allowed key/value alternatives covers every key.
    Optional commas cannot accept invalid JSON because the DB value is JSONB.
    No Python re escaping of whitespace, locale regex classes, or lookbehinds.
    Unsupported/oversized guards fail closed instead of weakening comparisons.
    """
    import json

    def pattern(item):
        if item is None:
            return "null"
        if isinstance(item, bool):
            return "true" if item else "false"
        if isinstance(item, str):
            return _literal(json.dumps(item, ensure_ascii=False))
        if isinstance(item, int):
            return str(item)
        if isinstance(item, float):
            if not math.isfinite(item):
                raise RepositoryError("nonfinite_json_guard")
            decimal = format(Decimal(repr(item)), "f")
            if "." not in decimal:
                # A fetched JSONB integer is decoded as int. Do not silently
                # pretend a float with exponent syntax was the same input type.
                decimal += ".0"
            decimal = decimal.rstrip("0")
            return _literal(decimal) + ("0+" if decimal.endswith(".") else "0*")
        if isinstance(item, list):
            return (
                r"\[\s*" + r"\s*,\s*".join(pattern(child) for child in item) + r"\s*\]"
            )
        if isinstance(item, dict):
            if not item:
                return r"\{\s*\}"
            if len(item) > 255 or any(not isinstance(key, str) for key in item):
                raise RepositoryError("json_guard_object_unsupported")
            alternatives = "|".join(
                _literal(json.dumps(key, ensure_ascii=False))
                + r"\s*:\s*"
                + pattern(child)
                for key, child in item.items()
            )
            return (
                r"\{(?:\s*(?:" + alternatives + r")\s*,?){" + str(len(item)) + r"}\s*\}"
            )
        raise RepositoryError("json_guard_value_unsupported")

    result = "^" + pattern(value) + "$"
    if len(result.encode()) > 512 * 1024:
        raise RepositoryError("json_guard_too_large")
    return result


def _column_guard(name, value):
    if value is None:
        return {name: {"_is_null": True}}
    if name in {"raw_source_data", "structured_extraction"}:
        # Hasura JSONB equality/containment is not a sufficiently strict source
        # guard (array ordering and numeric representations matter to evidence).
        return {name: {"_cast": {"String": {"_regex": jsonb_snapshot_pattern(value)}}}}
    return {name: {"_eq": json_value(value)}}


def persistence_guard(snapshot: JobSnapshot) -> dict:
    source = extraction_input(snapshot.source)
    clauses = [_column_guard(name, source[name]) for name in INPUT_FIELDS]
    clauses.extend(
        (
            _column_guard("structured_extraction", snapshot.structured_extraction),
            _column_guard("extraction_input_hash", snapshot.extraction_input_hash),
        )
    )
    return {"_and": clauses}


class JobRepository:
    def __init__(self, graphql_call: Callable[[str, dict], dict]):
        self.graphql_call = graphql_call

    def _call(self, operation, variables):
        try:
            data = self.graphql_call(operation, deepcopy(variables))
        except Exception as error:  # noqa: BLE001 -- transport exceptions must not expose credentials/URLs
            raise RepositoryTransportError(
                "graphql_transport_" + type(error).__name__
            ) from None
        if not isinstance(data, dict) or "data" in data or data.get("errors"):
            raise RepositoryError("graphql_inner_data_contract_invalid")
        return data

    def _one(self, data, job_id, fields):
        jobs = data.get("jobs")
        if not isinstance(jobs, list):
            raise RepositoryError("graphql_jobs_shape_invalid")
        if not jobs:
            raise RepositoryError("job_not_found")
        if len(jobs) != 1 or not isinstance(jobs[0], dict):
            raise RepositoryError("graphql_single_row_required")
        row = jobs[0]
        if not set(fields).issubset(row):
            raise RepositoryError("graphql_projection_incomplete")
        try:
            if str(UUID(str(row["id"]))) != job_id:
                raise RepositoryError("returned_job_identity_mismatch")
            canonical_json(row)
        except (TypeError, ValueError) as error:
            if isinstance(error, RepositoryError):
                raise
            raise RepositoryError("graphql_row_invalid") from None
        if row["extraction_input_hash"] is not None and not isinstance(
            row["extraction_input_hash"], str
        ):
            raise RepositoryError("graphql_hash_type_invalid")
        return row

    def fetch_job_for_extraction(self, job_id: UUID | str) -> JobSnapshot:
        job_id = str(UUID(str(job_id)))
        row = self._one(
            self._call(FETCH_JOB_QUERY, {"job_id": job_id}),
            job_id,
            (*SOURCE_FIELDS, *STORED_FIELDS),
        )
        source = {name: deepcopy(row[name]) for name in SOURCE_FIELDS}
        try:
            validated = JobExtractionInput.model_validate(source)
            if validated.source.strip().casefold() not in {
                "linkedin",
                "indeed",
                "naukri",
            }:
                raise ValueError("unsupported source")
        except (ValidationError, TypeError, ValueError):
            raise RepositoryError("extraction_input_invalid") from None
        return JobSnapshot(
            source, deepcopy(row["structured_extraction"]), row["extraction_input_hash"]
        )

    def fetch_existing_extraction(self, job_id: UUID | str) -> ExistingExtraction:
        job_id = str(UUID(str(job_id)))
        row = self._one(
            self._call(FETCH_EXISTING_QUERY, {"job_id": job_id}),
            job_id,
            ("id", *STORED_FIELDS),
        )
        return ExistingExtraction(
            job_id, deepcopy(row["structured_extraction"]), row["extraction_input_hash"]
        )

    def save_structured_extraction(
        self,
        snapshot,
        result,
        fingerprint: InputFingerprint,
        configuration: ExtractionConfiguration,
        *,
        authorize_write: bool = False,
    ) -> PersistenceOutcome:
        if authorize_write is not True:
            return PersistenceOutcome(
                "not_authorized", ("explicit_save_authorization_required",)
            )
        decision: EligibilityDecision = persistence_eligibility(
            result, snapshot.source, configuration
        )
        if not decision.allowed:
            return PersistenceOutcome("blocked", decision.reason_codes)
        if (
            build_fingerprint(snapshot.source, configuration).digest
            != fingerprint.digest
        ):
            return PersistenceOutcome("conflict", ("snapshot_fingerprint_mismatch",))
        try:
            envelope = make_stored_extraction(
                result, snapshot.source, configuration, fingerprint
            )
            guard = persistence_guard(snapshot)
        except (RepositoryError, ValueError) as error:
            return PersistenceOutcome(
                "blocked",
                (
                    error.code
                    if isinstance(error, RepositoryError)
                    else "persistence_payload_invalid",
                ),
            )
        result_size = len(canonical_json(envelope["result"]).encode())
        envelope_size = len(canonical_json(envelope).encode())
        variables = {
            "guard": guard,
            "result": envelope,
            "fingerprint": fingerprint.digest,
        }
        try:
            data = self._call(SAVE_EXTRACTION_MUTATION, variables)
        except RepositoryError:
            # A response can be lost after commit. Never retry blindly or claim
            # rollback; reread/review before an explicitly guarded new attempt.
            return PersistenceOutcome(
                "outcome_unknown",
                ("mutation_outcome_unconfirmed",),
                result_size,
                envelope_size,
                mutation_attempted=True,
            )
        response = data.get("update_jobs")
        if not isinstance(response, dict):
            return PersistenceOutcome(
                "outcome_unknown",
                ("mutation_response_invalid",),
                result_size,
                envelope_size,
                mutation_attempted=True,
            )
        affected = response.get("affected_rows")
        if type(affected) is int and affected == 0 and response.get("returning") == []:
            return PersistenceOutcome(
                "conflict",
                ("source_or_stored_extraction_changed",),
                result_size,
                envelope_size,
                mutation_attempted=True,
            )
        returning = response.get("returning")
        try:
            confirmed = (
                type(affected) is int
                and affected == 1
                and isinstance(returning, list)
                and len(returning) == 1
                and str(UUID(returning[0]["id"])) == snapshot.job_id
                and returning[0]["extraction_input_hash"] == fingerprint.digest
                and canonical_json(returning[0]["structured_extraction"])
                == canonical_json(envelope)
            )
        except (TypeError, ValueError, KeyError, IndexError):
            confirmed = False
        return PersistenceOutcome(
            "saved" if confirmed else "outcome_unknown",
            () if confirmed else ("mutation_response_mismatch",),
            result_size,
            envelope_size,
            mutation_attempted=True,
        )
