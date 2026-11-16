"""Version-aware fingerprints and separate preview-storage/cache decisions.

No I/O, environment access, credentials, inference, or main.py imports.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .adapters import NORMALIZATION_VERSION, normalize_job
from .contracts import build_tasks
from .deterministic import apply_jd_rules
from .extraction import classify_fields, prepare_sources
from .prompts import (
    EVIDENCE_POLICY_VERSION,
    EXTRACTION_PROMPT_VERSION,
    JOB_INTELLIGENCE_SCHEMA_VERSION,
    PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    build_ollama_request,
)
from .schemas import JobExtractionInput, JobExtractionResult

FINGERPRINT_VERSION = "job_extraction_fingerprint_v1"
PERSISTENCE_VERSION = "job_extraction_preview_storage_v1"
PERSISTENCE_POLICY_VERSION = "job_extraction_preview_policy_v2"
CACHE_POLICY_VERSION = "job_extraction_cache_policy_v1"
# Compatibility name for callers of the original preview persistence interface.
POLICY_VERSION = PERSISTENCE_POLICY_VERSION
MEANINGFUL_DETERMINISTIC_FIELDS = frozenset(
    {
        "city",
        "country",
        "seniority",
        "remote_policy",
        "employment_type",
        "skills",
        "salary_min",
        "salary_max",
        "currency",
        "visa_sponsorship",
    }
)
OPERATIONAL_FIELDS = frozenset({"is_active", "first_seen_at", "last_seen_at"})
INPUT_FIELDS = tuple(
    name for name in JobExtractionInput.model_fields if name not in OPERATIONAL_FIELDS
)


def json_value(value):
    """Preserve strings/numeric types; canonicalize UUID/date objects only."""
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def canonical_json(value) -> str:
    return json.dumps(
        json_value(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def current_versions() -> dict[str, str]:
    return {
        "normalization_version": NORMALIZATION_VERSION,
        "schema_version": JOB_INTELLIGENCE_SCHEMA_VERSION,
        "prompt_version": EXTRACTION_PROMPT_VERSION,
        "evidence_policy_version": EVIDENCE_POLICY_VERSION,
    }


def generation_settings(model_name: str) -> dict:
    request = build_ollama_request(
        model_name=model_name,
        sources={},
        resolved_fields={},
        eligible_fields=[],
        check_fields=[],
        blocked_fields=[],
    )
    return {
        key: deepcopy(request[key])
        for key in ("options", "stream", "think", "keep_alive")
    }


class ModelIdentity(BaseModel):
    """Operator-established immutable identity; a mutable tag is insufficient.

    Digests must be verified/pinned by the caller's deployment. This package
    cannot discover or attest a digest through an /api/generate callable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    name: str = Field(min_length=1)
    digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    deployment_fingerprint: str | None = Field(default=None, min_length=1)

    @property
    def trusted(self) -> bool:
        return bool(
            (self.digest or (self.deployment_fingerprint or "").strip())
            and self.name.strip()
        )


@dataclass(frozen=True)
class ExtractionConfiguration:
    model: ModelIdentity
    versions: dict[str, str] = field(default_factory=current_versions)
    generation: dict | None = None

    def payload(self):
        return {
            "versions": deepcopy(self.versions),
            "model_identity": self.model.model_dump(mode="json"),
            "generation": deepcopy(
                self.generation
                if self.generation is not None
                else generation_settings(self.model.name)
            ),
            "prompt_content_sha256": hashlib.sha256(
                (SYSTEM_PROMPT + "\n" + PROMPT_TEMPLATE).encode()
            ).hexdigest(),
            "policy_version": POLICY_VERSION,
        }


@dataclass(frozen=True)
class InputFingerprint:
    digest: str
    payload: dict


def extraction_input(row: dict) -> dict:
    validated = JobExtractionInput.model_validate(row)
    source = {key: deepcopy(row.get(key)) for key in INPUT_FIELDS}
    source["id"] = str(validated.id)
    return json_value(source)


def build_fingerprint(
    row: dict, configuration: ExtractionConfiguration
) -> InputFingerprint:
    source = extraction_input(row)
    payload = {
        "fingerprint_version": FINGERPRINT_VERSION,
        "input": source,
        "normalized_metadata": normalize_job(source).fields.model_dump(mode="json"),
        "configuration": configuration.payload(),
    }
    return InputFingerprint(
        hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest(), payload
    )


@dataclass(frozen=True)
class EligibilityDecision:
    allowed: bool
    reason_codes: tuple[str, ...]
    policy_version: str = PERSISTENCE_POLICY_VERSION

    def to_dict(self):
        return {
            "allowed": self.allowed,
            "reason_codes": list(self.reason_codes),
            "policy_version": self.policy_version,
        }


def _decision(reasons, policy_version):
    return EligibilityDecision(
        not reasons, tuple(dict.fromkeys(reasons)), policy_version
    )


def _source_at(row, path):
    value = row
    for component in re.sub(r"\[(\d+)\]", r".\1", path).split("."):
        value = value[int(component)] if isinstance(value, list) else value[component]
    return value


def evidence_matches_input(result, row) -> bool:
    """Validate audit provenance against the fetched input, not self-quoted text."""
    source = extraction_input(row)
    if canonical_json(result.source_metadata) != canonical_json(
        source.get("raw_source_data") or {}
    ):
        return False
    evidence = list(result.evidence)
    for flag in result.review_flags:
        evidence.extend(flag.existing_evidence)
        evidence.extend(flag.conflicting_evidence)
    for candidate in result.diagnostics.candidates:
        evidence.extend(candidate.evidence)
    try:
        for item in evidence:
            if canonical_json(_source_at(source, item.source_path)) != canonical_json(
                item.original_value
            ):
                return False
        for record in result.diagnostics.source_registry:
            if _source_at(source, record["source_path"]) != record["text"]:
                return False
            for path, text in record.get("equivalent_sources", {}).items():
                if _source_at(source, path) != text:
                    return False
    except (KeyError, IndexError, TypeError, ValueError):
        return False
    return True


def _current_plan(row):
    source = extraction_input(row)
    sources = prepare_sources(JobExtractionInput.model_validate(source))
    normalized, flags, signals, _ = apply_jd_rules(normalize_job(source), sources)
    policy = classify_fields(normalized)
    tasks = build_tasks(
        [p.field_name for p in policy if p.extraction_allowed],
        [p.field_name for p in policy if p.contradiction_check],
    )
    return normalized, tasks, flags, signals


def _result_eligibility(
    result: JobExtractionResult | dict,
    row: dict,
    configuration: ExtractionConfiguration,
    *,
    allow_partial: bool,
    policy_version: str,
) -> EligibilityDecision:
    """Shared integrity checks; only the content-volume gates vary by policy."""
    try:
        result = JobExtractionResult.model_validate(
            result.model_dump(mode="json")
            if isinstance(result, JobExtractionResult)
            else deepcopy(result)
        )
        canonical_json(result.model_dump(mode="json"))
    except (ValidationError, TypeError, ValueError):
        return _decision(["result_contract_invalid"], policy_version)
    source = extraction_input(row)
    reasons = []
    if (
        str(result.job_id) != source["id"]
        or result.source != source["source"].strip().casefold()
    ):
        reasons.append("job_identity_mismatch")
    if not configuration.model.trusted:
        reasons.append("immutable_model_identity_missing")
    if configuration.payload()["generation"] != generation_settings(
        configuration.model.name
    ):
        reasons.append("generation_configuration_mismatch")
    if (
        result.metadata.model_name != configuration.model.name
        or result.metrics.model_name != configuration.model.name
    ):
        reasons.append("model_name_mismatch")
    if configuration.versions != current_versions() or any(
        getattr(result.metadata, key) != value
        for key, value in configuration.versions.items()
    ):
        reasons.append("extraction_version_mismatch")
    if result.review_required or result.review_flags:
        reasons.append("verified_review_required")
    if result.manual_inspection_required or result.unverified_conflicts:
        reasons.append("manual_inspection_required")
    if not evidence_matches_input(result, source):
        reasons.append("evidence_input_mismatch")
    normalized, tasks, deterministic_flags, signals = _current_plan(source)
    useful_deterministic = any(
        name in MEANINGFUL_DETERMINISTIC_FIELDS
        and value is not None
        and value != ""
        and value != []
        for name, value in normalized.fields.model_dump(mode="json").items()
    )
    if not useful_deterministic:
        reasons.append("insufficient_deterministic_information")
    final_fields = result.fields.model_dump(mode="json")
    if any(
        value is not None
        and canonical_json(final_fields[name]) != canonical_json(value)
        for name, value in normalized.fields.model_dump(mode="json").items()
    ):
        reasons.append("protected_deterministic_value_changed")
    if deterministic_flags or signals:
        reasons.append("source_requires_review_or_inspection")
    diagnostics = result.diagnostics
    if diagnostics.skip_reason == "context_budget_exceeded":
        reasons.append("context_limit_skip")
    if result.extraction_status not in {
        "succeeded",
        "succeeded_with_warnings",
        "inference_skipped",
    }:
        reasons.append("extraction_not_complete")
    if result.inference_status == "skipped":
        # Independently establish zero necessary tasks. The frozen engine
        # normally still requests protected-scalar checks, so this is rare.
        deterministic_only = (
            not tasks
            and result.metrics.model_calls == 0
            and not diagnostics.candidates
            and diagnostics.skip_reason == "no_meaningful_tasks"
            and not any(e.extraction_method == "llm" for e in result.evidence)
            and canonical_json(result.fields.model_dump(mode="json"))
            == canonical_json(normalized.fields.model_dump(mode="json"))
        )
        if not deterministic_only:
            reasons.append("necessary_inference_skipped")
    else:
        if result.metrics.model_calls != 1:
            reasons.append("historical_model_call_count_invalid")
        if result.inference_status != "succeeded":
            reasons.append("inference_not_successful")
        if diagnostics.failure_reason or diagnostics.skip_reason:
            reasons.append("processing_failure_or_skip")
        if (
            diagnostics.provider_completed is not True
            or result.metrics.provider.done is not True
            or result.metrics.provider.done_reason not in {None, "stop", "eos"}
        ):
            reasons.append("provider_not_complete")
        if not result.metadata.full_source_processed:
            reasons.append("processing_not_complete")
        invalid_contract_reasons = {
            "invalid_candidate_schema",
            "invalid_task_schema",
            "unsupported_source_id",
            "duplicate_task_id",
            "unsolicited_task_id",
            "unsupported_response_contract",
        }
        if diagnostics.response_schema_status != "valid" or any(
            candidate.reason_code in invalid_contract_reasons
            for candidate in diagnostics.candidates
        ):
            reasons.append("response_contract_invalid")
        accounting = diagnostics.task_accounting
        expected = {task.task_id for task in tasks}
        accounted = (
            accounting.status == "complete"
            and set(accounting.requested) == expected
            and len(accounting.requested) == len(expected)
            and set(accounting.received) == expected
            and len(accounting.received) == len(expected)
            and set(accounting.dispositions) == expected
            and not any(
                (
                    accounting.missing,
                    accounting.invalid,
                    accounting.duplicate,
                    accounting.unsolicited,
                )
            )
            and result.metrics.requested_task_count == len(expected)
            and result.metrics.accounted_task_count == len(expected)
        )
        if not accounted:
            reasons.append("task_accounting_incomplete")
        extraction_tasks = [task.task_id for task in tasks if task.mode == "extract"]
        reported_all_unknown = bool(extraction_tasks) and all(
            accounting.dispositions.get(task_id) in {"not_found", "ambiguous"}
            for task_id in extraction_tasks
        )
        require_enrichment = not allow_partial or not useful_deterministic
        if require_enrichment and (
            diagnostics.all_unknown_response is True or reported_all_unknown
        ):
            reasons.append("all_unknown_response")
        if require_enrichment and diagnostics.empty_useful_content is not False:
            reasons.append("empty_useful_content")
        useful = {
            name
            for name, value in result.fields.model_dump(mode="json").items()
            if value is not None and getattr(normalized.fields, name) is None
        }
        if require_enrichment and not useful:
            reasons.append("no_new_useful_fields")
        if result.metrics.accepted_llm_field_count != len(useful):
            reasons.append("accepted_field_metrics_mismatch")
        # Waiving content-volume gates must not permit fields to be attributed
        # to tasks reported as unknown. Zero accepted fields remains legitimate.
        if any(
            not any(
                task.mode == "extract"
                and name in task.fields
                and accounting.dispositions.get(task.task_id)
                in {"found", "conflicting"}
                for task in tasks
            )
            for name in useful
        ):
            reasons.append("accepted_field_task_disposition_mismatch")
    return _decision(reasons, policy_version)


def preview_persistence_eligibility(
    result: JobExtractionResult | dict,
    row: dict,
    configuration: ExtractionConfiguration,
) -> EligibilityDecision:
    """Allow complete evidence-valid partial previews with useful source facts.

    Unknown/empty-content diagnostics remain unchanged. This is not canonical
    promotion approval. Freshness/CAS and authorization are repository gates.
    """
    return _result_eligibility(
        result,
        row,
        configuration,
        allow_partial=True,
        policy_version=PERSISTENCE_POLICY_VERSION,
    )


def cache_reuse_eligibility(
    result: JobExtractionResult | dict,
    row: dict,
    configuration: ExtractionConfiguration,
    *,
    allow_partial_cache: bool = False,
) -> EligibilityDecision:
    """Conservative reuse by default; optional partial reuse keeps every guard.

    The preference changes reuse, not the produced extraction or its identity.
    Envelope, fingerprint and configuration checks also run during lookup.
    """
    if type(allow_partial_cache) is not bool:
        raise TypeError("allow_partial_cache must be an explicit boolean")
    return _result_eligibility(
        result,
        row,
        configuration,
        allow_partial=allow_partial_cache,
        policy_version=CACHE_POLICY_VERSION,
    )


# Keep the repository's existing interface without changing its CAS implementation.
persistence_eligibility = preview_persistence_eligibility


class StoredExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    storage_version: str
    fingerprint_version: str
    input_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    configuration: dict[str, Any]
    write_id: UUID
    stored_at: datetime
    result: JobExtractionResult


def make_stored_extraction(result, row, configuration, fingerprint) -> dict:
    result = JobExtractionResult.model_validate(
        result.model_dump(mode="json")
        if isinstance(result, JobExtractionResult)
        else deepcopy(result)
    )
    decision = preview_persistence_eligibility(result, row, configuration)
    if not decision.allowed:
        raise ValueError("ineligible_extraction")
    if fingerprint.digest != build_fingerprint(row, configuration).digest:
        raise ValueError("fingerprint_mismatch")
    envelope = StoredExtraction(
        storage_version=PERSISTENCE_VERSION,
        fingerprint_version=FINGERPRINT_VERSION,
        input_fingerprint=fingerprint.digest,
        configuration=configuration.payload(),
        write_id=uuid4(),
        stored_at=datetime.now(timezone.utc),
        result=result.model_copy(deep=True),
    )
    return envelope.model_dump(mode="json")


@dataclass(frozen=True)
class CacheLookup:
    hit: bool
    result: JobExtractionResult | None
    reason_codes: tuple[str, ...]


def lookup_cached_extraction(
    row: dict,
    stored: Any,
    stored_hash: Any,
    configuration: ExtractionConfiguration,
    *,
    force_refresh: bool = False,
    allow_partial_cache: bool = False,
) -> CacheLookup:
    if type(force_refresh) is not bool or type(allow_partial_cache) is not bool:
        raise TypeError("cache preferences must be explicit booleans")
    if force_refresh:
        return CacheLookup(False, None, ("forced_refresh",))
    if not configuration.model.trusted:
        return CacheLookup(False, None, ("immutable_model_identity_missing",))
    fingerprint = build_fingerprint(row, configuration)
    if stored is None:
        return CacheLookup(False, None, ("no_stored_extraction",))
    if stored_hash != fingerprint.digest:
        return CacheLookup(False, None, ("input_fingerprint_mismatch",))
    try:
        canonical_json(stored)
        envelope = StoredExtraction.model_validate(deepcopy(stored))
    except (ValidationError, TypeError, ValueError):
        return CacheLookup(False, None, ("cached_envelope_invalid",))
    if (
        envelope.storage_version != PERSISTENCE_VERSION
        or envelope.fingerprint_version != FINGERPRINT_VERSION
    ):
        return CacheLookup(False, None, ("cached_storage_version_mismatch",))
    try:
        configuration_matches = canonical_json(
            envelope.configuration
        ) == canonical_json(configuration.payload())
    except (TypeError, ValueError):
        return CacheLookup(False, None, ("cached_configuration_invalid",))
    if envelope.input_fingerprint != fingerprint.digest or not configuration_matches:
        return CacheLookup(False, None, ("cached_configuration_mismatch",))
    decision = cache_reuse_eligibility(
        envelope.result,
        row,
        configuration,
        allow_partial_cache=allow_partial_cache,
    )
    if not decision.allowed:
        return CacheLookup(False, None, decision.reason_codes)
    return CacheLookup(True, envelope.result.model_copy(deep=True), ())
