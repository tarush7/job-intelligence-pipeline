"""Single-call hybrid extraction with protected deterministic values and no I/O.

The injected callable owns the existing bounded HTTP transport. This module
does not read credentials, databases or environment variables and never retries.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from .adapters import NORMALIZATION_VERSION, normalize_job
from .contracts import build_registry, build_tasks, process_tasks
from .deterministic import apply_jd_rules
from .prompts import (
    EVIDENCE_POLICY_VERSION,
    EXTRACTION_PROMPT_VERSION,
    JOB_INTELLIGENCE_SCHEMA_VERSION,
    NUM_CTX,
    build_ollama_request,
    context_estimate_breakdown,
    estimate_context_tokens,
)
from .schemas import (
    CandidateDiagnostic,
    ExtractionDiagnostics,
    ExtractionMetadata,
    ExtractionMetrics,
    FieldEligibility,
    JobExtractionInput,
    JobExtractionResult,
    JobNormalizationResult,
    NormalizationWarning,
    OllamaMetrics,
    RawCompensationCandidate,
    RawFieldCandidate,
    RawSkillCandidate,
    ReviewFlag,
    StructuredJobFields,
    TaskAccounting,
)
from .validators import (
    SALARY_FIELDS,
    CandidateRejected,
    make_evidence,
    validate_compensation,
    validate_scalar,
    validate_skill,
)

# Company/title are source identifiers, not generative replacement targets.
# Skills supplementation is deliberately deferred. Resolved scalar fields can
# still be checked for contradictions, including company/title and compensation.
EXCLUDED_EXTRACTION_FIELDS = frozenset({"company", "title"})
_CONFLICT_CODES = {"metadata_conflict", "unreconciled_metadata", "jd_location_conflict"}
_SOURCE_FIELDS = {
    "company": {"company"},
    "job_title": {"title"},
    "location": {"city", "country"},
    "employment_type": {"employment_type"},
    "seniority_level": {"seniority"},
    "raw_source_data.payload.companyName": {"company"},
    "raw_source_data.payload.company": {"company"},
    "raw_source_data.payload.title": {"title"},
    "raw_source_data.payload.location": {"city", "country"},
    "raw_source_data.payload.employmentType": {"employment_type"},
    "raw_source_data.payload.job_type": {"employment_type"},
    "raw_source_data.payload.seniorityLevel": {"seniority"},
    "raw_source_data.payload.job_level": {"seniority"},
    "raw_source_data.payload.salary": set(SALARY_FIELDS),
    "raw_source_data.payload.work_from_home_type": {"remote_policy"},
    "raw_source_data.jobType": {"employment_type"},
    "raw_source_data.wfhType": {"remote_policy"},
    "raw_source_data.workModeLabel": {"remote_policy"},
}


def classify_fields(normalized: JobNormalizationResult) -> list[FieldEligibility]:
    blocked = {
        field
        for warning in normalized.warnings
        if warning.code in _CONFLICT_CODES
        for field in warning.fields
    }
    if blocked.intersection(SALARY_FIELDS):
        blocked.update(SALARY_FIELDS)
    ambiguous = {field for warning in normalized.warnings for field in warning.fields}
    result = []
    for field, value in normalized.fields.model_dump(mode="json").items():
        if value is not None:
            category, reason = "resolved", "protected_deterministic_value"
        elif field in blocked:
            category, reason = "conflict_blocked", "source_metadata_requires_review"
        elif field in EXCLUDED_EXTRACTION_FIELDS:
            category, reason = "excluded", "source_identity_is_not_generative"
        elif field in ambiguous:
            category, reason = "ambiguous", "unsupported_or_ambiguous_source_value"
        else:
            category, reason = "missing", "no_resolved_source_value"
        result.append(
            FieldEligibility(
                field_name=field,
                category=category,
                extraction_allowed=category in {"missing", "ambiguous"},
                contradiction_check=category == "resolved" and field != "skills",
                reason=reason,
            )
        )
    return result


def _get(row: dict, path: str):
    value = row
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def prepare_sources(job: JobExtractionInput) -> dict[str, str]:
    """Only allowlisted plain source facts; no identifiers/URLs/HTML/statistics."""
    row = job.model_dump(mode="json")
    sources = {}
    source = job.source.strip().casefold()
    common_paths = [
        "job_description",
        "company",
        "job_title",
        "location",
        "employment_type",
        "seniority_level",
    ]
    platform_paths = {
        "linkedin": [
            "raw_source_data.payload." + name
            for name in (
                "descriptionText",
                "companyName",
                "title",
                "location",
                "employmentType",
                "seniorityLevel",
                "salary",
            )
        ],
        "indeed": [
            "raw_source_data.payload." + name
            for name in (
                "company",
                "title",
                "location",
                "job_type",
                "job_level",
                "work_from_home_type",
            )
        ],
        "naukri": [
            "raw_source_data." + name
            for name in ("jobType", "wfhType", "workModeLabel")
        ],
    }
    for path in [*common_paths, *platform_paths[source]]:
        value = _get(row, path)
        if isinstance(value, str) and value.strip():
            sources[path] = value
    skill_paths = {
        "linkedin": [],
        "indeed": ["raw_source_data.payload.skills"],
        "naukri": ["raw_source_data.skills"],
    }
    for path in skill_paths[source]:
        skills = _get(row, path)
        if isinstance(skills, list):
            for index, value in enumerate(skills):
                if isinstance(value, str) and value.strip():
                    sources[f"{path}[{index}]"] = value
    return sources


def _meaningful_sources(
    sources: dict[str, str], eligible: set[str], checks: list[str]
) -> bool:
    if any(
        path in sources
        for path in ("job_description", "raw_source_data.payload.descriptionText")
    ):
        return bool(eligible or checks)
    return any(
        bool(
            eligible.intersection(
                _SOURCE_FIELDS.get(path, {"skills"} if ".skills[" in path else set())
            )
        )
        for path in sources
    )


def _provider_metrics(data: dict) -> OllamaMetrics:
    values = {}
    for key in (
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    ):
        value = data.get(key)
        name = key + "_ns" if key.endswith("duration") else key
        values[name] = value if type(value) is int and value >= 0 else None
    values["done"] = data.get("done") if type(data.get("done")) is bool else None
    values["done_reason"] = (
        data.get("done_reason") if isinstance(data.get("done_reason"), str) else None
    )
    return OllamaMetrics(**values)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_response_key")
        result[key] = value
    return result


def _tracked_object(pairs, duplicates):
    result = {}
    for key, value in pairs:
        if key in result:
            duplicates.append(key)
        else:
            result[key] = value
    return result


class _Merge:
    def __init__(self, normalized, sources, eligibility):
        self.normalized = normalized
        self.sources = sources
        self.policy = {item.field_name: item for item in eligibility}
        self.values = normalized.fields.model_dump(mode="json")
        self.evidence = deepcopy(normalized.evidence)
        self.warnings = deepcopy(normalized.warnings)
        self.flags: list[ReviewFlag] = []
        self.diagnostics: list[CandidateDiagnostic] = []
        self.accepted_fields: set[str] = set()
        self.pending: dict[str, list[tuple[Any, Any, CandidateDiagnostic]]] = {}
        self.salaries: list[tuple[dict, list, CandidateDiagnostic]] = []
        for warning in normalized.warnings:
            if (
                warning.code in _CONFLICT_CODES
                and warning.code != "jd_location_conflict"
            ):
                for field in warning.fields:
                    self.flag(
                        field,
                        "source_metadata_conflict",
                        None,
                        [],
                        source_values=warning.source_values,
                    )

    def flag(
        self,
        field,
        reason,
        value,
        evidence,
        *,
        existing_value=None,
        existing_evidence=None,
        source_values=None,
    ):
        flag = ReviewFlag(
            field_name=field,
            reason_code=reason,
            existing_value=self.values[field]
            if existing_value is None
            else existing_value,
            existing_evidence=(
                [item for item in self.evidence if item.field_name == field]
                if existing_evidence is None
                else existing_evidence
            ),
            conflicting_value=value,
            conflicting_evidence=evidence,
            source_values=source_values or {},
        )
        if flag not in self.flags:
            self.flags.append(flag)
            self.warnings.append(
                NormalizationWarning(
                    code=reason,
                    fields=[field],
                    message="Field requires review; deterministic values were preserved.",
                    source_values=source_values or {},
                )
            )

    def diagnostic(
        self,
        kind,
        raw,
        status="rejected",
        reason="invalid_candidate_schema",
        fields=None,
    ):
        item = CandidateDiagnostic(
            kind=kind,
            candidate=deepcopy(raw),
            status=status,
            reason_code=reason,
            fields=fields or [],
        )
        self.diagnostics.append(item)
        return item

    def field(self, raw, *, contradiction=False):
        diag = self.diagnostic("contradiction" if contradiction else "field", raw)
        try:
            candidate = RawFieldCandidate.model_validate(raw)
            field = candidate.field_name
            diag.fields = [field]
            if candidate.value is None:
                diag.status, diag.reason_code = (
                    "unknown",
                    (
                        "ambiguous_information"
                        if candidate.null_reason == "ambiguous"
                        else "absent_information"
                        if candidate.null_reason == "absent"
                        else "legitimate_unknown"
                    ),
                )
                return
            policy = self.policy[field]
            if policy.category in {"conflict_blocked", "excluded"}:
                raise CandidateRejected(policy.reason)
            if contradiction and not policy.contradiction_check:
                raise CandidateRejected("field_not_eligible_for_contradiction_check")
            if field in SALARY_FIELDS or field == "skills":
                raise CandidateRejected("field_requires_dedicated_candidate_type")
            value = validate_scalar(candidate, self.sources)
            evidence = make_evidence(
                field=field,
                value=value,
                candidate=candidate,
                sources=self.sources,
                source_platform=self.normalized.source,
                rule="explicit_field_semantics_v1",
            )
            diag.normalized_values, diag.evidence = {field: value}, [evidence]
            diag.transformation_rules = ["explicit_field_semantics_v1"]
            if policy.category == "resolved":
                existing = self.values[field]
                equivalent = (
                    existing.casefold() == value.casefold()
                    if isinstance(existing, str) and isinstance(value, str)
                    else existing == value
                )
                if not equivalent:
                    reason = (
                        "contradictory_work_mode"
                        if field == "remote_policy"
                        else f"contradictory_{field}"
                    )
                    self.flag(field, reason, value, [evidence])
                    diag.status, diag.reason_code = "conflict", reason
                else:
                    diag.status, diag.reason_code = (
                        "accepted",
                        "corroborates_protected_value",
                    )
                return
            if not policy.extraction_allowed:
                raise CandidateRejected("field_not_eligible")
            self.pending.setdefault(field, []).append((value, evidence, diag))
        except (ValidationError, CandidateRejected) as error:
            diag.reason_code = (
                str(error)
                if isinstance(error, CandidateRejected)
                else "invalid_candidate_schema"
            )

    def skill(self, raw):
        diag = self.diagnostic("skill", raw, fields=["skills"])
        try:
            candidate = RawSkillCandidate.model_validate(raw)
            if candidate.value is None:
                diag.status, diag.reason_code = (
                    "unknown",
                    (
                        "ambiguous_information"
                        if candidate.null_reason == "ambiguous"
                        else "absent_information"
                        if candidate.null_reason == "absent"
                        else "legitimate_unknown"
                    ),
                )
                return
            if not self.policy["skills"].extraction_allowed:
                raise CandidateRejected("resolved_skills_are_not_supplemented")
            value = validate_skill(candidate, self.sources)
            evidence = make_evidence(
                field="skills",
                value=value,
                candidate=candidate,
                sources=self.sources,
                source_platform=self.normalized.source,
                rule="literal_required_skill_v1",
            )
            diag.normalized_values, diag.evidence = {"skills": value}, [evidence]
            diag.transformation_rules = [
                "literal_required_skill_v1",
                "trim_casefold_deduplicate_v1",
            ]
            self.pending.setdefault("skills", []).append((value, evidence, diag))
        except (ValidationError, CandidateRejected) as error:
            diag.reason_code = (
                str(error)
                if isinstance(error, CandidateRejected)
                else "invalid_candidate_schema"
            )

    def salary(self, raw):
        diag = self.diagnostic("compensation", raw, fields=list(SALARY_FIELDS))
        try:
            candidate = RawCompensationCandidate.model_validate(raw)
            if any(
                self.policy[field].category == "conflict_blocked"
                for field in SALARY_FIELDS
            ):
                raise CandidateRejected("source_metadata_requires_review")
            values, rules = validate_compensation(candidate, self.sources)
            evidence = [
                make_evidence(
                    field=field,
                    value=value,
                    candidate=candidate,
                    sources=self.sources,
                    source_platform=self.normalized.source,
                    rule=";".join(rules),
                )
                for field, value in values.items()
                if value is not None
            ]
            diag.normalized_values, diag.evidence, diag.transformation_rules = (
                values,
                evidence,
                rules,
            )
            self.salaries.append((values, evidence, diag))
        except (ValidationError, CandidateRejected) as error:
            diag.reason_code = (
                str(error)
                if isinstance(error, CandidateRejected)
                else "invalid_candidate_schema"
            )

    def finish(self):
        for field, candidates in self.pending.items():
            if field == "skills":
                skills, spellings = [], {}
                for value, evidence, diag in candidates:
                    chosen = spellings.get(value.casefold())
                    if chosen is None:
                        chosen = value
                        spellings[value.casefold()] = value
                        skills.append(value)
                    evidence = evidence.model_copy(update={"normalized_value": chosen})
                    diag.evidence = [evidence]
                    diag.normalized_values = {"skills": chosen}
                    diag.status, diag.reason_code = "accepted", "validated_skill"
                    self.evidence.append(evidence)
                self.values[field] = skills
                self.accepted_fields.add(field)
                continue
            first, first_evidence, _ = candidates[0]
            if any(value != first for value, _, _ in candidates[1:]):
                for value, evidence, diag in candidates:
                    diag.status, diag.reason_code = (
                        "conflict",
                        "contradictory_candidates",
                    )
                    self.flag(
                        field,
                        "contradictory_candidates",
                        value,
                        [evidence],
                        existing_value=first,
                        existing_evidence=[first_evidence],
                    )
                continue
            self.values[field] = first
            self.accepted_fields.add(field)
            for _, evidence, diag in candidates:
                self.evidence.append(evidence)
                diag.status, diag.reason_code = "accepted", "validated_field"
        if self.salaries:
            first_values, first_evidence, _ = self.salaries[0]
            incompatible = any(
                values != first_values for values, _, _ in self.salaries[1:]
            )
            protected_conflict = any(
                self.values[field] is not None
                and values[field] is not None
                and self.values[field] != values[field]
                for values, _, _ in self.salaries
                for field in SALARY_FIELDS
            )
            if incompatible or protected_conflict:
                for values, evidence, diag in self.salaries:
                    diag.status, diag.reason_code = (
                        "conflict",
                        "conflicting_compensation_candidates",
                    )
                    for field in SALARY_FIELDS:
                        self.flag(
                            field,
                            "conflicting_compensation_candidates",
                            values[field],
                            [item for item in evidence if item.field_name == field],
                            existing_value=(
                                None if protected_conflict else first_values[field]
                            ),
                            existing_evidence=(
                                None
                                if protected_conflict
                                else [
                                    item
                                    for item in first_evidence
                                    if item.field_name == field
                                ]
                            ),
                        )
            else:
                protected_bounds = any(
                    self.values[field] is not None for field in SALARY_FIELDS[:2]
                )
                for values, evidence, diag in self.salaries:
                    diag.status, diag.reason_code = (
                        "accepted",
                        "corroborates_protected_compensation"
                        if protected_bounds
                        else "validated_compensation",
                    )
                    # Never complete one source's range with another source's bound.
                    if protected_bounds:
                        continue
                    for field, value in values.items():
                        if self.values[field] is None and value is not None:
                            self.values[field] = value
                            self.accepted_fields.add(field)
                    self.evidence.extend(
                        item
                        for item in evidence
                        if self.values[item.field_name] == item.normalized_value
                    )
        for diag in self.diagnostics:
            if diag.status == "rejected":
                self.warnings.append(
                    NormalizationWarning(
                        code="candidate_rejected",
                        fields=diag.fields,
                        message=f"Candidate rejected: {diag.reason_code}",
                    )
                )


def extract_job(
    job: JobExtractionInput | dict[str, Any],
    *,
    ollama_call: Callable[[dict], dict],
    model_name: str,
) -> JobExtractionResult:
    """Preview one job, using at most one bounded generation and no persistence."""
    started = perf_counter()
    validated = JobExtractionInput.model_validate(job)
    normalization_started = perf_counter()
    normalized = normalize_job(deepcopy(job))
    sources = prepare_sources(validated)
    normalized, deterministic_flags, unverified, jd_fields = apply_jd_rules(
        normalized, sources
    )
    normalization_seconds = perf_counter() - normalization_started
    eligibility = classify_fields(normalized)
    eligible = {item.field_name for item in eligibility if item.extraction_allowed}
    checks = [item.field_name for item in eligibility if item.contradiction_check]
    tasks = build_tasks(
        [item.field_name for item in eligibility if item.extraction_allowed], checks
    )
    registry = build_registry(sources, tasks)
    merge = _Merge(normalized, sources, eligibility)
    merge.flags.extend(flag for flag in deterministic_flags if flag not in merge.flags)
    diagnostics = ExtractionDiagnostics(
        source_paths=list(sources),
        source_registry=[item.model_dump(mode="json") for item in registry],
        deterministic_jd_fields=jd_fields,
    )
    provider = OllamaMetrics()
    inference_status, model_calls = "skipped", 0
    llm_seconds, validation_seconds = None, 0.0
    full_source_processed = False
    requested_count = requested_check_count = 0
    if not eligible and not checks:
        diagnostics.skip_reason = "no_meaningful_tasks"
    elif not _meaningful_sources(sources, eligible, checks):
        diagnostics.skip_reason = "no_usable_supporting_content"
    else:
        payload = build_ollama_request(
            model_name=model_name,
            sources=sources,
            resolved_fields={
                key: value for key, value in merge.values.items() if value is not None
            },
            eligible_fields=[
                item.field_name for item in eligibility if item.extraction_allowed
            ],
            check_fields=checks,
            blocked_fields=[
                item.field_name
                for item in eligibility
                if item.category in {"conflict_blocked", "excluded"}
            ],
        )
        diagnostics.estimated_context_tokens = estimate_context_tokens(payload)
        diagnostics.context_estimate_breakdown = context_estimate_breakdown(
            payload, registry
        )
        diagnostics.context_estimation_policy = (
            "ascii_chars_div3_nonascii_utf8_bytes_plus_schema_output_and_framing"
        )
        if diagnostics.estimated_context_tokens > NUM_CTX:
            diagnostics.skip_reason = "context_budget_exceeded"
        else:
            requested_count, requested_check_count = len(eligible), len(checks)
            model_calls = 1
            diagnostics.task_accounting = TaskAccounting(
                status="incomplete",
                requested=[task.task_id for task in tasks],
                missing=[task.task_id for task in tasks],
            )
            request_started = perf_counter()
            try:
                data = ollama_call(payload)
            except Exception as error:  # noqa: BLE001 -- injected transport owns its exception classes
                inference_status = "failed"
                name = type(error).__name__.casefold()
                status_code = getattr(error, "status_code", None)
                diagnostics.failure_reason = (
                    "inference_timeout"
                    if "timeout" in name or status_code == 504
                    else "inference_connection_failure"
                    if "connect" in name or status_code == 503
                    else "inference_transport_failure"
                )
            else:
                if not isinstance(data, dict):
                    inference_status, diagnostics.failure_reason = (
                        "parse_failed",
                        "invalid_provider_envelope",
                    )
                else:
                    provider = _provider_metrics(data)
                    incomplete = (
                        provider.done is not True
                        or provider.done_reason not in {None, "stop", "eos"}
                    )
                    diagnostics.provider_completed = not incomplete
                    validation_started = perf_counter()
                    try:
                        content = data.get("response")
                        if not isinstance(content, str) or not content.strip():
                            raise ValueError("missing_response_content")
                        if len(content) > 65536:
                            raise ValueError("response_size_limit_exceeded")
                        duplicate_keys = []
                        content = json.loads(
                            content,
                            object_pairs_hook=lambda pairs: _tracked_object(
                                pairs, duplicate_keys
                            ),
                            parse_constant=lambda value: (_ for _ in ()).throw(
                                ValueError("non_finite_json")
                            ),
                        )
                        if any(
                            key not in {task.task_id for task in tasks}
                            for key in duplicate_keys
                        ):
                            raise ValueError("duplicate_response_key")
                        process_tasks(
                            content,
                            tasks=tasks,
                            registry=registry,
                            merge=merge,
                            diagnostics=diagnostics,
                            signals=unverified,
                            duplicate_keys=duplicate_keys,
                        )
                    except (ValueError, TypeError, ValidationError, RecursionError):
                        diagnostics.response_schema_status = "invalid"
                        inference_status = (
                            "incomplete" if incomplete else "parse_failed"
                        )
                        diagnostics.failure_reason = (
                            "incomplete_generation"
                            if incomplete
                            else "invalid_response_json_or_schema"
                        )
                    else:
                        merge.finish()
                        inference_status = "incomplete" if incomplete else "succeeded"
                        if incomplete:
                            diagnostics.failure_reason = "incomplete_generation"
                        elif diagnostics.task_accounting.status == "complete":
                            full_source_processed = True
                    validation_seconds = perf_counter() - validation_started
            finally:
                llm_seconds = perf_counter() - request_started - validation_seconds
    diagnostics.candidates = merge.diagnostics
    if model_calls and diagnostics.response_schema_status in {"valid", "partial"}:
        diagnostics.empty_useful_content = not bool(
            merge.accepted_fields or merge.flags
        )
    if (
        diagnostics.task_accounting.status == "incomplete"
        and inference_status == "succeeded"
    ):
        diagnostics.failure_reason = "incomplete_task_accounting"
    if (
        diagnostics.failure_reason
        or diagnostics.skip_reason == "context_budget_exceeded"
    ):
        merge.warnings.append(
            NormalizationWarning(
                code=diagnostics.failure_reason or "context_budget_exceeded",
                fields=[],
                message="Full extraction did not complete; valid deterministic and validated partial fields are retained.",
            )
        )
    fields = StructuredJobFields(**merge.values)
    unresolved = [name for name, value in fields.model_dump().items() if value is None]
    review_fields = list(dict.fromkeys(flag.field_name for flag in merge.flags))
    if (
        inference_status in {"failed", "parse_failed", "incomplete"}
        or diagnostics.skip_reason == "context_budget_exceeded"
        or diagnostics.task_accounting.status == "incomplete"
    ):
        extraction_status = "incomplete"
    elif merge.flags:
        extraction_status = "requires_review"
    elif unverified:
        extraction_status = "requires_inspection"
    elif inference_status == "skipped":
        extraction_status = "inference_skipped"
    elif merge.warnings:
        extraction_status = "succeeded_with_warnings"
    else:
        extraction_status = "succeeded"
    generation_seconds = (
        provider.eval_duration_ns / 1e9
        if provider.eval_duration_ns is not None
        else None
    )
    return JobExtractionResult(
        job_id=normalized.job_id,
        source=normalized.source,
        fields=fields,
        evidence=merge.evidence,
        warnings=merge.warnings,
        source_metadata=deepcopy(normalized.source_metadata),
        unresolved_fields=unresolved,
        review_flags=merge.flags,
        review_required=bool(merge.flags),
        extraction_status=extraction_status,
        inference_status=inference_status,
        unverified_conflicts=unverified,
        manual_inspection_required=bool(merge.flags or unverified),
        metadata=ExtractionMetadata(
            schema_version=JOB_INTELLIGENCE_SCHEMA_VERSION,
            prompt_version=EXTRACTION_PROMPT_VERSION,
            normalization_version=NORMALIZATION_VERSION,
            evidence_policy_version=EVIDENCE_POLICY_VERSION,
            model_name=model_name,
            eligibility=eligibility,
            full_source_processed=full_source_processed,
        ),
        metrics=ExtractionMetrics(
            total_wall_seconds=perf_counter() - started,
            normalization_seconds=normalization_seconds,
            llm_request_seconds=max(0.0, llm_seconds)
            if llm_seconds is not None
            else None,
            validation_seconds=validation_seconds,
            model_calls=model_calls,
            model_name=model_name,
            prompt_tokens=provider.prompt_eval_count,
            output_tokens=provider.eval_count,
            model_load_seconds=provider.load_duration_ns / 1e9
            if provider.load_duration_ns is not None
            else None,
            generation_seconds=generation_seconds,
            tokens_per_second=provider.eval_count / generation_seconds
            if provider.eval_count is not None and generation_seconds
            else None,
            deterministic_field_count=sum(
                value is not None for value in normalized.fields.model_dump().values()
            ),
            eligible_unresolved_field_count=len(eligible),
            requested_llm_field_count=requested_count,
            requested_contradiction_field_count=requested_check_count,
            accepted_llm_field_count=len(merge.accepted_fields),
            rejected_candidate_count=sum(
                item.status == "rejected" for item in merge.diagnostics
            ),
            remaining_unresolved_field_count=len(unresolved),
            conflict_count=len(merge.flags),
            fields_requiring_review=review_fields,
            provider=provider,
            requested_task_count=len(diagnostics.task_accounting.requested),
            accounted_task_count=len(diagnostics.task_accounting.dispositions)
            - len(
                set(diagnostics.task_accounting.invalid).intersection(
                    diagnostics.task_accounting.dispositions
                )
            ),
            unverified_conflict_count=len(unverified),
        ),
        diagnostics=diagnostics,
    )
