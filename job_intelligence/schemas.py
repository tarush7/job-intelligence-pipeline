"""Environment-independent normalization, raw-fact and hybrid result contracts."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

SupportedSource = Literal["linkedin", "indeed", "naukri"]
StructuredFieldName = Literal[
    "company",
    "title",
    "seniority",
    "city",
    "country",
    "remote_policy",
    "salary_min",
    "salary_max",
    "currency",
    "employment_type",
    "skills",
    "visa_sponsorship",
    "posted_date",
]
NonBlankString = Annotated[StrictStr, Field(min_length=1)]
SalaryAmount = Annotated[StrictFloat, Field(ge=0, allow_inf_nan=False)]


class JobExtractionInput(BaseModel):
    """A jobs-row projection, not a complete PostgreSQL constraint declaration.

    ID, platform, posting URLs and title identify a stored posting and are
    required. Other projected values may be null or omitted; no defaults are
    invented for business information. Additional table columns are ignored.
    Timestamp strings are parsed, but numeric timestamps and coerced text/bools
    are rejected. Raw source JSON is preserved for source-specific adapters.
    """

    model_config = ConfigDict(extra="ignore")

    id: UUID
    source: NonBlankString
    external_job_id: StrictStr | None = None
    job_url: NonBlankString
    canonical_job_url: NonBlankString
    job_title: NonBlankString
    company: StrictStr | None = None
    location: StrictStr | None = None
    job_description: StrictStr | None = None
    employment_type: StrictStr | None = None
    seniority_level: StrictStr | None = None
    posted_at: datetime | None = None
    raw_source_data: dict[StrictStr, JsonValue] | None = None
    is_active: StrictBool | None = None
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None

    @field_validator("source", "job_url", "canonical_job_url", "job_title")
    @classmethod
    def required_text_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("required identifiers and title must not be blank")
        return value

    @field_validator("id", mode="before")
    @classmethod
    def uuid_input_type(cls, value):
        if not isinstance(value, (str, UUID)):
            raise ValueError("id must be a UUID or UUID string")  # noqa: TRY004 -- Pydantic v2 does not wrap TypeError
        return value

    @field_validator("posted_at", "first_seen_at", "last_seen_at", mode="before")
    @classmethod
    def timestamp_input_type(cls, value):
        if value is not None and not isinstance(value, (str, datetime)):
            raise ValueError("timestamps must be datetimes, ISO strings, or null")
        return value

    @field_validator("raw_source_data")
    @classmethod
    def source_metadata_is_finite_json(cls, value):
        if value is not None:
            json.dumps(value, allow_nan=False)
        return value


class StructuredJobFields(BaseModel):
    """Canonical fields only. Salary amounts, if known, are annual base pay.

    The schema validates numeric bounds; the adapter establishes annual units
    and employer provenance before constructing this model. It never infers
    a currency, a work schedule, or an annualization factor.
    """

    model_config = ConfigDict(extra="forbid")

    company: NonBlankString | None = None
    title: NonBlankString | None = None
    seniority: Literal["junior", "mid", "senior", "staff_plus"] | None = None
    city: NonBlankString | None = None
    country: NonBlankString | None = None
    remote_policy: Literal["onsite", "hybrid", "remote"] | None = None
    salary_min: SalaryAmount | None = None
    salary_max: SalaryAmount | None = None
    currency: Annotated[StrictStr, Field(pattern=r"^[A-Z]{3}$")] | None = None
    employment_type: Literal["full_time", "contract", "intern"] | None = None
    skills: list[NonBlankString] | None = None
    visa_sponsorship: StrictBool | None = None
    posted_date: date | None = None

    @field_validator("company", "title", "city", "country")
    @classmethod
    def output_text_is_not_blank(cls, value):
        if value is not None and not value.strip():
            raise ValueError("blank values must be null")
        return value

    @field_validator("skills")
    @classmethod
    def skills_are_nonempty(cls, value):
        if value is not None and (not value or any(not item.strip() for item in value)):
            raise ValueError("unknown skills must be null, not empty strings or a list")
        return value

    @field_validator("posted_date", mode="before")
    @classmethod
    def date_input_type(cls, value):
        if value is not None and (
            isinstance(value, datetime) or not isinstance(value, (str, date))
        ):
            raise ValueError("posted_date must be a date, ISO date string, or null")
        return value

    @model_validator(mode="after")
    def salary_bounds_are_ordered(self):
        if (
            self.salary_min is not None
            and self.salary_max is not None
            and self.salary_max < self.salary_min
        ):
            raise ValueError("salary_max must be greater than or equal to salary_min")
        return self


class FieldEvidence(BaseModel):
    """One supporting source value, including one skill when field_name=skills.

    source_path uses jobs-row paths such as raw_source_data.payload.company.
    normalized_value is the exact value supported in the output (or one skill).
    JD evidence stores the original source string plus its exact substring and
    half-open Python character offsets. Offsets never refer to normalized text.
    """

    model_config = ConfigDict(extra="forbid")

    field_name: StructuredFieldName
    extraction_method: Literal["direct", "deterministic", "llm"]
    source_platform: SupportedSource
    original_value: JsonValue
    normalized_value: JsonValue
    source_path: NonBlankString
    supporting_values: dict[StrictStr, JsonValue] = Field(default_factory=dict)
    source_snippet: NonBlankString | None = None
    start_pos: Annotated[StrictInt, Field(ge=0)] | None = None
    end_pos: Annotated[StrictInt, Field(gt=0)] | None = None
    normalization_rule: NonBlankString | None = None

    @model_validator(mode="after")
    def evidence_is_coherent(self):
        if not self.source_path.strip():
            raise ValueError("source_path must not be blank")
        if self.original_value is None or self.normalized_value is None:
            raise ValueError("null values cannot support a resolved field")
        if isinstance(self.original_value, str) and not self.original_value.strip():
            raise ValueError("empty source text is not evidence")
        if isinstance(self.original_value, (list, dict)) and not self.original_value:
            raise ValueError("empty source containers are not evidence")
        if (
            self.extraction_method == "deterministic"
            and not (self.normalization_rule or "").strip()
        ):
            raise ValueError("deterministic evidence requires a normalization rule")
        if self.extraction_method == "direct" and (
            type(self.original_value) is not type(self.normalized_value)
            or self.original_value != self.normalized_value
        ):
            raise ValueError("direct evidence must preserve its original value")
        has_offsets = self.start_pos is not None or self.end_pos is not None
        jd_path = self.source_path in {
            "job_description",
            "raw_source_data.payload.descriptionText",
        }
        if jd_path and (self.source_snippet is None or not has_offsets):
            raise ValueError("JD-derived evidence needs a snippet and source offsets")
        if has_offsets or self.source_snippet is not None:
            if (
                self.start_pos is None
                or self.end_pos is None
                or self.source_snippet is None
            ):
                raise ValueError(
                    "snippet and both character offsets must be provided together"
                )
            if (
                not isinstance(self.original_value, str)
                or self.end_pos <= self.start_pos
                or self.end_pos > len(self.original_value)
            ):
                raise ValueError("invalid source evidence interval")
            if (
                self.original_value[self.start_pos : self.end_pos]
                != self.source_snippet
            ):
                raise ValueError(
                    "snippet must match the exact original source interval"
                )
        return self


class NormalizationWarning(BaseModel):
    """Inspectable ambiguity/conflict, separate from supporting field evidence."""

    model_config = ConfigDict(extra="forbid")

    code: NonBlankString
    fields: list[StructuredFieldName] = Field(default_factory=list)
    message: NonBlankString
    source_values: dict[StrictStr, JsonValue] = Field(default_factory=dict)


class JobNormalizationResult(BaseModel):
    """A pure normalization result; source_metadata retains source-only facts."""

    model_config = ConfigDict(extra="forbid")

    job_id: UUID
    source: SupportedSource
    fields: StructuredJobFields
    evidence: list[FieldEvidence] = Field(default_factory=list)
    warnings: list[NormalizationWarning] = Field(default_factory=list)
    unresolved_fields: list[StructuredFieldName]
    source_metadata: dict[StrictStr, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def resolved_fields_have_evidence(self):
        values = self.fields.model_dump(mode="json")
        expected_unresolved = {name for name, value in values.items() if value is None}
        if (
            len(set(self.unresolved_fields)) != len(self.unresolved_fields)
            or set(self.unresolved_fields) != expected_unresolved
        ):
            raise ValueError(
                "unresolved_fields must list exactly the null structured fields"
            )
        grouped: dict[str, list[FieldEvidence]] = {}
        for item in self.evidence:
            if item.source_platform != self.source:
                raise ValueError("evidence platform does not match result source")
            if values[item.field_name] is None:
                raise ValueError("unresolved fields must not have evidence")
            grouped.setdefault(item.field_name, []).append(item)
        for name, value in values.items():
            if value is None:
                continue
            items = grouped.get(name, [])
            if not items:
                raise ValueError(f"missing provenance for {name}")
            if name == "skills":
                supported = [item.normalized_value for item in items]
                if any(
                    not isinstance(item, str) or item not in value for item in supported
                ) or set(supported) != set(value):
                    raise ValueError("each skill needs matching supporting evidence")
            elif any(
                isinstance(item.normalized_value, bool) != isinstance(value, bool)
                or item.normalized_value != value
                for item in items
            ):
                raise ValueError(f"evidence does not support output value for {name}")
        return self


class RawFieldCandidate(BaseModel):
    """A proposed fact, never permission to replace a resolved field."""

    model_config = ConfigDict(extra="forbid")

    field_name: StructuredFieldName
    value: StrictStr | StrictBool | None
    null_reason: Literal["absent", "ambiguous"] | None = None
    source_path: NonBlankString | None = None
    evidence: NonBlankString | None = None
    start_pos: Annotated[StrictInt, Field(ge=0)] | None = None
    end_pos: Annotated[StrictInt, Field(gt=0)] | None = None


class RawSkillCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: StrictStr | None
    null_reason: Literal["absent", "ambiguous"] | None = None
    source_path: NonBlankString | None = None
    evidence: NonBlankString | None = None
    start_pos: Annotated[StrictInt, Field(ge=0)] | None = None
    end_pos: Annotated[StrictInt, Field(gt=0)] | None = None


class RawCompensationCandidate(BaseModel):
    """Amounts are as stated, before any Python unit/annual transformations."""

    model_config = ConfigDict(extra="forbid")

    amount_min: StrictFloat | StrictStr | None
    amount_max: StrictFloat | StrictStr | None
    currency: StrictStr | None
    pay_period: Literal["annual", "monthly", "weekly", "daily", "hourly", "unknown"]
    compensation_type: Literal[
        "base_salary",
        "total_compensation",
        "bonus",
        "allowance",
        "benefits",
        "other",
        "unknown",
    ]
    source_path: NonBlankString
    evidence: NonBlankString
    start_pos: Annotated[StrictInt, Field(ge=0)] | None = None
    end_pos: Annotated[StrictInt, Field(gt=0)] | None = None


class RawExtractionResponse(BaseModel):
    """Legacy Phase B schema for compatibility/replay, not the live wire format.

    Phase B.1 uses contracts.response_schema() for the actual requested tasks.
    """

    model_config = ConfigDict(extra="forbid")

    fields: list[RawFieldCandidate] | None = Field(max_length=24)
    skills: list[RawSkillCandidate] | None = Field(max_length=32)
    compensation: list[RawCompensationCandidate] | None = Field(max_length=12)
    contradictions: list[RawFieldCandidate] | None = Field(max_length=24)


class RawExtractionEnvelope(BaseModel):
    """Validate framing first, then validate each typed candidate independently.

    Missing top-level sections and extra sections are malformed responses.
    Invalid list entries can be rejected without losing other valid facts.
    """

    model_config = ConfigDict(extra="forbid")

    fields: list[JsonValue] | None = Field(max_length=24)
    skills: list[JsonValue] | None = Field(max_length=32)
    compensation: list[JsonValue] | None = Field(max_length=12)
    contradictions: list[JsonValue] | None = Field(max_length=24)


class FieldEligibility(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_name: StructuredFieldName
    category: Literal[
        "resolved", "missing", "ambiguous", "conflict_blocked", "excluded"
    ]
    extraction_allowed: bool
    contradiction_check: bool
    reason: str


class ReviewFlag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_name: StructuredFieldName
    reason_code: str
    existing_value: JsonValue = None
    existing_evidence: list[FieldEvidence] = Field(default_factory=list)
    conflicting_value: JsonValue = None
    conflicting_evidence: list[FieldEvidence] = Field(default_factory=list)
    source_values: dict[str, JsonValue] = Field(default_factory=dict)
    review_required: Literal[True] = True


class CandidateDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["field", "skill", "compensation", "contradiction"]
    candidate: JsonValue
    status: Literal["accepted", "rejected", "unknown", "conflict"]
    fields: list[StructuredFieldName] = Field(default_factory=list)
    reason_code: str
    normalized_values: dict[str, JsonValue] = Field(default_factory=dict)
    transformation_rules: list[str] = Field(default_factory=list)
    evidence: list[FieldEvidence] = Field(default_factory=list)
    task_id: str | None = None


class TaskAccounting(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["not_requested", "complete", "incomplete"] = "not_requested"
    requested: list[str] = Field(default_factory=list)
    received: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    duplicate: list[str] = Field(default_factory=list)
    unsolicited: list[str] = Field(default_factory=list)
    invalid: list[str] = Field(default_factory=list)
    dispositions: dict[
        str, Literal["found", "not_found", "ambiguous", "conflicting"]
    ] = Field(default_factory=dict)


class UnverifiedConflictSignal(BaseModel):
    """Manual inspection only; deliberately has no validated FieldEvidence."""

    model_config = ConfigDict(extra="forbid")
    task_id: str
    fields: list[StructuredFieldName]
    candidate: JsonValue
    reason_code: str
    verified: Literal[False] = False
    inspection_required: Literal[True] = True


class ExtractionDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[CandidateDiagnostic] = Field(default_factory=list)
    skip_reason: str | None = None
    failure_reason: str | None = None
    source_paths: list[str] = Field(default_factory=list)
    estimated_context_tokens: int | None = None
    context_estimate_breakdown: dict[str, JsonValue] = Field(default_factory=dict)
    deterministic_jd_fields: list[StructuredFieldName] = Field(default_factory=list)
    context_estimation_policy: str = "utf8_bytes_upper_bound_plus_schema_and_output"
    source_registry: list[dict[str, JsonValue]] = Field(default_factory=list)
    provider_completed: bool | None = None
    response_schema_status: Literal["not_attempted", "valid", "partial", "invalid"] = (
        "not_attempted"
    )
    task_accounting: TaskAccounting = Field(default_factory=TaskAccounting)
    all_unknown_response: bool | None = None
    empty_useful_content: bool | None = None


class OllamaMetrics(BaseModel):
    """Provider durations are nanoseconds; unavailable provider facts stay null."""

    model_config = ConfigDict(extra="forbid")

    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration_ns: int | None = None
    eval_count: int | None = None
    eval_duration_ns: int | None = None
    done: bool | None = None
    done_reason: str | None = None


class ExtractionMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_wall_seconds: float
    normalization_seconds: float
    llm_request_seconds: float | None
    validation_seconds: float
    model_calls: int
    model_name: str
    prompt_tokens: int | None
    output_tokens: int | None
    model_load_seconds: float | None
    generation_seconds: float | None
    tokens_per_second: float | None
    deterministic_field_count: int
    eligible_unresolved_field_count: int
    requested_llm_field_count: int
    requested_contradiction_field_count: int
    accepted_llm_field_count: int
    rejected_candidate_count: int
    remaining_unresolved_field_count: int
    conflict_count: int
    fields_requiring_review: list[StructuredFieldName]
    provider: OllamaMetrics
    requested_task_count: int = 0
    accounted_task_count: int = 0
    unverified_conflict_count: int = 0


class ExtractionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    prompt_version: str
    normalization_version: str
    evidence_policy_version: str
    model_name: str
    eligibility: list[FieldEligibility]
    full_source_processed: bool


class JobExtractionResult(JobNormalizationResult):
    """Final evidence supports only final fields; contrary evidence lives in flags.

    Status precedence: failed/partial inference => incomplete, then review,
    then skipped inference, then warnings, then success. Unknown facts are not
    failures. Deterministic values survive every inference failure.
    """

    review_flags: list[ReviewFlag]
    review_required: bool
    extraction_status: Literal[
        "succeeded",
        "succeeded_with_warnings",
        "requires_review",
        "incomplete",
        "inference_skipped",
        "requires_inspection",
    ]
    inference_status: Literal[
        "succeeded",
        "skipped",
        "failed",
        "parse_failed",
        "incomplete",
    ]
    metadata: ExtractionMetadata
    metrics: ExtractionMetrics
    diagnostics: ExtractionDiagnostics
    unverified_conflicts: list[UnverifiedConflictSignal] = Field(default_factory=list)
    manual_inspection_required: bool = False

    @model_validator(mode="after")
    def review_state_is_coherent(self):
        if self.review_required != bool(self.review_flags):
            raise ValueError("review_required must reflect review_flags")
        if self.manual_inspection_required != bool(
            self.review_flags or self.unverified_conflicts
        ):
            raise ValueError(
                "manual_inspection_required must reflect verified or unverified signals"
            )
        if (
            self.diagnostics.task_accounting.status == "incomplete"
            and self.extraction_status != "incomplete"
        ):
            raise ValueError("incomplete task accounting cannot be complete extraction")
        if (
            self.inference_status in {"failed", "parse_failed", "incomplete"}
            and self.extraction_status != "incomplete"
        ):
            raise ValueError("failed or partial inference must be incomplete")
        return self
