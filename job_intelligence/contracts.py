"""Requested-task schema and deterministic source registry for hybrid extraction."""

from __future__ import annotations

from html import unescape
from typing import Literal, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictFloat,
    StrictStr,
    ValidationError,
    create_model,
)

from job_intelligence.schemas import (
    ExtractionDiagnostics,
    StructuredFieldName,
    TaskAccounting,
    UnverifiedConflictSignal,
)
from job_intelligence.validators import (
    _EMPLOYMENT,
    _METADATA_FIELDS,
    _SENIORITY,
    _WORK_MODE,
    DESCRIPTION_PATHS,
    SALARY_FIELDS,
    _key,
)


class SourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_id: str
    source_path: str
    text: str
    fields: list[StructuredFieldName]
    equivalent_sources: dict[str, str] = Field(default_factory=dict)


class ExtractionTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str
    mode: Literal["extract", "check"]
    kind: Literal["scalar", "skills", "compensation"]
    fields: list[StructuredFieldName]


class ScalarFact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: StrictStr | StrictBool
    source_id: StrictStr = Field(min_length=1)
    quote: StrictStr = Field(min_length=1)


class SkillFact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: StrictStr = Field(min_length=1)
    source_id: StrictStr = Field(min_length=1)
    quote: StrictStr = Field(min_length=1)


class CompensationFact(BaseModel):
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
    source_id: StrictStr = Field(min_length=1)
    quote: StrictStr = Field(min_length=1)


class UnsupportedTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    disposition: Literal["not_found", "ambiguous"]


class ScalarTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    disposition: Literal["found", "conflicting"]
    candidates: list[ScalarFact] = Field(min_length=1, max_length=12)


class SkillsTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    disposition: Literal["found", "conflicting"]
    candidates: list[SkillFact] = Field(min_length=1, max_length=32)


class CompensationTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    disposition: Literal["found", "conflicting"]
    candidates: list[CompensationFact] = Field(min_length=1, max_length=12)


def build_tasks(
    eligible_fields: list[str], check_fields: list[str]
) -> list[ExtractionTask]:
    tasks = []
    for mode, names in (("extract", eligible_fields), ("check", check_fields)):
        salary_added = False
        for name in names:
            if name in SALARY_FIELDS:
                if salary_added:
                    continue
                target, kind, fields = (
                    "compensation",
                    "compensation",
                    list(SALARY_FIELDS),
                )
                salary_added = True
            else:
                target, kind, fields = (
                    name,
                    "skills" if name == "skills" else "scalar",
                    [name],
                )
            tasks.append(
                ExtractionTask(
                    task_id=("check_" if mode == "check" else "") + target,
                    mode=mode,
                    kind=kind,
                    fields=fields,
                )
            )
    return tasks


def build_registry(
    sources: dict[str, str], tasks: list[ExtractionTask]
) -> list[SourceRecord]:
    requested = {field for task in tasks for field in task.fields}
    records = []
    seen = {}
    for path, text in sources.items():
        if path in DESCRIPTION_PATHS:
            fields = list(get_args(StructuredFieldName))
        elif ".skills[" in path:
            fields = ["skills"]
        elif path == "raw_source_data.payload.salary":
            fields = list(SALARY_FIELDS)
        else:
            fields = [
                field for field, paths in _METADATA_FIELDS.items() if path in paths
            ]
        fields = [field for field in fields if field in requested]
        # Compare representations only. The selected text and all original
        # paths/texts remain untouched for quote grounding and provenance.
        comparison = (
            " ".join(unescape(text).split()) if path in DESCRIPTION_PATHS else text
        )
        key = (path in DESCRIPTION_PATHS, comparison, tuple(fields))
        if not text.strip() or not fields:
            continue
        if key in seen:
            records[seen[key]].equivalent_sources[path] = text
            continue
        seen[key] = len(records)
        records.append(
            SourceRecord(
                source_id=f"S{len(records) + 1}",
                source_path=path,
                text=text,
                fields=fields,
            )
        )
    return records


def response_schema(tasks: list[ExtractionTask], registry: list[SourceRecord]) -> dict:
    models = {
        "scalar": ScalarTask,
        "skills": SkillsTask,
        "compensation": CompensationTask,
    }
    task_model = create_model(
        "RequestedTasks",
        __config__=ConfigDict(extra="forbid"),
        **{task.task_id: (models[task.kind] | UnsupportedTask, ...) for task in tasks},
    )
    root = create_model(
        "JobFactResponse",
        __config__=ConfigDict(extra="forbid"),
        tasks=(task_model, ...),
    )
    schema = root.model_json_schema()
    for definition in schema.get("$defs", {}).values():
        source_property = definition.get("properties", {}).get("source_id")
        if source_property is not None:
            source_property["enum"] = [item.source_id for item in registry]
    return schema


class TaskEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tasks: dict[StrictStr, JsonValue]


class SupportedEnvelope(BaseModel):
    """Frame a task, then validate candidates independently."""

    model_config = ConfigDict(extra="forbid")
    disposition: Literal["found", "conflicting"]
    candidates: list[JsonValue] = Field(min_length=1, max_length=32)


def normalized_raw_value(field: str, value):
    mapping = {
        "remote_policy": _WORK_MODE,
        "employment_type": _EMPLOYMENT,
        "seniority": _SENIORITY,
    }.get(field)
    return mapping.get(_key(value), value) if mapping is not None else value


def _material(task, raw, protected):
    if not isinstance(raw, dict):
        return False
    if task.kind == "compensation":
        return raw.get("amount_min") is not None or raw.get("amount_max") is not None
    value = normalized_raw_value(task.fields[0], raw.get("value"))
    existing = protected.get(task.fields[0])
    if value is None:
        return False
    if isinstance(value, str) and isinstance(existing, str):
        return value.casefold() != existing.casefold()
    return value != existing


def unverified_signal(task, raw, reason, merge, signals):
    if task.mode == "check" and _material(task, raw, merge.values):
        signal = UnverifiedConflictSignal(
            task_id=task.task_id, fields=task.fields, candidate=raw, reason_code=reason
        )
        if signal not in signals:
            signals.append(signal)


def process_tasks(
    content,
    *,
    tasks,
    registry,
    merge,
    diagnostics: ExtractionDiagnostics,
    signals,
    duplicate_keys=(),
):
    """No inference or calculations here: enforce framing/accounting and ground IDs."""
    expected = {task.task_id: task for task in tasks}
    accounting = TaskAccounting(status="incomplete", requested=list(expected))
    diagnostics.task_accounting = accounting
    try:
        envelope = TaskEnvelope.model_validate(content)
    except ValidationError:
        accounting.missing = list(expected)
        diagnostics.response_schema_status = "invalid"
        # Preserve visibility of the captured old malformed contradiction shape,
        # without treating it as the current contract or as verified evidence.
        if isinstance(content, dict) and isinstance(
            content.get("contradictions"), list
        ):
            for raw in content["contradictions"]:
                if isinstance(raw, dict):
                    task = expected.get("check_" + str(raw.get("field_name")))
                    if task is not None:
                        unverified_signal(
                            task, raw, "unsupported_response_contract", merge, signals
                        )
        raise ValueError("invalid_task_envelope") from None
    received = envelope.tasks
    accounting.received = list(received)
    accounting.missing = [key for key in expected if key not in received]
    accounting.duplicate = list(
        dict.fromkeys(key for key in duplicate_keys if key in expected)
    )
    accounting.unsolicited = [key for key in received if key not in expected]
    registry_by_id = {source.source_id: source for source in registry}
    models = {
        "scalar": ScalarFact,
        "skills": SkillFact,
        "compensation": CompensationFact,
    }
    for task_id, raw_task in received.items():
        if task_id not in expected or task_id in accounting.duplicate:
            reason = (
                "unsolicited_task_id"
                if task_id not in expected
                else "duplicate_task_id"
            )
            candidates = (
                raw_task.get("candidates", [raw_task])
                if isinstance(raw_task, dict)
                else [raw_task]
            )
            for raw in candidates if isinstance(candidates, list) else [raw_task]:
                diag = merge.diagnostic("field", raw, reason=reason)
                diag.task_id = task_id
            continue
        task = expected[task_id]
        kind = (
            "contradiction"
            if task.mode == "check" and task.kind == "scalar"
            else "skill"
            if task.kind == "skills"
            else "compensation"
            if task.kind == "compensation"
            else "field"
        )
        try:
            if isinstance(raw_task, dict) and raw_task.get("disposition") in {
                "not_found",
                "ambiguous",
            }:
                unknown = UnsupportedTask.model_validate(raw_task)
                accounting.dispositions[task_id] = unknown.disposition
                continue
            supported = SupportedEnvelope.model_validate(raw_task)
            if supported.disposition == "conflicting" and len(supported.candidates) < 2:
                raise ValueError("conflicting_task_needs_multiple_candidates")
            accounting.dispositions[task_id] = supported.disposition
        except (ValidationError, ValueError):
            accounting.invalid.append(task_id)
            candidates = (
                raw_task.get("candidates", []) if isinstance(raw_task, dict) else []
            )
            for raw in candidates if isinstance(candidates, list) else []:
                unverified_signal(task, raw, "invalid_task_schema", merge, signals)
            diag = merge.diagnostic(
                kind, raw_task, fields=task.fields, reason="invalid_task_schema"
            )
            diag.task_id = task_id
            continue
        for raw in supported.candidates:
            transformed = False
            try:
                fact = models[task.kind].model_validate(raw)
                source = registry_by_id.get(fact.source_id)
                if source is None:
                    raise ValueError("unsupported_source_id")
                if not set(task.fields).intersection(source.fields):
                    raise ValueError("source_id_not_relevant_to_field")
                grounded = fact.model_dump(exclude={"source_id", "quote"})
                grounded.update(source_path=source.source_path, evidence=fact.quote)
                if task.kind == "scalar":
                    grounded["field_name"] = task.fields[0]
                    canonical = normalized_raw_value(task.fields[0], fact.value)
                    transformed = canonical != fact.value
                    grounded["value"] = canonical
                    merge.field(grounded, contradiction=task.mode == "check")
                elif task.kind == "skills":
                    merge.skill(grounded)
                else:
                    merge.salary(grounded)
                diag = merge.diagnostics[-1]
                # Diagnostics retain the model's raw ID/quote/value, not a repaired
                # model response. Grounded FieldEvidence contains actual paths.
                diag.candidate, diag.task_id = raw, task_id
                if transformed:
                    diag.transformation_rules.append("explicit_enum_alias_v1")
                if (
                    diag.status == "rejected"
                    and diag.reason_code != "invalid_candidate_schema"
                ):
                    unverified_signal(task, raw, diag.reason_code, merge, signals)
            except (ValidationError, ValueError) as error:
                reason = (
                    "invalid_candidate_schema"
                    if isinstance(error, ValidationError)
                    else str(error)
                )
                diag = merge.diagnostic(kind, raw, fields=task.fields, reason=reason)
                diag.task_id = task_id
                if isinstance(error, ValidationError):
                    accounting.invalid.append(task_id)
                unverified_signal(task, raw, reason, merge, signals)
    accounting.invalid = list(dict.fromkeys(accounting.invalid))
    accounting.status = (
        "incomplete"
        if any(
            (
                accounting.missing,
                accounting.duplicate,
                accounting.unsolicited,
                accounting.invalid,
            )
        )
        else "complete"
    )
    diagnostics.response_schema_status = (
        "valid" if accounting.status == "complete" else "partial"
    )
    extraction_tasks = [task.task_id for task in tasks if task.mode == "extract"]
    diagnostics.all_unknown_response = (
        all(
            accounting.dispositions.get(key) in {"not_found", "ambiguous"}
            for key in extraction_tasks
        )
        if extraction_tasks and accounting.status == "complete"
        else None
    )
