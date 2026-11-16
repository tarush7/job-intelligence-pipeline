"""Small JD-only location rules and conservative identity review detection."""

from __future__ import annotations

import re
from html import unescape

from .jd_locations import identity_claims, location_claims
from .schemas import (
    FieldEvidence,
    NormalizationWarning,
    RawFieldCandidate,
    ReviewFlag,
    StructuredJobFields,
    UnverifiedConflictSignal,
)
from .validators import DESCRIPTION_PATHS, CandidateRejected, validate_scalar


def apply_jd_rules(normalized, sources):
    """Add only explicit locations; preserve protected values and all conflicts.

    About-company text is not a hiring-employer label. A discrepancy there is
    visible for inspection but never presented as a verified contradiction.
    """
    result = normalized.model_copy(deep=True)
    values = result.fields.model_dump(mode="json")
    flags, signals, additions = [], [], []
    proposals = {}
    blocked = {
        field
        for warning in result.warnings
        if warning.code in {"metadata_conflict", "unreconciled_metadata"}
        for field in warning.fields
    }
    seen = set()
    for path, source in sources.items():
        if path not in DESCRIPTION_PATHS:
            continue
        key = " ".join(unescape(source).split())
        if key in seen:
            continue
        seen.add(key)
        locations = location_claims(source)
        for claim in locations:
            if not claim.values:
                blocked.update({"city", "country"})
                result.warnings.append(
                    NormalizationWarning(
                        code="jd_location_conflict",
                        fields=["city", "country"],
                        message="Explicit JD location is ambiguous; no location winner was selected.",
                        source_values={path: source[claim.start : claim.end]},
                    )
                )
                signals.append(
                    UnverifiedConflictSignal(
                        task_id="deterministic_check_location",
                        fields=["city", "country"],
                        candidate={
                            "source_path": path,
                            "quote": source[claim.start : claim.end],
                            "start_pos": claim.start,
                            "end_pos": claim.end,
                            "rule": claim.rule,
                        },
                        reason_code="ambiguous_job_locations",
                    )
                )
        claims = [
            (field, value, claim.start, claim.end, claim.rule)
            for claim in locations
            for field, value in claim.values.items()
        ]
        claims.extend(
            ("company", value, start, end, "jd_explicit_employer_review_v1")
            for value, start, end in identity_claims(source)
        )
        for field, value, start, end, rule in claims:
            quote = source[start:end]
            candidate = RawFieldCandidate(
                field_name=field,
                value=value,
                source_path=path,
                evidence=quote,
                start_pos=start,
                end_pos=end,
            )
            try:
                value = validate_scalar(candidate, sources)
            except CandidateRejected:
                continue
            evidence = FieldEvidence(
                field_name=field,
                extraction_method="deterministic",
                source_platform=result.source,
                original_value=source,
                normalized_value=value,
                source_path=path,
                source_snippet=quote,
                start_pos=start,
                end_pos=end,
                normalization_rule=rule,
            )
            proposals.setdefault(field, []).append((value, evidence))
        protected = values.get("company")
        if not protected:
            continue
        about = re.compile(
            r"^[ \t]*(?:\*\*|__)?About(?: us| the company)?[ \t]*:?(?:\*\*|__)?[ \t]*\r?\n(?:[ \t]*\r?\n)*(?P<company>[A-Z][^\r\n.!?]{1,100}?(?:Limited|Ltd|LLC|Inc|Corporation|Corp|GmbH))[ \t]+(?:is|was)\b[^\r\n]*",
            re.MULTILINE,
        )
        for match in about.finditer(source):
            employer = match["company"].strip()
            if employer.casefold() == str(protected).casefold() or re.search(
                r"\b(?:our client|our parent|our partner|our customer)\b",
                match.group(0),
                re.IGNORECASE,
            ):
                continue
            signals.append(
                UnverifiedConflictSignal(
                    task_id="deterministic_check_company",
                    fields=["company"],
                    candidate={
                        "value": employer,
                        "protected_value": protected,
                        "source_path": path,
                        "quote": match.group(0),
                        "start_pos": match.start(),
                        "end_pos": match.end(),
                        "rule": "jd_about_company_inspection_v1",
                    },
                    reason_code="about_company_identity_requires_inspection",
                )
            )
    for field, candidates in proposals.items():
        distinct = {str(value).casefold() for value, _ in candidates}
        existing = values[field]
        if field in blocked and existing is None:
            for value, evidence in candidates:
                signals.append(
                    UnverifiedConflictSignal(
                        task_id="deterministic_check_" + field,
                        fields=[field],
                        candidate={
                            "value": value,
                            "source_path": evidence.source_path,
                            "quote": evidence.source_snippet,
                            "start_pos": evidence.start_pos,
                            "end_pos": evidence.end_pos,
                        },
                        reason_code="blocked_location_requires_inspection",
                    )
                )
            continue
        if existing is None and len(distinct) == 1 and field != "company":
            values[field] = candidates[0][0]
            additions.append(field)
            result.evidence.extend(evidence for _, evidence in candidates)
        elif existing is not None:
            for value, evidence in candidates:
                if str(existing).casefold() == str(value).casefold():
                    result.evidence.append(
                        evidence.model_copy(update={"normalized_value": existing})
                    )
                else:
                    flags.append(
                        ReviewFlag(
                            field_name=field,
                            reason_code="contradictory_jd_employer"
                            if field == "company"
                            else "contradictory_jd_location",
                            existing_value=existing,
                            existing_evidence=[
                                e for e in result.evidence if e.field_name == field
                            ],
                            conflicting_value=value,
                            conflicting_evidence=[evidence],
                        )
                    )
        elif len(distinct) > 1:
            # Do not select one JD location or allow the model to choose a winner.
            result.warnings.append(
                NormalizationWarning(
                    code="jd_location_conflict",
                    fields=[field],
                    message="Explicit JD locations disagree; the field remains unresolved.",
                    source_values={
                        str(i): e.source_snippet for i, (_, e) in enumerate(candidates)
                    },
                )
            )
            first, first_evidence = candidates[0]
            for value, evidence in candidates[1:]:
                if str(value).casefold() != str(first).casefold():
                    flags.append(
                        ReviewFlag(
                            field_name=field,
                            reason_code="contradictory_jd_location",
                            existing_value=first,
                            existing_evidence=[first_evidence],
                            conflicting_value=value,
                            conflicting_evidence=[evidence],
                        )
                    )
    for flag in flags:
        result.warnings.append(
            NormalizationWarning(
                code=flag.reason_code,
                fields=[flag.field_name],
                message="Explicit JD evidence requires review; protected metadata was preserved.",
            )
        )
    result.fields = StructuredJobFields(**values)
    result.unresolved_fields = [
        field for field, value in values.items() if value is None
    ]
    return result, flags, signals, additions
