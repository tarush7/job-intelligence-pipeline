"""Conservative evidence checks and deterministic raw-fact transformations.

This module is separate because monetary parsing and field semantics are
independently testable policies, rather than transport/orchestration concerns.
Exact quotations establish provenance; these checks deliberately accept only a
bounded set of explicit formulations and do not establish measured accuracy.
"""

from __future__ import annotations

import math
import re
from datetime import date
from decimal import Decimal, InvalidOperation

from .adapters import (
    _COUNTRIES,
    _EMPLOYMENT,
    _SENIORITY,
    _WORK_MODE,
    _is_region,
    _key,
    _parse_location,
)
from .jd_locations import identity_claims, location_claims
from .schemas import FieldEvidence, RawCompensationCandidate

SALARY_FIELDS = ("salary_min", "salary_max", "currency")
DESCRIPTION_PATHS = {"job_description", "raw_source_data.payload.descriptionText"}


class CandidateRejected(ValueError):
    """Stable reason code for an individually rejected candidate."""


def checked_quote(candidate, sources: dict[str, str]) -> tuple[str, int, int, str]:
    path, quote = candidate.source_path, candidate.evidence
    if path not in sources:
        raise CandidateRejected("unsupported_source_path")
    if not isinstance(quote, str) or not quote.strip():
        raise CandidateRejected("missing_evidence")
    source = sources[path]
    start, end = candidate.start_pos, candidate.end_pos
    if start is not None or end is not None:
        if (
            start is None
            or end is None
            or end <= start
            or end > len(source)
            or source[start:end] != quote
        ):
            raise CandidateRejected("invalid_evidence_offsets")
    else:
        start = source.find(quote)
        if start < 0:
            raise CandidateRejected("fabricated_evidence")
        if source.find(quote, start + 1) >= 0:
            raise CandidateRejected("repeated_evidence_requires_offsets")
        end = start + len(quote)
    return source, start, end, quote


def _context(source: str, start: int, end: int) -> str:
    """Check the containing sentence/line, so a narrow quote cannot hide context."""
    left = max(source.rfind("\n", 0, start) + 1, 0)
    boundaries_before = list(re.finditer(r"[.!?]\s+", source[left:start]))
    if boundaries_before:
        left += boundaries_before[-1].end()
    # A complete quoted sentence already includes its right boundary. Searching
    # past it used to absorb an unrelated following sentence (e.g. visa refusal
    # into a positive skill requirement). Partial quotes still expand to their
    # containing sentence, preserving all existing semantic/negation checks.
    if (
        end > 0
        and source[end - 1] in ".!?"
        and (end == len(source) or source[end].isspace())
    ):
        return source[left:end]
    boundaries = [position for position in (source.find("\n", end),) if position >= 0]
    suffix = re.search(r"[.!?](?:\s|$)", source[end:])
    if suffix:
        boundaries.append(end + suffix.end())
    return source[left : min(boundaries) if boundaries else len(source)]


def make_evidence(
    *,
    field: str,
    value,
    candidate,
    sources: dict[str, str],
    source_platform: str,
    rule: str,
) -> FieldEvidence:
    source, start, end, quote = checked_quote(candidate, sources)
    return FieldEvidence(
        field_name=field,
        extraction_method="llm",
        source_platform=source_platform,
        original_value=source,
        normalized_value=value,
        source_path=candidate.source_path,
        source_snippet=quote,
        start_pos=start,
        end_pos=end,
        normalization_rule=rule,
    )


_METADATA_FIELDS = {
    "company": {
        "company",
        "raw_source_data.payload.companyName",
        "raw_source_data.payload.company",
    },
    "title": {"job_title", "raw_source_data.payload.title"},
    "seniority": {
        "seniority_level",
        "raw_source_data.payload.seniorityLevel",
        "raw_source_data.payload.job_level",
    },
    "employment_type": {
        "employment_type",
        "raw_source_data.payload.employmentType",
        "raw_source_data.payload.job_type",
        "raw_source_data.jobType",
    },
    "remote_policy": {
        "location",
        "raw_source_data.payload.location",
        "raw_source_data.wfhType",
        "raw_source_data.workModeLabel",
        "raw_source_data.payload.work_from_home_type",
    },
    "city": {"location", "raw_source_data.payload.location"},
    "country": {"location", "raw_source_data.payload.location"},
}


def _literal(value: str, text: str) -> bool:
    return bool(
        re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text, re.IGNORECASE)
    )


def validate_scalar(candidate, sources: dict[str, str]):
    """Return a normalized scalar only when explicit field-specific tests pass."""
    source, start, end, quote = checked_quote(candidate, sources)
    text = _context(source, start, end)
    lower = text.casefold()
    value, field = candidate.value, candidate.field_name
    metadata = candidate.source_path in _METADATA_FIELDS.get(field, set())
    if candidate.source_path not in DESCRIPTION_PATHS and not metadata:
        raise CandidateRejected("source_path_not_relevant_to_field")
    if field == "visa_sponsorship":
        if not isinstance(value, bool):
            raise CandidateRejected("invalid_sponsorship_value")
        if re.search(
            r"\b(if|only|depending|restricted|residents|citizens|may|might|could|select|certain)\b|subject to|case.by.case|based in",
            lower,
        ):
            raise CandidateRejected("conditional_sponsorship")
        country_terms = "|".join(re.escape(alias) for alias in _COUNTRIES)
        if re.search(rf"\b(?:for|in|within|from)\s+(?:{country_terms})\b", lower):
            raise CandidateRejected("geographically_restricted_sponsorship")
        refusal_pattern = (
            r"(?:visa\s+sponsorship\s+(?:is\s+)?(?:not\s+(?:available|offered|provided)|unavailable))"
            r"|(?:do not|cannot|can't)\s+(?:offer|provide)\s+(?:visa\s+)?sponsorship"
            r"|\bno\s+visa\s+sponsorship\b"
        )
        offer_pattern = (
            r"visa\s+sponsorship\s+(?:is\s+)?(?:available|offered|provided)"
            r"|(?:offer|provide)\s+visa\s+sponsorship"
        )
        refused = bool(re.search(refusal_pattern, lower))
        offered = bool(re.search(offer_pattern, re.sub(refusal_pattern, "", lower)))
        quote_supports = re.search(
            offer_pattern if value else refusal_pattern, quote, re.IGNORECASE
        )
        if (
            not quote_supports
            or (refused and offered)
            or not (refused or offered)
            or value != (not refused)
        ):
            raise CandidateRejected("unsupported_sponsorship_claim")
        return value
    if not isinstance(value, str) or not value.strip():
        raise CandidateRejected("invalid_scalar_value")
    value = value.strip()
    if field == "remote_policy":
        if metadata:
            actual = _WORK_MODE.get(_key(source))
            if actual == value and _WORK_MODE.get(_key(quote)) == value:
                return value
            raise CandidateRejected("unsupported_work_mode")
        if re.search(r"\b(?:not|never|occasionally|occasional|sometimes)\b", lower):
            raise CandidateRejected("ambiguous_work_arrangement")
        modes = set()
        for mode, term in (
            ("remote", r"(?:fully\s+|100%\s+)?remote(?:ly)?"),
            ("hybrid", r"hybrid"),
            ("onsite", r"(?:onsite|on-site|in[- ]office|in the office)"),
        ):
            if re.search(
                rf"\b(?:position|role|job|work arrangement|work mode|workplace)\s*(?:is|will be|:)\s+(?:a\s+)?{term}\b"
                rf"|\b{term}\s+(?:position|role|job|work)\b"
                rf"|\b(?:work|works|working)\s+(?:fully\s+)?{term}\b",
                lower,
            ):
                modes.add(mode)
        if re.search(r"\b(?:you|employee)\s+(?:will|must)\s+work from home\b", lower):
            modes.add("remote")
        quote_terms = {
            "remote": r"\bremote(?:ly)?\b|\bwork from home\b",
            "onsite": r"\b(?:onsite|on-site|in[- ]office|in the office)\b",
            "hybrid": r"\bhybrid\b",
        }
        if modes != {value} or not re.search(
            quote_terms.get(value, r"$^"), quote, re.IGNORECASE
        ):
            raise CandidateRejected("unsupported_work_arrangement")
        return value
    if field == "seniority":
        if re.search(r"mid[-\s]+senior", lower):
            raise CandidateRejected("ambiguous_seniority")
        if metadata:
            actual = _SENIORITY.get(_key(source))
        else:
            labels = []
            for label in (
                "junior",
                "entry level",
                "mid level",
                "senior",
                "staff",
                "principal",
            ):
                if re.search(
                    rf"(?:seniority|job level|level)\s*:\s*{label}\b"
                    rf"|(?:^|\n)\s*(?:job title|position title)\s*:\s*{label}(?:[- ]level)?\s+(?:\w+\s+){{0,2}}(?:engineer|developer|manager|analyst|role|position|job)\b"
                    rf"|\bthis is a\s+{label}(?:[- ]level)?\s+(?:\w+\s+){{0,2}}(?:engineer|developer|manager|analyst|role|position|job)\b",
                    lower,
                ):
                    labels.append(_SENIORITY.get(label))
            actual = labels[0] if labels and len(set(labels)) == 1 else None
        if actual != value:
            raise CandidateRejected("unsupported_seniority")
        if not any(
            _literal(label, quote)
            for label, mapped in _SENIORITY.items()
            if mapped == value
        ):
            raise CandidateRejected("insufficient_seniority_quote")
        return value
    if field == "employment_type":
        if metadata:
            actual = _EMPLOYMENT.get(_key(source))
            if actual == value and _EMPLOYMENT.get(_key(quote)) == value:
                return value
        terms = {
            "full_time": r"full[- ]time",
            "contract": r"contract",
            "intern": r"internship|intern",
        }
        if (
            value not in terms
            or not re.search(terms[value], quote, re.IGNORECASE)
            or not re.search(
                rf"(?:employment type|job type)\s*:\s*(?:{terms.get(value, '$^')})\b"
                rf"|\b(?:{terms.get(value, '$^')})\s+(?:role|position|job|employment)\b",
                lower,
            )
        ):
            raise CandidateRejected("unsupported_employment_type")
        return value
    if field in {"city", "country"}:
        if candidate.source_path in DESCRIPTION_PATHS:
            if any(
                not claim.values and start <= claim.start and end >= claim.end
                for claim in location_claims(source)
            ):
                raise CandidateRejected("ambiguous_or_multiple_locations")
            claims = [
                claim
                for claim in location_claims(source)
                if start <= claim.start and end >= claim.end and field in claim.values
            ]
            if claims:
                values = {claim.values[field] for claim in claims}
                expected = (
                    _COUNTRIES.get(value.casefold()) if field == "country" else value
                )
                if (
                    len(values) == 1
                    and expected
                    and next(iter(values)).casefold() == expected.casefold()
                ):
                    return next(iter(values))
                raise CandidateRejected("ambiguous_or_multiple_locations")
        if metadata:
            location = source
        else:
            match = re.search(
                r"(?:^|\n)\s*(?:job location|work location|location)\s*:\s*([^\n.!?]+)"
                r"|(?:this (?:position|role|job) is|you will be) based in\s+([^\n.!?]+)",
                text,
                re.IGNORECASE,
            )
            if not match:
                raise CandidateRejected("unsupported_job_location")
            location = next(
                group for group in match.groups() if group is not None
            ).strip()
        parsed, ambiguous = _parse_location(location)
        if ambiguous:
            # An explicitly labelled single city needs no inferred country.
            if (
                field == "city"
                and not _is_region(location)
                and not re.search(r"[,/;|]|\b(?:or|and)\b", location, re.IGNORECASE)
            ):
                parsed = {"city": location.strip()}
            else:
                raise CandidateRejected("ambiguous_or_multiple_locations")
        actual = parsed.get(field)
        expected = _COUNTRIES.get(value.casefold()) if field == "country" else value
        quoted = (
            any(
                _literal(alias, quote)
                for alias, country in _COUNTRIES.items()
                if country == expected
            )
            if field == "country"
            else _literal(value, quote)
        )
        if (
            not quoted
            or not actual
            or not expected
            or actual.casefold() != expected.casefold()
        ):
            raise CandidateRejected("unsupported_job_location")
        return actual
    if field == "posted_date":
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise CandidateRejected("unsupported_posted_date") from error
        if value not in quote or not re.search(
            r"\b(?:posted|published)(?:\s+(?:on|date))?\s*:?\s*" + re.escape(value),
            text,
            re.IGNORECASE,
        ):
            raise CandidateRejected("unsupported_posted_date")
        return parsed.isoformat()
    if field in {"company", "title"}:
        if not _literal(value, quote):
            raise CandidateRejected("insufficient_identity_quote")
        if field == "company" and candidate.source_path in DESCRIPTION_PATHS:
            for employer, claim_start, claim_end in identity_claims(source):
                if (
                    start <= claim_start
                    and end >= claim_end
                    and employer.casefold() == value.casefold()
                ):
                    return employer
        label = "employer|company" if field == "company" else "job title|position title"
        match = re.search(rf"(?:{label})\s*:\s*([^\n.!?]+)", text, re.IGNORECASE)
        if metadata and source.strip().casefold() == value.casefold():
            return value
        if match and match[1].strip().casefold() == value.casefold():
            return value
        raise CandidateRejected("unsupported_identity_claim")
    raise CandidateRejected("field_requires_dedicated_candidate_type")


def validate_skill(candidate, sources: dict[str, str]) -> str:
    source, start, end, quote = checked_quote(candidate, sources)
    value = candidate.value.strip()
    if not value or not _literal(value, quote):
        raise CandidateRejected("unsupported_skill_terminology")
    context = _context(source, start, end)
    structured = bool(
        re.fullmatch(
            r"raw_source_data(?:\.payload)?\.skills\[\d+\]", candidate.source_path
        )
    )
    if candidate.source_path not in DESCRIPTION_PATHS and not structured:
        raise CandidateRejected("source_path_not_relevant_to_field")
    if not structured and not re.search(
        r"\b(?:require|required|requirements|skills|proficiency|proficient|experience|knowledge|must|preferred)\b",
        context,
        re.IGNORECASE,
    ):
        raise CandidateRejected("skill_not_a_job_requirement")
    if re.search(
        r"\b(?:our company|our customers|company uses|we sell)\b",
        context,
        re.IGNORECASE,
    ):
        raise CandidateRejected("incidental_skill_mention")
    if re.search(r"\b(?:no|not|without)\b", context, re.IGNORECASE):
        raise CandidateRejected("negated_skill_requirement")
    return value


_AMOUNT = re.compile(
    r"(?<![\w.])(?P<number>\d[\d,]*(?:\.\d+)?)"
    r"(?:\s*(?P<unit>lakhs?|lacs?|crores?|thousand|million|k|m)\b)?",
    re.IGNORECASE,
)
_NUMBER_FORMAT = re.compile(
    r"(?:\d+|\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})+,\d{3})(?:\.\d+)?$"
)
_UNITS = {
    "": Decimal(1),
    "lakh": Decimal(100000),
    "lakhs": Decimal(100000),
    "lac": Decimal(100000),
    "lacs": Decimal(100000),
    "crore": Decimal(10000000),
    "crores": Decimal(10000000),
    "thousand": Decimal(1000),
    "k": Decimal(1000),
    "million": Decimal(1000000),
    "m": Decimal(1000000),
}
_CURRENCY = re.compile(
    r"\b(?:INR|USD|EUR|GBP|CAD|AUD|SGD|AED|CHF|JPY|NZD)\b|₹|€|£|US\$"
)


def _amounts(text: str):
    amounts = []
    for match in _AMOUNT.finditer(text):
        number, unit = match["number"], (match["unit"] or "").casefold()
        if not _NUMBER_FORMAT.fullmatch(number):
            raise CandidateRejected("unsupported_numeric_format")
        if re.match(
            r"\s*(?:billion|trillion|bn|lpa|percent|%)\b",
            text[match.end() :],
            re.IGNORECASE,
        ):
            raise CandidateRejected("unsupported_monetary_unit")
        raw = Decimal(number.replace(",", ""))
        amounts.append((raw, raw * _UNITS[unit], unit, match.start(), match.end()))
    return amounts


def _matches_amount(proposed, parsed) -> bool:
    if proposed is None or isinstance(proposed, bool):
        return False
    try:
        if isinstance(proposed, str):
            values = _amounts(proposed.strip())
            if (
                len(values) != 1
                or values[0][3] != 0
                or values[0][4] != len(proposed.strip())
            ):
                return False
            # Unit-bearing candidates must preserve the stated unit, not calculate.
            return (
                values[0][:3] == parsed[:3]
                if values[0][2]
                else values[0][0] == parsed[0]
            )
        return (
            Decimal(str(proposed)).is_finite() and Decimal(str(proposed)) == parsed[0]
        )
    except (InvalidOperation, CandidateRejected):
        return False


def validate_compensation(
    candidate: RawCompensationCandidate,
    sources: dict[str, str],
) -> tuple[dict, list[str]]:
    source, start, end, quote = checked_quote(candidate, sources)
    if candidate.source_path not in DESCRIPTION_PATHS | {
        "raw_source_data.payload.salary"
    }:
        raise CandidateRejected("source_path_not_relevant_to_field")
    context = _context(source, start, end)
    text = context.casefold()
    preceding_label = source[:start].rstrip().rsplit("\n", 1)[-1].casefold()
    if candidate.compensation_type != "base_salary":
        raise CandidateRejected("not_base_salary")
    if len(preceding_label) <= 100 and re.search(
        r"\b(?:estimated|estimate|estimates|allowances?|bonuses?|benefits|revenue|equity)\b|total compensation",
        preceding_label,
    ):
        raise CandidateRejected("non_base_or_estimated_compensation")
    if re.search(
        r"\b(?:allowance|allowances|bonus|bonuses|benefits|revenue|equity|stock|estimated|estimate|estimates|budget|valuation|profit|funding|glassdoor)\b"
        r"|total compensation|total pay|not (?:a )?base salary|salary estimate|market average"
        r"|(?:no|without) base (?:salary|pay)|not (?:offered|available|provided|paid)",
        text,
    ):
        raise CandidateRejected("non_base_or_estimated_compensation")
    if not re.search(r"\bbase\s+(?:salary|pay)\b", text):
        raise CandidateRejected("base_salary_attribution_unestablished")
    if not re.search(r"\bbase\s+(?:salary|pay)\b", quote, re.IGNORECASE):
        raise CandidateRejected("insufficient_compensation_quote")
    currencies = {
        {"₹": "INR", "€": "EUR", "£": "GBP", "US$": "USD"}.get(match[0], match[0])
        for match in _CURRENCY.finditer(quote)
    }
    if (
        len(currencies) != 1
        or not candidate.currency
        or candidate.currency.upper() not in currencies
    ):
        raise CandidateRejected("unsupported_or_ambiguous_currency")
    currency = next(iter(currencies))
    periods = set()
    for period, pattern in (
        ("annual", r"\b(?:annual|annually|yearly|per year|per annum)\b|\bp\.a\."),
        ("monthly", r"\b(?:monthly|per month|a month)\b"),
        ("weekly", r"\b(?:weekly|per week)\b"),
        ("daily", r"\b(?:daily|per day)\b"),
        ("hourly", r"\b(?:hourly|per hour)\b"),
    ):
        if re.search(pattern, quote, re.IGNORECASE):
            periods.add(period)
    if periods != {candidate.pay_period} or candidate.pay_period not in {
        "annual",
        "monthly",
    }:
        raise CandidateRejected("unsupported_or_ambiguous_pay_period")
    if re.search(r"(?:^|[₹$€£]|\b(?:INR|USD|EUR|GBP)\s*)-\s*\d", quote):
        raise CandidateRejected("negative_salary")
    amounts = _amounts(quote)
    if not 1 <= len(amounts) <= 2:
        raise CandidateRejected("ambiguous_monetary_amounts")
    if re.search(r"-\s*(?:[A-Z]{3}|[₹$€£])?\s*$", quote[: amounts[0][3]]):
        raise CandidateRejected("negative_salary")
    minimum = maximum = None
    if len(amounts) == 2:
        connector = quote[amounts[0][4] : amounts[1][3]]
        if bool(amounts[0][2]) != bool(amounts[1][2]):
            raise CandidateRejected("ambiguous_or_shared_monetary_units")
        if re.search(r"\b(?:to|and)\s*-\s*$", connector, re.IGNORECASE):
            raise CandidateRejected("negative_salary")
        if not re.search(r"[-–—]|\b(?:and|to)\b", connector, re.IGNORECASE):
            raise CandidateRejected("unrelated_salary_bounds")
        if not _matches_amount(candidate.amount_min, amounts[0]) or not _matches_amount(
            candidate.amount_max, amounts[1]
        ):
            raise CandidateRejected("amount_not_supported_by_evidence")
        minimum, maximum = amounts[0][1], amounts[1][1]
    else:
        upper = bool(
            re.search(
                r"\b(?:up to|at most|maximum|no more than)\b", quote, re.IGNORECASE
            )
        )
        lower = bool(
            re.search(
                r"\b(?:at least|minimum|starting at|from)\b", quote, re.IGNORECASE
            )
        )
        if upper and lower:
            raise CandidateRejected("ambiguous_salary_bound")
        if upper:
            if candidate.amount_min is not None or not _matches_amount(
                candidate.amount_max, amounts[0]
            ):
                raise CandidateRejected("amount_not_supported_by_evidence")
            maximum = amounts[0][1]
        else:
            if not _matches_amount(candidate.amount_min, amounts[0]):
                raise CandidateRejected("amount_not_supported_by_evidence")
            if lower:
                if candidate.amount_max is not None:
                    raise CandidateRejected("fabricated_salary_bound")
                minimum = amounts[0][1]
            else:
                if candidate.amount_max is not None and not _matches_amount(
                    candidate.amount_max, amounts[0]
                ):
                    raise CandidateRejected("amount_not_supported_by_evidence")
                minimum = maximum = amounts[0][1]
    if minimum is not None and maximum is not None and maximum < minimum:
        raise CandidateRejected("invalid_salary_bounds")
    multiplier = Decimal(12) if candidate.pay_period == "monthly" else Decimal(1)
    values = {
        "salary_min": float(minimum * multiplier) if minimum is not None else None,
        "salary_max": float(maximum * multiplier) if maximum is not None else None,
        "currency": currency,
    }
    if any(
        isinstance(value, float) and not math.isfinite(value)
        for value in values.values()
    ):
        raise CandidateRejected("salary_overflow")
    rules = [
        "explicit_base_salary_v1",
        "point_equal_bounds_or_explicit_sided_bounds_v1",
    ]
    rules.extend(
        f"unit_{unit}_v1" for unit in dict.fromkeys(item[2] for item in amounts) if unit
    )
    rules.append("monthly_x12_v1" if multiplier == 12 else "annual_x1_v1")
    return values, rules
