"""Pure Phase A adapters for the current LinkedIn, Indeed and Naukri payloads.

Precedence and safety policy
----------------------------
* Existing jobs columns precede platform metadata for spelling/display only.
  All populated candidates must reconcile after normalization. Contradictions
  or an unsupported competing value leave that field null with a warning.
* Country aliases and a small explicit region allowlist support conservative
  location parsing. No headquarters, search-country or geocoding inference.
  Multi-location and unrecognized free-form locations remain unresolved.
* Specific work-mode strings precede Indeed's weaker is_remote flag. False
  cannot establish onsite; true is a remote fallback, compatible with hybrid.
* Seniority comes only from explicit seniority metadata, never title/years.
* Compensation is supported only for Indeed's explicit employer/direct-data,
  numeric annual base-pay fields. Other units, estimates, and salary strings
  remain unresolved. No currency, hours/week or annualization factor is guessed.
* Skills come only from the platform's structured skill list. No JD parsing,
  sponsorship inference, first-seen/posted-date substitution, I/O or LLM calls.

Every source metadata value is copied into source_metadata, including Naukri
experience, repost, content-hash, reference and application URL information.
These source-only facts are deliberately outside StructuredJobFields.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from pydantic import JsonValue

from .schemas import (
    FieldEvidence,
    JobExtractionInput,
    JobNormalizationResult,
    NormalizationWarning,
    StructuredJobFields,
)

NORMALIZATION_VERSION = "job_normalization_v2"


class UnsupportedJobSourceError(ValueError):
    """The input source has no Phase A adapter."""


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return (
        value
        if value
        and value.casefold()
        not in {
            "none",
            "null",
            "nan",
            "nat",
            "n/a",
            "unknown",
            "unknown company",
            "unknown location",
        }
        else None
    )


def _present(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and not value.strip())


def _key(value: Any) -> str | None:
    text = _text(value)
    return " ".join(re.sub(r"[-_,]", " ", text.casefold()).split()) if text else None


_EMPLOYMENT = {
    "full time": "full_time",
    "fulltime": "full_time",
    "full time permanent": "full_time",
    "contract": "contract",
    "contractor": "contract",
    "contractual": "contract",
    "fixed term contract": "contract",
    "intern": "intern",
    "internship": "intern",
}
_SENIORITY = {
    "junior": "junior",
    "entry level": "junior",
    "mid": "mid",
    "mid level": "mid",
    "intermediate": "mid",
    "senior": "senior",
    "senior level": "senior",
    "staff": "staff_plus",
    "senior staff": "staff_plus",
    "principal": "staff_plus",
    "staff plus": "staff_plus",
}
_WORK_MODE = {
    "office": "onsite",
    "onsite": "onsite",
    "on site": "onsite",
    "in office": "onsite",
    "work from office": "onsite",
    "hybrid": "hybrid",
    "remote": "remote",
    "fully remote": "remote",
    "work from home": "remote",
    "wfh": "remote",
}
_COUNTRIES = {
    "india": "India",
    "in": "India",
    "ind": "India",
    "united states": "United States",
    "united states of america": "United States",
    "us": "United States",
    "usa": "United States",
    "united kingdom": "United Kingdom",
    "uk": "United Kingdom",
    "gb": "United Kingdom",
    "canada": "Canada",
    "ca": "Canada",
    "australia": "Australia",
    "au": "Australia",
    "germany": "Germany",
    "de": "Germany",
    "france": "France",
    "fr": "France",
    "singapore": "Singapore",
    "sg": "Singapore",
    "united arab emirates": "United Arab Emirates",
    "uae": "United Arab Emirates",
    "ae": "United Arab Emirates",
}
_REGIONS = {
    "karnataka",
    "maharashtra",
    "tamil nadu",
    "telangana",
    "haryana",
    "uttar pradesh",
    "west bengal",
    "gujarat",
    "kerala",
    "delhi",
    "california",
    "new york",
    "texas",
    "ny",
    "ca",
    "tx",
    "ontario",
    "england",
    "greater london",
}

# Country-scoped region codes are not city names. They never establish country.
_REGION_CODES = {
    "India": set(
        "AP AR AS BR CG CH DL GA GJ HR HP JH JK KA KL LA MH ML MN MP MZ NL OD OR PB PY RJ SK TN TS TG TR UK UP UT WB".casefold().split()
    ),
    "United States": set(
        "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC".casefold().split()
    ),
    "Canada": set("AB BC MB NB NL NS NT NU ON PE QC SK YT".casefold().split()),
    "Australia": set("ACT NSW NT QLD SA TAS VIC WA".casefold().split()),
}
# These names also denote cities. An explicit city slot (object/triplet) is
# required; a two-part region/country string remains geographically ambiguous.
_CITY_REGION_NAMES = {"United States": {"new york"}, "India": {"delhi"}}


def _is_region(value: str, country: str | None = None) -> bool:
    key = value.strip().casefold()
    codes = (
        _REGION_CODES.get(country, set())
        if country
        else set().union(*_REGION_CODES.values())
    )
    return key in _REGIONS or key in codes


@dataclass
class _Candidate:
    path: str
    original: JsonValue
    value: JsonValue
    rule: str
    supporting_values: dict[str, JsonValue] | None = None


class _Context:
    def __init__(self, job: JobExtractionInput, original: dict[str, Any]):
        self.job = job
        self.source = job.source.strip().casefold()
        self.row = job.model_dump(mode="json")
        # Preserve the supplied timestamp representation in field provenance.
        timestamp = original.get("posted_at")
        if timestamp is not None:
            self.row["posted_at"] = (
                timestamp.isoformat() if hasattr(timestamp, "isoformat") else timestamp
            )
        self.values: dict[str, Any] = {}
        self.evidence: list[FieldEvidence] = []
        self.warnings: list[NormalizationWarning] = []

    def get(self, path: str) -> JsonValue:
        value: Any = self.row
        for part in path.split("."):
            if not isinstance(value, dict):
                return None
            value = value.get(part)
        return value

    def warn(
        self,
        code: str,
        fields: list[str],
        message: str,
        values: dict[str, JsonValue] | None = None,
    ):
        self.warnings.append(
            NormalizationWarning(
                code=code,
                fields=fields,
                message=message,
                source_values=deepcopy(values or {}),
            )
        )

    def add_evidence(self, field: str, candidate: _Candidate, value: JsonValue):
        direct = type(candidate.original) is type(value) and candidate.original == value
        self.evidence.append(
            FieldEvidence(
                field_name=field,
                extraction_method="direct" if direct else "deterministic",
                source_platform=self.source,
                original_value=deepcopy(candidate.original),
                normalized_value=deepcopy(value),
                source_path=candidate.path,
                normalization_rule=None if direct else candidate.rule,
                supporting_values=deepcopy(candidate.supporting_values or {}),
            )
        )

    def resolve(self, field: str, candidates: list[_Candidate]):
        if not candidates:
            return
        raw_values = {item.path: item.original for item in candidates}
        valid = [item for item in candidates if item.value is not None]
        if len(valid) != len(candidates):
            self.warn(
                "unreconciled_metadata" if valid else "unsupported_value",
                [field],
                "A populated source value is ambiguous or unsupported; no value was guessed.",
                raw_values,
            )
            return
        chosen = valid[0].value
        comparable = lambda value: value.casefold() if isinstance(value, str) else value
        if any(comparable(item.value) != comparable(chosen) for item in valid[1:]):
            self.warn(
                "metadata_conflict",
                [field],
                "Source values disagree after normalization; field left unresolved.",
                raw_values,
            )
            return
        self.values[field] = chosen
        for item in valid:
            self.add_evidence(field, item, chosen)

    def scalar(
        self,
        field: str,
        paths: list[str],
        normalizer: Callable[[Any], JsonValue],
        rule: str,
    ):
        self.resolve(
            field,
            [
                _Candidate(path, raw, normalizer(raw), rule)
                for path in paths
                if _present(raw := self.get(path))
            ],
        )

    def finish(self) -> JobNormalizationResult:
        fields = StructuredJobFields(**self.values)
        return JobNormalizationResult(
            job_id=self.job.id,
            source=self.source,
            fields=fields,
            evidence=self.evidence,
            warnings=self.warnings,
            unresolved_fields=[
                name for name, value in fields.model_dump().items() if value is None
            ],
            source_metadata=deepcopy(self.job.raw_source_data or {}),
        )


def _parse_location(raw: Any) -> tuple[dict[str, JsonValue], bool]:
    """Return only explicitly supported components; flag ambiguous locations."""
    if isinstance(raw, dict):
        city = _text(raw.get("city"))
        country_raw = raw.get("country")
        country = _COUNTRIES.get((_text(country_raw) or "").casefold())
        invalid = any(
            _present(raw.get(key)) and not isinstance(raw.get(key), str)
            for key in ("city", "country")
        )
        if city and re.search(r"[,/;|\n]|\s+(?:or|and)\s+", city, flags=re.IGNORECASE):
            return {}, True
        values: dict[str, JsonValue] = {}
        if city and (
            not _is_region(city, country)
            or city.casefold() in _CITY_REGION_NAMES.get(country, set())
        ):
            values["city"] = city
        if _present(country_raw):
            values["country"] = country
        return values, invalid
    text = _text(raw)
    if not text:
        return {}, True
    if re.search(r"[/;|\n]|\s+(?:or|and)\s+", text, flags=re.IGNORECASE):
        return {}, True
    values = {}
    mode = _WORK_MODE.get(_key(text))
    if mode:
        return {"remote_policy": mode}, False
    suffix = re.search(r"\s*\(([^()]+)\)\s*$", text)
    if suffix and (mode := _WORK_MODE.get(_key(suffix[1]))):
        values["remote_policy"] = mode
        text = text[: suffix.start()].strip()
    parts = [part.strip() for part in text.split(",")]
    country = _COUNTRIES.get(parts[-1].casefold())
    if not country:
        return values, True
    if len(parts) == 1:
        values["country"] = country
    elif len(parts) in {2, 3} and all(parts):
        if len(parts) == 3 and not _is_region(parts[1], country):
            return values, True
        values["country"] = country
        first_mode = _WORK_MODE.get(_key(parts[0]))
        if first_mode:
            if mode and mode != first_mode:
                return {}, True
            values["remote_policy"] = first_mode
        elif (
            not _is_region(parts[0], country)
            or len(parts) == 3
            and parts[0].casefold() in _CITY_REGION_NAMES.get(country, set())
        ):
            values["city"] = parts[0]
    else:
        return values, True
    return values, False


def _locations(context: _Context, paths: list[str]) -> list[_Candidate]:
    candidates: dict[str, list[_Candidate]] = {
        "city": [],
        "country": [],
        "remote_policy": [],
    }
    for path in paths:
        raw = context.get(path)
        if not _present(raw):
            continue
        parsed, ambiguous = _parse_location(raw)
        if ambiguous:
            context.warn(
                "ambiguous_location",
                ["city", "country"],
                "Location cannot be reduced safely to a single city/country; original information retained.",
                {path: raw},
            )
            for field in ("city", "country"):
                candidates[field].append(
                    _Candidate(path, raw, None, "explicit_location_v2")
                )
        else:
            for field in ("city", "country"):
                if field in parsed:
                    candidates[field].append(
                        _Candidate(path, raw, parsed[field], "explicit_location_v2")
                    )
        if parsed.get("remote_policy"):
            candidates["remote_policy"].append(
                _Candidate(
                    path, raw, parsed["remote_policy"], "explicit_location_work_mode_v1"
                )
            )
    for field in ("city", "country"):
        context.resolve(field, candidates[field])
    return candidates["remote_policy"]


def _skills(context: _Context, path: str):
    raw = context.get(path)
    if raw is None or raw == [] or isinstance(raw, str) and not raw.strip():
        return
    if not isinstance(raw, list):
        context.warn(
            "unsupported_skills",
            ["skills"],
            "Skills must be an explicit list; free text was not split or inferred.",
            {path: raw},
        )
        return
    values: list[str] = []
    seen: dict[str, str] = {}
    for index, original in enumerate(raw):
        item_path = f"{path}[{index}]"
        skill = _text(original)
        if not skill:
            context.warn(
                "invalid_skill",
                ["skills"],
                "An empty or non-string skill was skipped.",
                {item_path: original},
            )
            continue
        chosen = seen.get(skill.casefold())
        if chosen is None:
            chosen = skill
            values.append(chosen)
            seen[skill.casefold()] = chosen
        context.add_evidence(
            "skills",
            _Candidate(item_path, original, chosen, "trim_and_deduplicate_skills_v1"),
            chosen,
        )
    if values:
        context.values["skills"] = values


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _currency(value: Any) -> str | None:
    text = _text(value)
    if text == "₹":
        return "INR"
    return text.upper() if text and re.fullmatch(r"[A-Za-z]{3}", text) else None


def _indeed_salary(context: _Context):
    prefix = "raw_source_data.payload."
    paths = [
        prefix + name
        for name in (
            "min_amount",
            "max_amount",
            "currency",
            "interval",
            "salary_source",
        )
    ]
    raw = {path: context.get(path) for path in paths}
    if not any(_present(raw[path]) for path in paths[:2]):
        context.warn(
            "missing_salary",
            ["salary_min", "salary_max"],
            "No structured base salary amounts are available.",
        )
        return
    trusted = {"direct data", "employer", "employer stated"}
    if _key(raw[prefix + "salary_source"]) not in trusted:
        context.warn(
            "untrusted_compensation_source",
            ["salary_min", "salary_max", "currency"],
            "Estimated or unattributed compensation is not employer-stated base pay.",
            raw,
        )
        return
    context.scalar(
        "currency", [prefix + "currency"], _currency, "explicit_currency_code_v1"
    )
    if _key(raw[prefix + "interval"]) not in {
        "year",
        "yearly",
        "annual",
        "annually",
        "per year",
        "per annum",
    }:
        context.warn(
            "unsupported_compensation_interval",
            ["salary_min", "salary_max"],
            "Only explicit annual base-pay amounts are supported in Phase A; no annualization assumptions were made.",
            raw,
        )
        return
    minimum, maximum = (
        _number(raw[prefix + name]) for name in ("min_amount", "max_amount")
    )
    if minimum is not None and maximum is not None and maximum < minimum:
        context.warn(
            "invalid_salary_bounds",
            ["salary_min", "salary_max"],
            "Maximum annual base salary is less than minimum; both bounds left unresolved.",
            raw,
        )
        return
    for field, path in (
        ("salary_min", prefix + "min_amount"),
        ("salary_max", prefix + "max_amount"),
    ):
        amount = context.get(path)
        if _present(amount):
            context.resolve(
                field,
                [
                    _Candidate(
                        path,
                        amount,
                        _number(amount),
                        "explicit_annual_base_pay_v1",
                        {
                            prefix + "interval": raw[prefix + "interval"],
                            prefix + "salary_source": raw[prefix + "salary_source"],
                        },
                    )
                ],
            )


def _application_url(context: _Context, paths: list[str]):
    # Application URLs remain source metadata, never an extra canonical field.
    if not any(_text(context.get(path)) for path in paths):
        context.warn(
            "missing_application_url",
            [],
            "No explicit application URL is available; the posting URL was not substituted.",
            {path: context.get(path) for path in paths},
        )


def _common(
    context: _Context, text_paths: dict[str, list[str]], locations: list[str]
) -> list[_Candidate]:
    for field in ("company", "title"):
        context.scalar(
            field, text_paths[field], _text, "trim_text_and_prefer_column_spelling_v1"
        )
    context.scalar(
        "employment_type",
        text_paths["employment_type"],
        lambda value: _EMPLOYMENT.get(_key(value)),
        "explicit_employment_type_v1",
    )
    context.scalar(
        "seniority",
        text_paths["seniority"],
        lambda value: _SENIORITY.get(_key(value)),
        "explicit_seniority_v1",
    )
    if context.job.posted_at is not None:
        context.resolve(
            "posted_date",
            [
                _Candidate(
                    "posted_at",
                    context.get("posted_at"),
                    context.job.posted_at.date().isoformat(),
                    "posted_at_calendar_date_v1",
                )
            ],
        )
    return _locations(context, locations)


def _linkedin(context: _Context):
    prefix = "raw_source_data.payload."
    modes = _common(
        context,
        {
            "company": ["company", prefix + "companyName"],
            "title": ["job_title", prefix + "title"],
            "employment_type": ["employment_type", prefix + "employmentType"],
            "seniority": ["seniority_level", prefix + "seniorityLevel"],
        },
        ["location", prefix + "location"],
    )
    context.resolve("remote_policy", modes)
    salary = context.get(prefix + "salary")
    context.warn(
        "salary_metadata_unresolved" if _present(salary) else "missing_salary",
        ["salary_min", "salary_max", "currency"],
        "LinkedIn salary text has no supported explicit annual-base-pay contract in Phase A; no compensation was inferred.",
        {prefix + "salary": salary} if _present(salary) else {},
    )
    _application_url(
        context, [prefix + "applyUrl", "raw_source_data.normalized_metadata.apply_url"]
    )


def _indeed(context: _Context):
    prefix = "raw_source_data.payload."
    modes = _common(
        context,
        {
            "company": ["company", prefix + "company"],
            "title": ["job_title", prefix + "title"],
            "employment_type": ["employment_type", prefix + "job_type"],
            "seniority": ["seniority_level", prefix + "job_level"],
        },
        ["location", prefix + "location"],
    )
    path = prefix + "work_from_home_type"
    if _present(raw := context.get(path)):
        modes.append(
            _Candidate(path, raw, _WORK_MODE.get(_key(raw)), "explicit_work_mode_v1")
        )
    flag_path = prefix + "is_remote"
    flag = context.get(flag_path)
    if modes:
        context.resolve("remote_policy", modes)
        if flag is True and context.values.get("remote_policy") == "onsite":
            context.values.pop("remote_policy")
            context.evidence = [
                item for item in context.evidence if item.field_name != "remote_policy"
            ]
            context.warn(
                "metadata_conflict",
                ["remote_policy"],
                "Explicit onsite work contradicts a true remote flag.",
                {flag_path: flag, **{item.path: item.original for item in modes}},
            )
    elif flag is True:
        context.resolve(
            "remote_policy",
            [_Candidate(flag_path, True, "remote", "indeed_true_remote_flag_v1")],
        )
    elif flag is False:
        context.warn(
            "remote_false_is_ambiguous",
            ["remote_policy"],
            "is_remote=false does not establish onsite work.",
            {flag_path: flag},
        )
    elif flag is not None:
        context.warn(
            "invalid_remote_flag",
            ["remote_policy"],
            "Remote flag must be a boolean; no coercion was applied.",
            {flag_path: flag},
        )
    _skills(context, prefix + "skills")
    _indeed_salary(context)
    _application_url(
        context,
        [prefix + "job_url_direct", "raw_source_data.normalized_metadata.apply_url"],
    )


def _naukri(context: _Context):
    modes = _common(
        context,
        {
            "company": ["company"],
            "title": ["job_title"],
            "employment_type": ["employment_type", "raw_source_data.jobType"],
            "seniority": ["seniority_level"],
        },
        ["location"],
    )
    for path in ("raw_source_data.wfhType", "raw_source_data.workModeLabel"):
        if _present(raw := context.get(path)):
            modes.append(
                _Candidate(
                    path, raw, _WORK_MODE.get(_key(raw)), "naukri_explicit_work_mode_v1"
                )
            )
    context.resolve("remote_policy", modes)
    _skills(context, "raw_source_data.skills")
    context.warn(
        "missing_salary",
        ["salary_min", "salary_max", "currency"],
        "No supported structured base salary is available from this Naukri metadata contract.",
    )
    _application_url(context, ["raw_source_data.normalized_metadata.apply_url"])


def normalize_job(job: JobExtractionInput | dict[str, Any]) -> JobNormalizationResult:
    """Normalize one supplied jobs row. No fetching, writes, inference or mutation.

    Unsupported platforms raise UnsupportedJobSourceError. Invalid input types
    raise Pydantic ValidationError. Unsupported field values are retained as raw
    metadata/warnings and remain null in the output for a future extraction stage.
    """
    original = (
        job.model_dump(mode="python") if isinstance(job, JobExtractionInput) else job
    )
    validated = (
        job
        if isinstance(job, JobExtractionInput)
        else JobExtractionInput.model_validate(job)
    )
    context = _Context(validated, original)
    adapters = {"linkedin": _linkedin, "indeed": _indeed, "naukri": _naukri}
    adapter = adapters.get(context.source)
    if adapter is None:
        raise UnsupportedJobSourceError(f"Unsupported job source: {validated.source!r}")
    payload = context.get("raw_source_data.payload")
    if (
        context.source in {"linkedin", "indeed"}
        and payload is not None
        and not isinstance(payload, dict)
    ):
        context.warn(
            "invalid_source_payload",
            [],
            "Platform payload must be an object; only usable jobs columns were considered.",
            {"raw_source_data.payload": payload},
        )
    adapter(context)
    return context.finish()
