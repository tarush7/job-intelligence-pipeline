"""Conservative, explicitly framed JD locations with original character spans.

No gazetteer, headquarters inference, geocoding, or source-text rewriting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .adapters import _COUNTRIES, _is_region, _parse_location


@dataclass(frozen=True)
class LocationClaim:
    values: dict[str, str]
    start: int
    end: int
    rule: str


def _parsed_location(raw):
    raw = raw.strip().strip("*_").strip()
    values, ambiguous = _parse_location(raw)
    if not ambiguous:
        return {
            key: value for key, value in values.items() if key in {"city", "country"}
        }
    # Existing explicit single-city policy, excluding region codes and prose.
    if (
        re.fullmatch(r"[A-Z][a-zA-Z]+(?:[ -][A-Z][a-zA-Z]+){0,3}", raw)
        and not _is_region(raw)
        and raw.casefold()
        not in {
            "not specified",
            "not disclosed",
            "multiple locations",
            "various locations",
        }
    ):
        return {"city": raw}
    return {}


def location_claims(text: str) -> list[LocationClaim]:
    claims = []
    # Markdown decoration and blank lines may surround an explicit field label;
    # capture against original text so no offsets refer to cleaned strings.
    label = re.compile(
        r"^[ \t]*(?:\*\*|__)?(?:job location|work location|location)(?:\*\*|__)?[ \t]*:[ \t]*(?:\*\*|__)?[ \t]*(?:\r?\n[ \t]*)*(?P<value>[^\r\n]+)",
        re.IGNORECASE | re.MULTILINE,
    )
    based = re.compile(
        r"^[ \t]*(?:(?:this (?:position|role|job) is|you will be) based in|(?:candidates|applicants) must (?:be based in|reside in))[ \t]+(?P<value>[^\r\n.!?]+)",
        re.IGNORECASE | re.MULTILINE,
    )
    for pattern in (label, based):
        for match in pattern.finditer(text):
            values = _parsed_location(match["value"])
            if values:
                claims.append(
                    LocationClaim(
                        values, match.start(), match.end(), "jd_explicit_location_v1"
                    )
                )
            elif re.search(r"[/;|]|\b(?:or|and)\b", match["value"], re.IGNORECASE):
                claims.append(
                    LocationClaim(
                        {}, match.start(), match.end(), "jd_ambiguous_location_v1"
                    )
                )
    remote = re.compile(
        r"^[ \t]*(?:\*\*|__)?(?P<prefix>[^\r\n|]{1,80}\|[ \t]*)?(?:fully[ \t]+|100%[ \t]+)?remote[ \t]+\((?P<country>[A-Za-z][A-Za-z ]+?)[ -]based\)(?:[ \t]*\|[^\r\n]+)?[ \t]*(?:\*\*|__)?[ \t]*$",
        re.IGNORECASE | re.MULTILINE,
    )
    for match in remote.finditer(text):
        if re.search(
            r"\b(?:customers?|clients?|partners?|offices?|headquarters|teams?|not|no|example)\b",
            match.group(0),
            re.IGNORECASE,
        ):
            continue
        country = _COUNTRIES.get(match["country"].strip().casefold())
        if country:
            claims.append(
                LocationClaim(
                    {"country": country},
                    match.start(),
                    match.end(),
                    "jd_remote_country_restriction_v1",
                )
            )
    return claims


def identity_claims(text: str):
    """Only explicit hiring identity labels establish verified employer claims."""
    pattern = re.compile(
        r"^[ \t]*(?:\*\*|__)?(?:hiring employer|hiring company|employer|company)(?:\*\*|__)?[ \t]*:[ \t]*(?:\*\*|__)?[ \t]*(?P<value>[^\r\n]+)",
        re.IGNORECASE | re.MULTILINE,
    )
    for match in pattern.finditer(text):
        value = match["value"].strip().strip("*_").strip().rstrip(".")
        if (
            value
            and len(value) <= 120
            and not re.search(
                r"\b(?:client|partner|parent|customer|not|unknown|confidential)\b|[;|]",
                value,
                re.IGNORECASE,
            )
        ):
            yield value, match.start(), match.end()
