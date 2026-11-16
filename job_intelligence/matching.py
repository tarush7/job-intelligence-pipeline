"""Conservative identity primitives. No I/O, redirects or employer aliases."""

from __future__ import annotations

import html
import re
import unicodedata
from urllib.parse import parse_qsl, unquote_plus, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict

TRACKING_KEYS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "gclid",
        "fbclid",
        "msclkid",
        "mc_cid",
        "mc_eid",
    }
)
VACANCY_QUERY_KEYS = frozenset(
    {
        "jobid",
        "job_id",
        "gh_jid",
        "requisitionid",
        "requisition_id",
        "requisitionnumber",
        "reqid",
        "req_id",
        "vacancyid",
        "positionid",
    }
)
REQUISITION_QUERY_KEYS = frozenset(
    {"requisitionid", "requisition_id", "requisitionnumber", "reqid", "req_id"}
)
NON_DIRECT_HOSTS = (
    "linkedin.com",
    "indeed.com",
    "indeed.co.in",
    "naukri.com",
    "t.co",
    "bit.ly",
    "tinyurl.com",
    "forms.gle",
    "google.com",
    "facebook.com",
)


def _text(value: str | None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    return None if value in {"unknown", "n/a", "none", "null"} else value


def normalize_company(value: str | None) -> str | None:
    value = _text(value)
    return " ".join(re.sub(r"[.,]", " ", value).split()) or None if value else None


def normalize_title(value: str | None) -> str | None:
    value = _text(value)
    return " ".join(re.sub(r"[.,()–—-]", " ", value).split()) or None if value else None


def normalize_description(value: str | None) -> str:
    if not value:
        return ""
    # Comparison-only representation; originals remain in the source snapshot.
    value = re.sub(r"<[^>]+>", " ", value)
    return " ".join(
        re.findall(
            r"\w+", unicodedata.normalize("NFKC", html.unescape(value)).casefold()
        )
    )


def identifier(value) -> str | None:
    """Preserve ID case; never coerce booleans, floats or unlabeled provider IDs."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", value):
        return None
    return value if any(c.isdigit() for c in value) else None


class ApplicationURL(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    original: str
    normalized: str | None
    vacancy_specific: bool
    reason_code: str
    requisition_id: str | None = None
    requisition_namespace: str | None = None


def normalize_application_url(value: str) -> ApplicationURL:
    """Recognized vacancy routes/query IDs only; retain all other URL semantics."""

    def outcome(normalized, strong, reason, req=None, namespace=None):
        return ApplicationURL(
            original=value,
            normalized=normalized,
            vacancy_specific=strong,
            reason_code=reason,
            requisition_id=req,
            requisition_namespace=namespace,
        )

    try:
        parts = urlsplit(value.strip())
        host = (parts.hostname or "").casefold()
        port = parts.port
        if (
            parts.scheme not in {"http", "https"}
            or "." not in host
            or parts.username is not None
            or parts.password is not None
            or re.search(r"[\x00-\x20\x7f]", value.strip())
        ):
            return outcome(None, False, "invalid_application_url")
        netloc = host + (f":{port}" if port is not None else "")
        query = "&".join(
            pair
            for pair in parts.query.split("&")
            if pair
            and unquote_plus(pair.partition("=")[0]).casefold() not in TRACKING_KEYS
        )
        normalized = urlunsplit(
            (parts.scheme, netloc, parts.path or "/", query, parts.fragment)
        )
        if any(
            host == suffix or host.endswith("." + suffix) for suffix in NON_DIRECT_HOSTS
        ):
            return outcome(normalized, False, "provider_or_redirect_url")
        if parts.fragment:
            return outcome(normalized, False, "unvalidated_fragment_identity")
        segments = [s for s in parts.path.split("/") if s]
        lowered = [s.casefold() for s in segments]
        if any(
            s
            in {
                "search",
                "login",
                "signin",
                "sign-in",
                "register",
                "redirect",
                "redirects",
            }
            for s in lowered
        ):
            return outcome(normalized, False, "generic_or_redirect_application_page")
        values = {}
        for key, item in parse_qsl(query, keep_blank_values=True):
            if key.casefold() in {
                "url",
                "redirect",
                "redirect_uri",
                "redirect_url",
                "destination",
                "target",
                "returnurl",
                "return_url",
            }:
                return outcome(
                    normalized, False, "redirect_query_requires_external_verification"
                )
            if key.casefold() in VACANCY_QUERY_KEYS:
                values.setdefault(key.casefold(), set()).add(item)
        if any(
            len(items) != 1 or identifier(next(iter(items))) is None
            for items in values.values()
        ):
            return outcome(normalized, False, "ambiguous_vacancy_query_identifier")
        path_identity = any(
            marker
            in {"job", "jobs", "positions", "vacancies", "requisitions", "openings"}
            and index + 1 < len(segments)
            and identifier(segments[index + 1]) is not None
            for index, marker in enumerate(lowered)
        )
        # A longer /job/location/title-ID route is deliberately unsupported.
        if not values and not path_identity:
            return outcome(normalized, False, "generic_or_unsupported_application_url")
        requisitions = {
            next(iter(items))
            for key, items in values.items()
            if key in REQUISITION_QUERY_KEYS
        }
        if len(requisitions) > 1:
            return outcome(
                normalized, False, "conflicting_requisition_query_identifiers"
            )
        req = next(iter(requisitions)) if requisitions else None
        namespace = f"url-origin:{parts.scheme}://{netloc}" if req else None
        return outcome(
            normalized, True, "validated_vacancy_application_url", req, namespace
        )
    except (ValueError, TypeError):
        return outcome(None, False, "invalid_application_url")


def token_similarity(left: str, right: str) -> float | None:
    a, b = set(left.split()), set(right.split())
    return round(len(a & b) / len(a | b), 6) if a and b else None
