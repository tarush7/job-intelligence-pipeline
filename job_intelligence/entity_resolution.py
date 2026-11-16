"""Deterministic, explainable relationship proposals; never persistence or LLM I/O."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from copy import deepcopy
from datetime import date
from itertools import combinations
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from .adapters import UnsupportedJobSourceError, normalize_job
from .cache import canonical_json
from .deterministic import apply_jd_rules
from .extraction import prepare_sources
from .matching import (
    identifier,
    normalize_application_url,
    normalize_company,
    normalize_description,
    normalize_title,
    token_similarity,
)
from .schemas import JobExtractionInput, StructuredJobFields

MATCHING_POLICY_VERSION = "cross_source_vacancy_resolution_v1"
DESCRIPTION_SUPPORT_THRESHOLD = 0.85
DESCRIPTION_CONFLICT_THRESHOLD = 0.25
TITLE_CANDIDATE_THRESHOLD = 0.5
POSTING_SUPPORT_DAYS = 30
POSTING_CONFLICT_DAYS = 180


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceEvidence(Contract):
    job_id: str
    source: str
    source_path: str
    original_value: JsonValue
    normalized_value: JsonValue
    rule: str
    start_pos: int | None = None
    end_pos: int | None = None


class VacancyIdentity(Contract):
    kind: Literal["application_url", "requisition_id", "provider_job_id"]
    value: str
    namespace: str | None
    strong: bool
    evidence: list[SourceEvidence]


class NormalizedListing(Contract):
    job_id: str
    source: str
    existing_canonical_id: str | None
    company_key: str | None
    title_key: str | None
    unresolved_geographic_scope: str | None
    fields: StructuredJobFields
    description: str
    identities: list[VacancyIdentity]
    field_provenance: dict[str, list[SourceEvidence]]
    original_source: dict
    employer_inspection_required: bool
    material_signals: list[str]
    identity_notes: list[str]


class PairDecision(Contract):
    left_job_id: str
    right_job_id: str
    left_source: str
    right_source: str
    decision: Literal["match", "review", "no_match"]
    strong_identity_signals: list[str]
    supporting_signals: list[str]
    missing_evidence: list[str]
    contradictory_evidence: list[str]
    inspection_signals: list[str]
    description_similarity: float | None
    reason_codes: list[str]
    evidence: list[SourceEvidence]
    policy_version: str = MATCHING_POLICY_VERSION


class ClusterProposal(Contract):
    proposal_id: str
    member_job_ids: list[str]
    source_platforms: list[str]
    existing_canonical_ids: list[str]
    proposed_company: str | None
    proposed_title: str | None
    supported_attributes: dict[str, JsonValue]
    conflicting_fields: dict[str, list[SourceEvidence]]
    field_provenance: dict[str, list[SourceEvidence]]
    unknown_member_fields: dict[str, list[str]]
    review_status: Literal["proposed", "requires_review"]
    reason_codes: list[str]
    promotion_status: Literal["not_authorized"] = "not_authorized"


class CanonicalizationPlan(Contract):
    policy_version: str = MATCHING_POLICY_VERSION
    input_snapshot_hash: str
    source_job_count: int
    possible_cross_source_pairs: int
    listings: list[NormalizedListing]
    candidate_pairs: list[PairDecision]
    cluster_consistency_checks: list[PairDecision]
    clusters: list[ClusterProposal]
    unmatched_job_ids: list[str]
    existing_canonical_links: dict[str, str]
    existing_link_inconsistencies: list[PairDecision]
    rejected_inputs: list[dict[str, str]]
    notice: str = (
        "Read-only relationship proposals. No source rows are merged or moved. "
        "Similarity is not probability, is_active is not verified availability, "
        "and a matching decision does not authorize canonical promotion."
    )


def _get(row, path):
    value = row
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def normalize_listing(row: dict) -> NormalizedListing:
    original = deepcopy(row)
    job = JobExtractionInput.model_validate(original)
    job_id, source = str(job.id), job.source.strip().casefold()
    normalized, flags, signals, _ = apply_jd_rules(
        normalize_job(original), prepare_sources(job)
    )
    company_key = normalize_company(normalized.fields.company)
    title_key = normalize_title(normalized.fields.title)

    def evidence(path, value, rule):
        return SourceEvidence(
            job_id=job_id,
            source=source,
            source_path=path,
            original_value=deepcopy(_get(original, path)),
            normalized_value=value,
            rule=rule,
        )

    provenance = defaultdict(list)
    for item in normalized.evidence:
        ev = evidence(
            item.source_path,
            item.normalized_value,
            item.normalization_rule or "direct_source_value",
        )
        if item.source_snippet is not None:
            ev = ev.model_copy(
                update={
                    "original_value": item.source_snippet,
                    "start_pos": item.start_pos,
                    "end_pos": item.end_pos,
                }
            )
        provenance[item.field_name].append(ev)
    identities, notes = [], []
    urls = {
        "linkedin": ["raw_source_data.payload.applyUrl"],
        "indeed": ["raw_source_data.payload.job_url_direct"],
        "naukri": [],
    }[source] + ["raw_source_data.normalized_metadata.apply_url"]
    for path in urls:
        value = _get(original, path)
        if not isinstance(value, str) or not value.strip():
            continue
        url = normalize_application_url(value)
        notes.append(path + ":" + url.reason_code)
        if url.normalized:
            ev = evidence(path, url.normalized, url.reason_code)
            identities.append(
                VacancyIdentity(
                    kind="application_url",
                    value=url.normalized,
                    namespace=None,
                    strong=url.vacancy_specific,
                    evidence=[ev],
                )
            )
            if url.requisition_id:
                identities.append(
                    VacancyIdentity(
                        kind="requisition_id",
                        value=url.requisition_id,
                        namespace=url.requisition_namespace,
                        strong=bool(company_key),
                        evidence=[ev],
                    )
                )

    material = ["jd_review:" + f.reason_code for f in flags]
    material += ["jd_inspection:" + s.reason_code for s in signals]
    employer_inspection = any(f.field_name == "company" for f in flags) or any(
        "company" in s.fields for s in signals
    )
    for warning in normalized.warnings:
        if warning.code != "remote_false_is_ambiguous" and (
            "conflict" in warning.code or "ambiguous" in warning.code
        ):
            material.append(
                "normalization:" + warning.code + ":" + ",".join(warning.fields)
            )
            employer_inspection |= "company" in warning.fields

    for prefix in (
        "raw_source_data",
        "raw_source_data.payload",
        "raw_source_data.normalized_metadata",
    ):
        namespace_value = _get(original, prefix + ".requisition_namespace")
        namespace_value = namespace_value or _get(
            original, prefix + ".requisitionNamespace"
        )
        namespace = (
            namespace_value.strip().casefold()
            if isinstance(namespace_value, str)
            else None
        )
        if namespace in {
            "linkedin",
            "indeed",
            "naukri",
            "provider",
            "external",
            "unknown",
            "",
        }:
            namespace = None
        employer = _get(original, prefix + ".requisition_employer")
        if employer is not None and normalize_company(employer) != company_key:
            employer_inspection = True
            material.append("requisition_employer_conflict")
        for key in (
            "requisition_id",
            "requisitionId",
            "jobRequisitionId",
            "job_requisition_id",
            "requisitionNumber",
        ):
            path = prefix + "." + key
            value = _get(original, path)
            if value is None:
                continue
            req = identifier(value)
            if req is None:
                notes.append(path + ":unsupported_requisition_identifier")
                continue
            evs = [evidence(path, req, "explicit_requisition_identifier_v1")]
            for ns_key in (
                "requisition_namespace",
                "requisitionNamespace",
                "requisition_employer",
            ):
                if _get(original, prefix + "." + ns_key) is not None:
                    evs.append(
                        evidence(
                            prefix + "." + ns_key,
                            _get(original, prefix + "." + ns_key),
                            "declared_employer_requisition_scope_v1",
                        )
                    )
            identities.append(
                VacancyIdentity(
                    kind="requisition_id",
                    value=req,
                    namespace=namespace,
                    strong=bool(company_key and namespace),
                    evidence=evs,
                )
            )
            if not namespace:
                notes.append(path + ":missing_employer_requisition_namespace")
    if job.external_job_id:
        identities.append(
            VacancyIdentity(
                kind="provider_job_id",
                value=job.external_job_id,
                namespace=source,
                strong=False,
                evidence=[
                    evidence(
                        "external_job_id",
                        job.external_job_id,
                        "provider_scoped_only_v1",
                    )
                ],
            )
        )

    if (
        len({i.value for i in identities if i.kind == "application_url" and i.strong})
        > 1
    ):
        material.append("multiple_distinct_vacancy_application_urls")
    scoped_ids = defaultdict(set)
    for identity in identities:
        if identity.kind == "requisition_id" and identity.namespace:
            scoped_ids[identity.namespace].add(identity.value)
    if any(len(values) > 1 for values in scoped_ids.values()):
        material.append("internally_conflicting_requisition_ids")
    canonical = original.get("canonical_job_id")
    canonical = str(UUID(str(canonical))) if canonical is not None else None
    description = normalize_description(job.job_description)
    # Do not stitch disjoint descriptions into a new employer claim.
    if not description:
        for path, text in prepare_sources(job).items():
            if path.endswith(("descriptionText", "description")):
                description = normalize_description(text)
                break
    raw_location = original.get("location")
    unresolved_scope = None
    if isinstance(raw_location, str) and raw_location.strip():
        scope = normalize_title(raw_location)
        if (
            scope
            not in {
                None,
                "unknown",
                "unknown location",
                "remote",
                "work from home",
                "wfh",
            }
            and normalized.fields.city is None
            and (normalized.fields.country is None or "," in raw_location)
        ):
            unresolved_scope = scope
    return NormalizedListing(
        job_id=job_id,
        source=source,
        existing_canonical_id=canonical,
        company_key=company_key,
        title_key=title_key,
        fields=normalized.fields,
        unresolved_geographic_scope=unresolved_scope,
        description=description,
        identities=identities,
        field_provenance=dict(provenance),
        original_source=original,
        employer_inspection_required=employer_inspection,
        material_signals=sorted(set(material)),
        identity_notes=sorted(set(notes)),
    )


def _urls(listing):
    return {
        i.value for i in listing.identities if i.kind == "application_url" and i.strong
    }


def _reqs(listing):
    return {
        (listing.company_key, i.namespace, i.value)
        for i in listing.identities
        if i.kind == "requisition_id" and i.strong
    }


def is_candidate(left: NormalizedListing, right: NormalizedListing) -> bool:
    if _urls(left) & _urls(right) or _reqs(left) & _reqs(right):
        return True
    return bool(
        left.company_key
        and left.company_key == right.company_key
        and left.title_key
        and right.title_key
        and (
            left.title_key == right.title_key
            or (token_similarity(left.title_key, right.title_key) or 0)
            >= TITLE_CANDIDATE_THRESHOLD
        )
    )


def compare_listings(left: NormalizedListing, right: NormalizedListing) -> PairDecision:
    left, right = sorted((left, right), key=lambda row: row.job_id)
    strong, support, missing, conflicts, inspections = [], [], [], [], []
    shared_urls, shared_reqs = _urls(left) & _urls(right), _reqs(left) & _reqs(right)
    if shared_urls:
        strong.append("shared_vacancy_application_url")
    if shared_reqs:
        strong.append("shared_employer_scoped_requisition")
    same_company = bool(left.company_key and left.company_key == right.company_key)
    same_title = bool(left.title_key and left.title_key == right.title_key)
    for field, same, lvalue, rvalue in (
        ("company", same_company, left.company_key, right.company_key),
        ("title", same_title, left.title_key, right.title_key),
    ):
        if same:
            support.append("same_normalized_" + field)
        elif lvalue and rvalue:
            conflicts.append("incompatible_" + field)
        else:
            missing.append("missing_" + field)
    for label, listing in (("left", left), ("right", right)):
        if listing.employer_inspection_required:
            inspections.append(label + ":unresolved_employer_inspection")
        inspections.extend(label + ":" + code for code in listing.material_signals)
        if not _urls(listing):
            missing.append(label + ":no_valid_vacancy_application_url")
        if not _reqs(listing):
            missing.append(label + ":no_scoped_requisition_id")
    for name in ("country", "city", "remote_policy", "employment_type", "seniority"):
        a, b = getattr(left.fields, name), getattr(right.fields, name)
        if a is None or b is None:
            missing.append("unknown_" + name)
        elif normalize_title(a) == normalize_title(b):
            support.append("compatible_" + name)
        else:
            conflicts.append("incompatible_" + name)
    if left.unresolved_geographic_scope or right.unresolved_geographic_scope:
        if left.unresolved_geographic_scope == right.unresolved_geographic_scope:
            support.append("same_explicit_geographic_scope")
        else:
            inspections.append("unresolved_geographic_restriction_requires_review")
    if same_company:
        for namespace in {ns for _, ns, _ in _reqs(left)} & {
            ns for _, ns, _ in _reqs(right)
        }:
            a = {value for _, ns, value in _reqs(left) if ns == namespace}
            b = {value for _, ns, value in _reqs(right) if ns == namespace}
            if a.isdisjoint(b):
                conflicts.append(
                    "distinct_requisition_ids_in_common_employer_namespace"
                )
    salary_names = ("salary_min", "salary_max", "currency")
    a, b = (
        tuple(getattr(row.fields, name) for name in salary_names)
        for row in (left, right)
    )
    if any(v is not None for v in a) and any(v is not None for v in b) and a != b:
        conflicts.append("incompatible_compensation_packages")
    similarity = token_similarity(left.description, right.description)
    if similarity is None:
        missing.append("description_similarity_unavailable")
    elif similarity >= DESCRIPTION_SUPPORT_THRESHOLD:
        support.append("high_description_similarity")
    elif (
        similarity < DESCRIPTION_CONFLICT_THRESHOLD
        and min(len(left.description.split()), len(right.description.split())) >= 20
    ):
        conflicts.append("material_description_difference")
    dates = (left.fields.posted_date, right.fields.posted_date)
    if all(dates):
        distance = abs((dates[0] - dates[1]).days)
        if distance <= POSTING_SUPPORT_DAYS:
            support.append("posting_dates_within_30_days")
        elif distance > POSTING_CONFLICT_DAYS:
            conflicts.append("posting_dates_more_than_180_days_apart")
    else:
        missing.append("posting_date_proximity_unavailable")
    if (
        left.existing_canonical_id
        and right.existing_canonical_id
        and left.existing_canonical_id != right.existing_canonical_id
    ):
        conflicts.append("different_existing_canonical_links")
    provider_ids = [
        {(i.namespace, i.value) for i in row.identities if i.kind == "provider_job_id"}
        for row in (left, right)
    ]
    if provider_ids[0] & provider_ids[1]:
        support.append("same_provider_scoped_external_id")
    elif {v for _, v in provider_ids[0]} & {v for _, v in provider_ids[1]}:
        missing.append("provider_ids_are_source_scoped")
    if not strong:
        missing.append("no_shared_strong_vacancy_identity")
    if strong and same_company and same_title and not conflicts and not inspections:
        decision, reasons = "match", ["strong_identity_and_compatible_source_facts"]
    elif strong:
        decision, reasons = (
            "review",
            ["strong_identity_requires_conflict_or_missing_evidence_review"],
        )
    elif conflicts:
        decision, reasons = "no_match", ["incompatible_or_distinct_vacancy_evidence"]
    elif inspections and is_candidate(left, right):
        decision, reasons = "review", ["source_ambiguity_or_inspection_requires_review"]
    elif same_company and same_title:
        decision, reasons = (
            "review",
            ["similar_listing_without_shared_vacancy_identity"],
        )
    elif is_candidate(left, right):
        decision, reasons = "review", ["role_similarity_requires_manual_review"]
    else:
        decision, reasons = "no_match", ["insufficient_compatible_identity_evidence"]
    evidence = []
    for listing in (left, right):
        for name in (
            "company",
            "title",
            "city",
            "country",
            "remote_policy",
            "employment_type",
            "seniority",
            "salary_min",
            "salary_max",
            "currency",
            "posted_date",
        ):
            evidence.extend(listing.field_provenance.get(name, []))
        evidence.extend(
            ev for identity in listing.identities for ev in identity.evidence
        )
    return PairDecision(
        left_job_id=left.job_id,
        right_job_id=right.job_id,
        left_source=left.source,
        right_source=right.source,
        decision=decision,
        strong_identity_signals=sorted(strong),
        supporting_signals=sorted(support),
        missing_evidence=sorted(set(missing)),
        contradictory_evidence=sorted(set(conflicts)),
        inspection_signals=sorted(set(inspections)),
        description_similarity=similarity,
        reason_codes=reasons,
        evidence=evidence,
    )


def _cluster(members, decisions):
    members = sorted(members, key=lambda row: row.job_id)
    member_ids = [row.job_id for row in members]
    proposal = (
        "vacancy-proposal-"
        + hashlib.sha256(
            canonical_json(
                {"policy": MATCHING_POLICY_VERSION, "members": member_ids}
            ).encode()
        ).hexdigest()[:24]
    )
    provenance, conflicts, unknowns, attributes = {}, {}, {}, {}
    for name in StructuredJobFields.model_fields:
        values = [getattr(row.fields, name) for row in members]
        provenance[name] = [
            ev for row in members for ev in row.field_provenance.get(name, [])
        ]
        unknowns[name] = [
            row.job_id for row, value in zip(members, values) if value is None
        ]
        known = {canonical_json(value) for value in values if value is not None}
        if name in {"company", "title"}:
            keys = {getattr(row, name + "_key") for row in members}
            attributes[name] = (
                (
                    getattr(members[0].fields, name)
                    if len(known) == 1
                    else next(iter(keys))
                )
                if len(keys) == 1
                else None
            )
        elif name == "skills":
            if all(value is not None for value in values):
                common = set.intersection(*(set(value) for value in values))
                attributes[name] = sorted(common) or None
            else:
                attributes[name] = None
        elif len(known) <= 1:
            attributes[name] = next(
                (
                    value.isoformat() if isinstance(value, date) else value
                    for value in values
                    if value is not None
                ),
                None,
            )
        else:
            attributes[name] = None
            conflicts[name] = provenance[name]
    # Compensation is one source-supported package, never synthesized bounds.
    salary_names = ("salary_min", "salary_max", "currency")
    packages = {
        tuple(getattr(row.fields, name) for name in salary_names)
        for row in members
        if any(getattr(row.fields, name) is not None for name in salary_names)
    }
    if len(packages) > 1:
        for name in salary_names:
            attributes[name] = None
            conflicts[name] = provenance[name]
    canonical_ids = sorted(
        {row.existing_canonical_id for row in members if row.existing_canonical_id}
    )
    reasons = []
    if any(pair.decision != "match" for pair in decisions):
        reasons.append("unsafe_or_ambiguous_transitive_bridge")
    if len(canonical_ids) > 1:
        reasons.append("existing_canonical_links_disagree")
    if conflicts:
        reasons.append("canonical_attribute_conflicts_require_review")
    return ClusterProposal(
        proposal_id=proposal,
        member_job_ids=member_ids,
        source_platforms=sorted({row.source for row in members}),
        existing_canonical_ids=canonical_ids,
        proposed_company=attributes.pop("company"),
        proposed_title=attributes.pop("title"),
        supported_attributes=attributes,
        conflicting_fields=conflicts,
        field_provenance=provenance,
        unknown_member_fields=unknowns,
        review_status="requires_review" if reasons else "proposed",
        reason_codes=reasons or ["all_member_pairs_match"],
    )


def resolve_jobs(rows: list[dict]) -> CanonicalizationPlan:
    """O(n²) blocking/comparisons; complete-link validation of match components."""
    listings, rejected, seen = [], [], set()
    for row in rows:
        if not isinstance(row, dict):
            rejected.append(
                {"job_id": "unknown", "reason_code": "invalid_source_input"}
            )
            continue
        try:
            listing = normalize_listing(row)
        except (
            ValueError,
            TypeError,
            ValidationError,
            UnsupportedJobSourceError,
        ) as error:
            rejected.append(
                {
                    "job_id": str(row.get("id", "unknown")),
                    "reason_code": "unsupported_source"
                    if isinstance(error, UnsupportedJobSourceError)
                    else "invalid_source_input",
                }
            )
            continue
        if listing.job_id in seen:
            raise ValueError("duplicate_input_job_id")
        seen.add(listing.job_id)
        listings.append(listing)
    listings.sort(key=lambda row: row.job_id)
    by_id = {row.job_id: row for row in listings}
    candidates, pair_map, graph = [], {}, defaultdict(set)
    possible = 0
    for left, right in combinations(listings, 2):
        if left.source == right.source:
            continue
        possible += 1
        if not is_candidate(left, right):
            continue
        pair = compare_listings(left, right)
        candidates.append(pair)
        pair_map[(left.job_id, right.job_id)] = pair
        if pair.decision == "match":
            graph[left.job_id].add(right.job_id)
            graph[right.job_id].add(left.job_id)
    clusters, checks, visited = [], [], set()
    for job_id in sorted(graph):
        if job_id in visited:
            continue
        component, pending = set(), [job_id]
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(graph[current] - component)
        visited |= component
        decisions = []
        for a, b in combinations(sorted(component), 2):
            if (a, b) not in pair_map:
                pair_map[(a, b)] = compare_listings(by_id[a], by_id[b])
                checks.append(pair_map[(a, b)])
            decisions.append(pair_map[(a, b)])
        clusters.append(_cluster([by_id[i] for i in sorted(component)], decisions))
    links = {
        row.job_id: row.existing_canonical_id
        for row in listings
        if row.existing_canonical_id
    }
    inconsistencies = [
        pair
        for pair in candidates
        if pair.strong_identity_signals
        and "different_existing_canonical_links" in pair.contradictory_evidence
    ]
    for left, right in combinations(listings, 2):
        if (
            left.existing_canonical_id
            and left.existing_canonical_id == right.existing_canonical_id
        ):
            pair = compare_listings(left, right)
            if pair.decision != "match":
                inconsistencies.append(pair)
    safely_clustered = {
        i
        for cluster in clusters
        if cluster.review_status == "proposed"
        for i in cluster.member_job_ids
    }
    snapshot = [
        {
            key: row.original_source.get(key)
            for key in (
                *JobExtractionInput.model_fields,
                "canonical_job_id",
                "updated_at",
            )
        }
        for row in listings
    ]
    return CanonicalizationPlan(
        input_snapshot_hash=hashlib.sha256(
            canonical_json(snapshot).encode()
        ).hexdigest(),
        source_job_count=len(rows),
        possible_cross_source_pairs=possible,
        listings=listings,
        candidate_pairs=candidates,
        cluster_consistency_checks=checks,
        clusters=clusters,
        unmatched_job_ids=sorted(set(by_id) - safely_clustered),
        existing_canonical_links=links,
        existing_link_inconsistencies=inconsistencies,
        rejected_inputs=sorted(
            rejected, key=lambda value: (value["job_id"], value["reason_code"])
        ),
    )
