"""Single-job preview/cache/save orchestration with injected transports only."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from uuid import UUID

from .cache import (
    CACHE_POLICY_VERSION,
    EligibilityDecision,
    ExtractionConfiguration,
    build_fingerprint,
    cache_reuse_eligibility,
    canonical_json,
    lookup_cached_extraction,
    preview_persistence_eligibility,
)
from .extraction import extract_job
from .repository import JobRepository, PersistenceOutcome
from .schemas import JobExtractionResult


@dataclass(frozen=True)
class JobProcessingOutcome:
    job_id: str
    input_fingerprint: str
    cache_hit: bool
    cache_miss_reasons: tuple[str, ...]
    result: JobExtractionResult
    current_invocation: dict
    persistence_eligibility: EligibilityDecision
    persistence: PersistenceOutcome
    cache_reuse_eligibility: EligibilityDecision
    allow_partial_cache: bool

    @property
    def preview_kind(self):
        """Descriptive completion/enrichment label, never eligibility approval."""
        result = self.result
        diagnostics = result.diagnostics
        completed = (
            result.extraction_status in {"succeeded", "succeeded_with_warnings"}
            and result.inference_status == "succeeded"
            and result.metadata.full_source_processed
            and diagnostics.provider_completed is True
            and result.metrics.provider.done is True
            and result.metrics.provider.done_reason in {None, "stop", "eos"}
            and diagnostics.response_schema_status == "valid"
            and diagnostics.task_accounting.status == "complete"
            and not any(
                (
                    diagnostics.task_accounting.missing,
                    diagnostics.task_accounting.invalid,
                    diagnostics.task_accounting.duplicate,
                    diagnostics.task_accounting.unsolicited,
                )
            )
            and not diagnostics.failure_reason
            and not diagnostics.skip_reason
            and not result.review_required
            and not result.manual_inspection_required
        )
        if completed:
            return (
                "enriched_preview"
                if result.metrics.accepted_llm_field_count > 0
                else "completed_partial_preview"
            )
        return "unqualified_preview"

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "input_fingerprint": self.input_fingerprint,
            "cache": {
                "hit": self.cache_hit,
                "miss_reason_codes": list(self.cache_miss_reasons),
                "policy_version": CACHE_POLICY_VERSION,
                "allow_partial_cache": self.allow_partial_cache,
            },
            "preview_kind": self.preview_kind,
            "result": self.result.model_dump(mode="json"),
            "current_invocation": dict(self.current_invocation),
            "persistence_eligibility": self.persistence_eligibility.to_dict(),
            "cache_reuse_eligibility": self.cache_reuse_eligibility.to_dict(),
            "persistence": self.persistence.to_dict(),
            "notice": "Preview extraction; unknown fields are not verified negative facts. Storage or cache eligibility does not establish semantic accuracy or canonical promotion approval.",
        }


def process_job(
    job_id: UUID | str,
    *,
    repository: JobRepository,
    ollama_call,
    configuration: ExtractionConfiguration,
    force_refresh: bool = False,
    save: bool = False,
    allow_partial_cache: bool = False,
) -> JobProcessingOutcome:
    if any(
        type(flag) is not bool for flag in (save, force_refresh, allow_partial_cache)
    ):
        raise TypeError("save and cache preferences must be explicit booleans")
    started = perf_counter()
    fetch_started = perf_counter()
    snapshot = repository.fetch_job_for_extraction(job_id)
    fetch_seconds = perf_counter() - fetch_started
    source = snapshot.source
    fingerprint = build_fingerprint(source, configuration)
    cache_started = perf_counter()
    cached = lookup_cached_extraction(
        source,
        snapshot.structured_extraction,
        snapshot.extraction_input_hash,
        configuration,
        force_refresh=force_refresh,
        allow_partial_cache=allow_partial_cache,
    )
    cache_seconds = perf_counter() - cache_started
    model_calls = 0
    inference_started = perf_counter()
    if cached.hit:
        result = cached.result
    else:

        def transport(payload):
            nonlocal model_calls
            if (
                model_calls
                or payload["model"] != configuration.model.name
                or any(
                    payload[key] != value
                    for key, value in configuration.payload()["generation"].items()
                )
            ):
                raise ValueError("generation_configuration_mismatch")
            model_calls += 1
            response = ollama_call(payload)
            if isinstance(response, dict) and response.get("model") not in {
                None,
                configuration.model.name,
            }:
                raise ValueError("provider_model_name_mismatch")
            return response

        result = extract_job(
            source, ollama_call=transport, model_name=configuration.model.name
        )
    extraction_seconds = perf_counter() - inference_started
    decision = preview_persistence_eligibility(result, source, configuration)
    reuse_decision = cache_reuse_eligibility(
        result, source, configuration, allow_partial_cache=allow_partial_cache
    )
    result_bytes = len(canonical_json(result.model_dump(mode="json")).encode())
    persistence_started = perf_counter()
    if not save:
        persistence = PersistenceOutcome("preview_only", (), result_bytes)
    elif cached.hit:
        persistence = PersistenceOutcome(
            "already_stored",
            (),
            result_bytes,
            len(canonical_json(snapshot.structured_extraction).encode()),
        )
    elif not decision.allowed:
        persistence = PersistenceOutcome("blocked", decision.reason_codes, result_bytes)
    else:
        persistence = repository.save_structured_extraction(
            snapshot, result, fingerprint, configuration, authorize_write=True
        )
    persistence_seconds = perf_counter() - persistence_started
    # The result's metrics are historical when it came from cache. Never mutate
    # those metrics to make them look like measurements of this invocation.
    metrics = {
        "model_calls": model_calls,
        "graphql_fetches": 1,
        "mutation_attempted": persistence.mutation_attempted,
        "graphql_mutations": int(persistence.mutation_attempted),
        "result_origin": "cache" if cached.hit else "fresh",
        "source_fetch_seconds": fetch_seconds,
        "cache_lookup_seconds": cache_seconds,
        "extraction_seconds": extraction_seconds,
        "persistence_seconds": persistence_seconds,
        "total_wall_seconds": perf_counter() - started,
    }
    return JobProcessingOutcome(
        snapshot.job_id,
        fingerprint.digest,
        cached.hit,
        cached.reason_codes,
        result,
        metrics,
        decision,
        persistence,
        reuse_decision,
        allow_partial_cache,
    )
