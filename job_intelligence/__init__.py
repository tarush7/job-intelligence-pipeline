"""Independent deterministic normalization and evidence-aware hybrid extraction."""

from .adapters import UnsupportedJobSourceError, normalize_job
from .contracts import ExtractionTask, SourceRecord
from .extraction import classify_fields, extract_job
from .schemas import (
    FieldEvidence,
    JobExtractionInput,
    JobExtractionResult,
    JobNormalizationResult,
    NormalizationWarning,
    RawCompensationCandidate,
    RawExtractionResponse,
    RawFieldCandidate,
    RawSkillCandidate,
    ReviewFlag,
    StructuredJobFields,
    TaskAccounting,
    UnverifiedConflictSignal,
)

__all__ = [
    "ExtractionTask",
    "FieldEvidence",
    "JobExtractionInput",
    "JobExtractionResult",
    "JobNormalizationResult",
    "NormalizationWarning",
    "RawCompensationCandidate",
    "RawExtractionResponse",
    "RawFieldCandidate",
    "RawSkillCandidate",
    "ReviewFlag",
    "SourceRecord",
    "StructuredJobFields",
    "TaskAccounting",
    "UnsupportedJobSourceError",
    "UnverifiedConflictSignal",
    "classify_fields",
    "extract_job",
    "normalize_job",
]
