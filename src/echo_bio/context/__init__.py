"""Evidence-preserving provider-context compilation."""

from echo_bio.context.compiler import CompiledContext, ContextCompiler
from echo_bio.context.contracts import (
    CompactionDecision,
    CompactionEconomics,
    ContextCompilerConfig,
    ContextProjectionRecord,
)
from echo_bio.context.economics import evaluate_compaction
from echo_bio.context.evidence_pins import EvidenceCompleteness, build_context_snapshot
from echo_bio.context.ledger import ContextProjectionLedger

__all__ = [
    "CompactionDecision",
    "CompactionEconomics",
    "CompiledContext",
    "ContextCompiler",
    "ContextCompilerConfig",
    "ContextProjectionLedger",
    "ContextProjectionRecord",
    "EvidenceCompleteness",
    "build_context_snapshot",
    "evaluate_compaction",
]
