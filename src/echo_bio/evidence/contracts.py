"""Closed, versioned Phase 4 evidence contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from echo_bio.artifacts.contracts import ArtifactRef


def utc_now() -> datetime:
    return datetime.now(UTC)


class EvidenceContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


ExitState = Literal["success", "failure", "timeout", "unknown"]


class DiagnosticQuote(EvidenceContract):
    schema_version: Literal["echo.diagnostic-quote.v1"] = "echo.diagnostic-quote.v1"
    quote: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    quote_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_interval(self) -> DiagnosticQuote:
        if self.end_char <= self.start_char:
            raise ValueError("quote end must be after start")
        if self.end_char - self.start_char != len(self.quote):
            raise ValueError("quote interval length does not match quote")
        return self


class DiagnosticReceiptCandidate(EvidenceContract):
    """Untrusted model/reducer proposal; it has no source-hash authority."""

    schema_version: Literal["echo.diagnostic-receipt-candidate.v1"] = (
        "echo.diagnostic-receipt-candidate.v1"
    )
    proposed_quotes: tuple[str, ...] = ()
    proposed_exit_state: ExitState = "unknown"


class DiagnosticReceipt(EvidenceContract):
    schema_version: Literal["echo.diagnostic-receipt.v1"] = "echo.diagnostic-receipt.v1"
    receipt_id: str = Field(pattern=r"^diag_[a-f0-9]{64}$")
    execution_id: str = Field(pattern=r"^exec_[a-f0-9]{64}$")
    source_artifact: ArtifactRef
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_size_bytes: int = Field(ge=0)
    summary_artifact: ArtifactRef
    quotes: tuple[DiagnosticQuote, ...] = ()
    observed_exit_state: ExitState
    reducer_kind: Literal["deterministic-diagnostics-only"] = "deterministic-diagnostics-only"
    scientific_conclusions_allowed: Literal[False] = False
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_source_identity(self) -> DiagnosticReceipt:
        if self.source_sha256 != self.source_artifact.sha256:
            raise ValueError("diagnostic source hash differs from artifact")
        if self.source_size_bytes != self.source_artifact.size_bytes:
            raise ValueError("diagnostic source size differs from artifact")
        return self


class RandomSeedRecord(EvidenceContract):
    schema_version: Literal["echo.random-seed.v1"] = "echo.random-seed.v1"
    source: str = Field(min_length=1)
    value: int


ParameterValue = str | int | float | bool | None


class ScientificReceiptCandidate(EvidenceContract):
    """Untrusted proposal fields accepted only after deterministic validation."""

    schema_version: Literal["echo.scientific-receipt-candidate.v1"] = (
        "echo.scientific-receipt-candidate.v1"
    )
    declared_exit_state: ExitState = "unknown"
    parameters: dict[str, ParameterValue] = Field(default_factory=dict)


class ScientificExecutionReceipt(EvidenceContract):
    schema_version: Literal["echo.scientific-execution-receipt.v1"] = (
        "echo.scientific-execution-receipt.v1"
    )
    receipt_id: str = Field(pattern=r"^sci_[a-f0-9]{64}$")
    execution_id: str = Field(pattern=r"^exec_[a-f0-9]{64}$")
    run_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    language: Literal["python", "r", "bash", "cli"]
    code_artifact: ArtifactRef
    input_artifacts: tuple[ArtifactRef, ...] = ()
    output_artifacts: tuple[ArtifactRef, ...] = ()
    diagnostic_receipt_artifact: ArtifactRef
    parameters: dict[str, ParameterValue] = Field(default_factory=dict)
    random_seeds: tuple[RandomSeedRecord, ...] = ()
    environment_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    data_manifest_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    observed_exit_state: ExitState
    declared_exit_state: ExitState
    source_trace_event_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)


class EvidenceLocator(EvidenceContract):
    schema_version: Literal["echo.evidence-locator.v1"] = "echo.evidence-locator.v1"
    locator_id: str = Field(pattern=r"^loc_[a-f0-9]{64}$")
    artifact_uri: str = Field(min_length=1)
    kind: Literal["artifact", "exact_quote", "table_cell", "json_pointer", "recomputation"]
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, gt=0)
    quote_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    row: int | None = Field(default=None, ge=0)
    column: str | None = None
    json_pointer: str | None = None
    execution_receipt_uri: str | None = None

    @model_validator(mode="after")
    def validate_kind_fields(self) -> EvidenceLocator:
        if self.kind == "exact_quote" and (
            self.start_char is None or self.end_char is None or self.quote_sha256 is None
        ):
            raise ValueError("exact_quote requires interval and quote hash")
        if self.kind == "table_cell" and (self.row is None or not self.column):
            raise ValueError("table_cell requires row and column")
        if self.kind == "json_pointer" and not self.json_pointer:
            raise ValueError("json_pointer requires a pointer")
        if self.kind == "recomputation" and not self.execution_receipt_uri:
            raise ValueError("recomputation requires an execution receipt")
        return self


class ClaimRecord(EvidenceContract):
    schema_version: Literal["echo.claim.v1"] = "echo.claim.v1"
    claim_id: str = Field(pattern=r"^claim_[a-f0-9]{64}$")
    text: str = Field(min_length=1)
    text_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    claim_type: Literal["factual", "statistical", "mechanistic", "causal"]
    importance: Literal["important", "supporting"] = "important"
    scope: str = Field(min_length=1)
    limitations: tuple[str, ...] = ()


class EvidenceLink(EvidenceContract):
    schema_version: Literal["echo.evidence-link.v1"] = "echo.evidence-link.v1"
    link_id: str = Field(pattern=r"^link_[a-f0-9]{64}$")
    claim_id: str = Field(pattern=r"^claim_[a-f0-9]{64}$")
    locator_id: str = Field(pattern=r"^loc_[a-f0-9]{64}$")
    relation: Literal["supports", "limits", "refutes", "context"]
    source_execution_id: str | None = Field(default=None, pattern=r"^exec_[a-f0-9]{64}$")
    entailment_assessed: Literal[False] = False


class ClaimEvidenceGraph(EvidenceContract):
    schema_version: Literal["echo.claim-evidence.v1"] = "echo.claim-evidence.v1"
    run_id: str = Field(min_length=1)
    claims: tuple[ClaimRecord, ...]
    locators: tuple[EvidenceLocator, ...] = ()
    links: tuple[EvidenceLink, ...] = ()
    scientific_receipt_uris: tuple[str, ...] = ()
    important_claim_count: int = Field(ge=0)
    linked_important_claim_count: int = Field(ge=0)
    important_claim_linkage_coverage: float = Field(ge=0, le=1)
    entailment_assessed: Literal[False] = False

    @model_validator(mode="after")
    def validate_graph(self) -> ClaimEvidenceGraph:
        claim_ids = {claim.claim_id for claim in self.claims}
        locator_ids = {locator.locator_id for locator in self.locators}
        if len(claim_ids) != len(self.claims) or len(locator_ids) != len(self.locators):
            raise ValueError("claim and locator IDs must be unique")
        if any(link.claim_id not in claim_ids or link.locator_id not in locator_ids for link in self.links):
            raise ValueError("evidence link references an unknown claim or locator")
        important = {claim.claim_id for claim in self.claims if claim.importance == "important"}
        linked = {link.claim_id for link in self.links if link.claim_id in important}
        expected = len(linked) / len(important) if important else 1.0
        if self.important_claim_count != len(important):
            raise ValueError("important claim count mismatch")
        if self.linked_important_claim_count != len(linked):
            raise ValueError("linked important claim count mismatch")
        if abs(self.important_claim_linkage_coverage - expected) > 1e-12:
            raise ValueError("important claim linkage coverage mismatch")
        return self


class EvidenceLedgerEntry(EvidenceContract):
    schema_version: Literal["echo.evidence-ledger-entry.v1"] = "echo.evidence-ledger-entry.v1"
    entry_id: str = Field(pattern=r"^ev_[a-f0-9]{64}$")
    run_id: str = Field(min_length=1)
    sequence_no: int = Field(ge=0)
    record_type: Literal["diagnostic_receipt", "scientific_receipt", "claim_evidence_graph", "failure"]
    payload: dict[str, Any]
    artifact_refs: tuple[str, ...] = ()
    previous_entry_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    entry_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    occurred_at: datetime = Field(default_factory=utc_now)
