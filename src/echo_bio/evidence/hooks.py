"""Phase 4 hook: archive execution evidence and assemble non-authoritative claim links."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from echo_bio.artifacts.hooks import ArtifactBackedTelemetryHooks
from echo_bio.contracts import (
    ExecutionDecision,
    ExecutionDraft,
    FinalizationDecision,
    OutcomeDraft,
    RawExecutionObserved,
)
from echo_bio.evidence.claim_linker import ClaimEvidenceAssembler, LocatorDraft, StructuredClaimDraft
from echo_bio.evidence.diagnostic_receipt import DiagnosticReceiptBuilder
from echo_bio.evidence.ledger import EvidenceLedger
from echo_bio.evidence.scientific_receipt import ScientificReceiptBuilder, persist_receipt
from echo_bio.telemetry.hooks import GROUNDING_PATTERN

ARTIFACT_URI_PATTERN = re.compile(r"artifact://[A-Za-z0-9_.-]+/art_[a-f0-9]{64}\?sha256=[a-f0-9]{64}")
STATISTICAL_PATTERN = re.compile(r"(?i)(\bp\s*[<=>]|fdr|confidence interval|\bci\b|odds ratio|log2?fc|effect size)")
CAUSAL_PATTERN = re.compile(r"(?i)(causes?|caused|causal|drives?|results? in|leads? to)")
MECHANISTIC_PATTERN = re.compile(r"(?i)(mechanis|pathway|mediates?|regulates?|inhibits?|activates?)")


@dataclass(frozen=True)
class PendingExecution:
    execution_id: str
    code_artifact_uri: str


class EvidenceBackedTelemetryHooks(ArtifactBackedTelemetryHooks):
    def __init__(self, *args, evidence_ledger: EvidenceLedger, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.evidence_ledger = evidence_ledger
        self._pending: PendingExecution | None = None
        self._execution_ordinal = 0
        self.scientific_receipt_uris: list[str] = []

    def before_execution(self, event: ExecutionDraft) -> ExecutionDecision:
        decision = super().before_execution(event)
        self._execution_ordinal += 1
        identity = {
            "run_id": event.identity.run_id,
            "attempt_id": event.identity.attempt_id,
            "ordinal": self._execution_ordinal,
            "language": event.language,
            "code": decision.code,
        }
        execution_id = "exec_" + hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        try:
            code_ref = self.artifact_store.put_text(
                run_id=event.identity.run_id,
                text=decision.code,
                media_type="text/x-python" if event.language == "python" else "text/plain",
                data_class=self.config.data_class,
            )
            self._pending = PendingExecution(execution_id=execution_id, code_artifact_uri=code_ref.uri)
        except Exception as error:
            self._pending = None
            self.evidence_ledger.append(
                run_id=event.identity.run_id,
                record_type="failure",
                payload={
                    "stage": "code_artifact",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
        return decision

    def after_execution(self, event: RawExecutionObserved):
        projection = super().after_execution(event)
        pending = self._pending
        self._pending = None
        if pending is None:
            return projection
        try:
            execution_event = next(
                trace_event
                for trace_event in reversed(self.ledger.read_events(event.identity.run_id))
                if trace_event.event_type == "execution_finished"
            )
            if not execution_event.output_refs:
                raise RuntimeError("raw execution artifact is unavailable")
            source_ref = self.artifact_store.resolve(execution_event.output_refs[0])
            diagnostic = DiagnosticReceiptBuilder(self.artifact_store).build(
                execution_id=pending.execution_id,
                source_artifact=source_ref,
            )
            diagnostic_ref = persist_receipt(self.artifact_store, event.identity.run_id, diagnostic)
            self.evidence_ledger.append(
                run_id=event.identity.run_id,
                record_type="diagnostic_receipt",
                payload=diagnostic.model_dump(mode="json"),
                artifact_refs=(source_ref.uri, diagnostic.summary_artifact.uri, diagnostic_ref.uri),
            )
            scientific = ScientificReceiptBuilder(self.artifact_store).build(
                execution_id=pending.execution_id,
                run_id=event.identity.run_id,
                attempt_id=event.identity.attempt_id,
                language=event.language,
                code_artifact=self.artifact_store.resolve(pending.code_artifact_uri),
                output_artifacts=(source_ref,),
                diagnostic_receipt_artifact=diagnostic_ref,
                environment_digest=self.config.environment_digest,
                data_manifest_digest=self.config.data_manifest_digest,
                source_trace_event_id=execution_event.event_id,
            )
            scientific_ref = persist_receipt(self.artifact_store, event.identity.run_id, scientific)
            self.scientific_receipt_uris.append(scientific_ref.uri)
            self.evidence_ledger.append(
                run_id=event.identity.run_id,
                record_type="scientific_receipt",
                payload=scientific.model_dump(mode="json"),
                artifact_refs=(
                    scientific.code_artifact.uri,
                    *(ref.uri for ref in scientific.input_artifacts),
                    *(ref.uri for ref in scientific.output_artifacts),
                    diagnostic_ref.uri,
                    scientific_ref.uri,
                ),
            )
        except Exception as error:
            self.evidence_ledger.append(
                run_id=event.identity.run_id,
                record_type="failure",
                payload={
                    "stage": "receipt_assembly",
                    "execution_id": pending.execution_id,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
        return projection

    def before_finalize(self, event: OutcomeDraft) -> FinalizationDecision:
        decision = super().before_finalize(event)
        try:
            drafts = self._claim_drafts(decision.content)
            graph = ClaimEvidenceAssembler(self.artifact_store).assemble(
                run_id=event.identity.run_id,
                claims=drafts,
                scientific_receipt_uris=tuple(self.scientific_receipt_uris),
            )
            graph_ref = self.artifact_store.put_text(
                run_id=event.identity.run_id,
                text=json.dumps(graph.model_dump(mode="json"), sort_keys=True, separators=(",", ":")),
                media_type="application/json",
                data_class=self.config.data_class,
            )
            self.evidence_ledger.append(
                run_id=event.identity.run_id,
                record_type="claim_evidence_graph",
                payload=graph.model_dump(mode="json"),
                artifact_refs=(*graph.scientific_receipt_uris, graph_ref.uri),
            )
        except Exception as error:
            self.evidence_ledger.append(
                run_id=event.identity.run_id,
                record_type="failure",
                payload={
                    "stage": "claim_linking",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
        return decision

    @staticmethod
    def _claim_drafts(content: str) -> tuple[StructuredClaimDraft, ...]:
        plain = re.sub(r"</?(?:solution|think|execute)[^>]*>", " ", content, flags=re.IGNORECASE)
        statements = [item.strip() for item in re.split(r"(?<=[.!?])\s+|\n+", plain) if len(item.strip()) >= 12]
        drafts = []
        for statement in statements:
            if STATISTICAL_PATTERN.search(statement):
                claim_type = "statistical"
            elif CAUSAL_PATTERN.search(statement):
                claim_type = "causal"
            elif MECHANISTIC_PATTERN.search(statement):
                claim_type = "mechanistic"
            else:
                claim_type = "factual"
            evidence = ()
            uris = ARTIFACT_URI_PATTERN.findall(statement)
            if uris and claim_type != "statistical":
                evidence = tuple(LocatorDraft(artifact_uri=uri, kind="artifact") for uri in uris)
            limitations = () if GROUNDING_PATTERN.search(statement) else ("No explicit evidence locator in final text",)
            drafts.append(
                StructuredClaimDraft(
                    text=statement,
                    claim_type=claim_type,
                    scope="final answer",
                    limitations=limitations,
                    evidence=evidence,
                )
            )
        return tuple(drafts)

