"""Deterministic diagnostic extraction with exact-source validation."""

from __future__ import annotations

import hashlib
import re

from echo_bio.artifacts.contracts import ArtifactRef
from echo_bio.artifacts.store import LocalArtifactStore
from echo_bio.evidence.contracts import (
    DiagnosticQuote,
    DiagnosticReceipt,
    DiagnosticReceiptCandidate,
    ExitState,
)
from echo_bio.telemetry.redaction import canonical_json

DIAGNOSTIC_PATTERN = re.compile(r"(?i)(traceback|error|exception|warning|failed|failure|timeout)")


class ReceiptValidationError(ValueError):
    pass


def infer_exit_state(source: str) -> ExitState:
    stripped = source.lstrip().lower()
    if stripped.startswith("timeout") or "timeouterror" in stripped or re.search(r"(?im)^timeout\b", source):
        return "timeout"
    if (
        stripped.startswith(("traceback", "error", "exception"))
        or "command failed" in stripped
        or re.search(r"(?im)^(traceback|error|exception|.*\bfailed\b)", source)
    ):
        return "failure"
    return "success"


def diagnostic_quotes(source: str) -> tuple[DiagnosticQuote, ...]:
    quotes = []
    offset = 0
    for line_with_end in source.splitlines(keepends=True):
        line = line_with_end.rstrip("\r\n")
        if line and DIAGNOSTIC_PATTERN.search(line):
            quotes.append(
                DiagnosticQuote(
                    quote=line,
                    start_char=offset,
                    end_char=offset + len(line),
                    quote_sha256=hashlib.sha256(line.encode()).hexdigest(),
                )
            )
        offset += len(line_with_end)
    return tuple(quotes)


def diagnostic_summary(source: str) -> str:
    quotes = diagnostic_quotes(source)
    header = "Deterministic diagnostic signals only; no scientific conclusions are generated."
    if not quotes:
        return header + "\nNo error, warning, failure, exception, traceback, or timeout line detected.\n"
    return header + "\n" + "\n".join(f"- {quote.quote}" for quote in quotes) + "\n"


def _receipt_id(execution_id: str, source: ArtifactRef, summary: ArtifactRef, quotes) -> str:
    payload = {
        "execution_id": execution_id,
        "source_sha256": source.sha256,
        "summary_sha256": summary.sha256,
        "quotes": [quote.model_dump(mode="json") for quote in quotes],
    }
    return "diag_" + hashlib.sha256(canonical_json(payload)).hexdigest()


class DiagnosticReceiptBuilder:
    def __init__(self, store: LocalArtifactStore) -> None:
        self.store = store

    def build(
        self,
        *,
        execution_id: str,
        source_artifact: ArtifactRef,
        candidate: DiagnosticReceiptCandidate | None = None,
    ) -> DiagnosticReceipt:
        source = self.store.get_text(source_artifact)
        quotes = diagnostic_quotes(source)
        observed = infer_exit_state(source)
        if candidate is not None:
            actual = {quote.quote for quote in quotes}
            fabricated = set(candidate.proposed_quotes) - actual
            if fabricated:
                raise ReceiptValidationError(f"candidate proposed non-source diagnostic quotes: {sorted(fabricated)}")
            if candidate.proposed_exit_state not in {"unknown", observed}:
                raise ReceiptValidationError("candidate exit state conflicts with deterministic source state")
        summary_ref = self.store.put_text(
            run_id=source_artifact.run_id,
            text=diagnostic_summary(source),
            media_type="text/plain",
            data_class=source_artifact.data_class,
        )
        receipt = DiagnosticReceipt(
            receipt_id=_receipt_id(execution_id, source_artifact, summary_ref, quotes),
            execution_id=execution_id,
            source_artifact=source_artifact,
            source_sha256=source_artifact.sha256,
            source_size_bytes=source_artifact.size_bytes,
            summary_artifact=summary_ref,
            quotes=quotes,
            observed_exit_state=observed,
        )
        DiagnosticReceiptValidator(self.store).validate(receipt)
        return receipt


class DiagnosticReceiptValidator:
    def __init__(self, store: LocalArtifactStore) -> None:
        self.store = store

    def validate(self, receipt: DiagnosticReceipt) -> None:
        source_bytes = self.store.get_bytes(receipt.source_artifact)
        source = source_bytes.decode(receipt.source_artifact.encoding or "utf-8")
        self.store.verify(receipt.summary_artifact)
        if receipt.source_sha256 != hashlib.sha256(source_bytes).hexdigest():
            raise ReceiptValidationError("diagnostic source hash mismatch")
        if receipt.source_size_bytes != len(source_bytes):
            raise ReceiptValidationError("diagnostic source size mismatch")
        expected_quotes = diagnostic_quotes(source)
        if receipt.quotes != expected_quotes:
            raise ReceiptValidationError("diagnostic quote set is forged, shifted, or incomplete")
        for quote in receipt.quotes:
            if source[quote.start_char : quote.end_char] != quote.quote:
                raise ReceiptValidationError("diagnostic quote locator does not read back exactly")
            if hashlib.sha256(quote.quote.encode()).hexdigest() != quote.quote_sha256:
                raise ReceiptValidationError("diagnostic quote hash mismatch")
        if self.store.get_text(receipt.summary_artifact) != diagnostic_summary(source):
            raise ReceiptValidationError("diagnostic summary is not the deterministic source projection")
        if receipt.observed_exit_state != infer_exit_state(source):
            raise ReceiptValidationError("diagnostic exit state mismatch")
        expected_id = _receipt_id(
            receipt.execution_id,
            receipt.source_artifact,
            receipt.summary_artifact,
            receipt.quotes,
        )
        if receipt.receipt_id != expected_id:
            raise ReceiptValidationError("diagnostic receipt identity mismatch")
