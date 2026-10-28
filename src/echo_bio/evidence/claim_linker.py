"""Deterministic assembly of structured claims and evidence locators."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from typing import Literal

from echo_bio.artifacts.store import LocalArtifactStore
from echo_bio.evidence.contracts import ClaimEvidenceGraph, ClaimRecord, EvidenceLink, EvidenceLocator
from echo_bio.evidence.diagnostic_receipt import ReceiptValidationError
from echo_bio.telemetry.redaction import canonical_json


@dataclass(frozen=True)
class LocatorDraft:
    artifact_uri: str
    kind: Literal["artifact", "exact_quote", "table_cell", "json_pointer", "recomputation"]
    relation: Literal["supports", "limits", "refutes", "context"] = "supports"
    start_char: int | None = None
    end_char: int | None = None
    row: int | None = None
    column: str | None = None
    json_pointer: str | None = None
    execution_receipt_uri: str | None = None
    source_execution_id: str | None = None


@dataclass(frozen=True)
class StructuredClaimDraft:
    text: str
    claim_type: Literal["factual", "statistical", "mechanistic", "causal"]
    scope: str
    importance: Literal["important", "supporting"] = "important"
    limitations: tuple[str, ...] = ()
    evidence: tuple[LocatorDraft, ...] = ()


def _digest(value) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


class ClaimEvidenceAssembler:
    def __init__(self, store: LocalArtifactStore) -> None:
        self.store = store

    def assemble(
        self,
        *,
        run_id: str,
        claims: tuple[StructuredClaimDraft, ...],
        scientific_receipt_uris: tuple[str, ...] = (),
    ) -> ClaimEvidenceGraph:
        for uri in scientific_receipt_uris:
            self.store.verify(self.store.resolve(uri))
        claim_records = []
        locators: dict[str, EvidenceLocator] = {}
        links = []
        for draft in claims:
            normalized = draft.text.strip()
            text_sha = hashlib.sha256(normalized.encode()).hexdigest()
            claim_id = f"claim_{_digest({'text': normalized, 'type': draft.claim_type, 'scope': draft.scope})}"
            claim_records.append(
                ClaimRecord(
                    claim_id=claim_id,
                    text=normalized,
                    text_sha256=text_sha,
                    claim_type=draft.claim_type,
                    importance=draft.importance,
                    scope=draft.scope,
                    limitations=draft.limitations,
                )
            )
            for locator_draft in draft.evidence:
                locator = self._materialize_locator(locator_draft)
                if draft.claim_type == "statistical" and locator.kind not in {
                    "table_cell",
                    "json_pointer",
                    "recomputation",
                }:
                    raise ReceiptValidationError(
                        "statistical claims require table-cell, JSON-pointer, or recomputation evidence"
                    )
                locators[locator.locator_id] = locator
                link_id = "link_" + _digest(
                    {
                        "claim_id": claim_id,
                        "locator_id": locator.locator_id,
                        "relation": locator_draft.relation,
                    }
                )
                links.append(
                    EvidenceLink(
                        link_id=link_id,
                        claim_id=claim_id,
                        locator_id=locator.locator_id,
                        relation=locator_draft.relation,
                        source_execution_id=locator_draft.source_execution_id,
                    )
                )
        important = {claim.claim_id for claim in claim_records if claim.importance == "important"}
        linked = {link.claim_id for link in links if link.claim_id in important}
        return ClaimEvidenceGraph(
            run_id=run_id,
            claims=tuple(claim_records),
            locators=tuple(locators.values()),
            links=tuple(links),
            scientific_receipt_uris=scientific_receipt_uris,
            important_claim_count=len(important),
            linked_important_claim_count=len(linked),
            important_claim_linkage_coverage=len(linked) / len(important) if important else 1.0,
        )

    def _materialize_locator(self, draft: LocatorDraft) -> EvidenceLocator:
        ref = self.store.resolve(draft.artifact_uri)
        content = self.store.get_text(ref)
        quote_sha = None
        if draft.kind == "exact_quote":
            if draft.start_char is None or draft.end_char is None:
                raise ReceiptValidationError("exact quote draft requires an interval")
            quote = content[draft.start_char : draft.end_char]
            if not quote:
                raise ReceiptValidationError("exact quote interval is empty")
            quote_sha = hashlib.sha256(quote.encode()).hexdigest()
        elif draft.kind == "table_cell":
            if draft.row is None or not draft.column:
                raise ReceiptValidationError("table cell draft requires row and column")
            rows = list(csv.DictReader(io.StringIO(content)))
            if draft.row >= len(rows) or draft.column not in rows[draft.row]:
                raise ReceiptValidationError("table cell locator is outside the source table")
        elif draft.kind == "json_pointer":
            self._read_json_pointer(json.loads(content), draft.json_pointer or "")
        elif draft.kind == "recomputation":
            if not draft.execution_receipt_uri:
                raise ReceiptValidationError("recomputation requires an execution receipt URI")
            self.store.verify(self.store.resolve(draft.execution_receipt_uri))
        identity = {
            "artifact_uri": draft.artifact_uri,
            "kind": draft.kind,
            "start_char": draft.start_char,
            "end_char": draft.end_char,
            "row": draft.row,
            "column": draft.column,
            "json_pointer": draft.json_pointer,
            "execution_receipt_uri": draft.execution_receipt_uri,
        }
        return EvidenceLocator(
            locator_id=f"loc_{_digest(identity)}",
            artifact_uri=draft.artifact_uri,
            kind=draft.kind,
            start_char=draft.start_char,
            end_char=draft.end_char,
            quote_sha256=quote_sha,
            row=draft.row,
            column=draft.column,
            json_pointer=draft.json_pointer,
            execution_receipt_uri=draft.execution_receipt_uri,
        )

    @staticmethod
    def _read_json_pointer(value, pointer: str):
        if not pointer.startswith("/"):
            raise ReceiptValidationError("JSON pointer must start with /")
        current = value
        for raw in pointer.removeprefix("/").split("/"):
            key = raw.replace("~1", "/").replace("~0", "~")
            try:
                current = current[int(key)] if isinstance(current, list) else current[key]
            except (KeyError, IndexError, ValueError, TypeError) as error:
                raise ReceiptValidationError("JSON pointer is not reachable") from error
        return current
