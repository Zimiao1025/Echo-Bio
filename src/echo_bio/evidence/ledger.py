"""Content-chained append-only ledger for receipts and claim graphs."""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
from pathlib import Path
from typing import Any

from echo_bio.evidence.contracts import EvidenceLedgerEntry
from echo_bio.telemetry.redaction import canonical_json

SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]+$")


class EvidenceLedgerIntegrityError(RuntimeError):
    pass


def _entry_hash(payload: dict[str, Any]) -> str:
    value = dict(payload)
    value.pop("entry_sha256", None)
    return hashlib.sha256(canonical_json(value)).hexdigest()


class EvidenceLedger:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def append(
        self,
        *,
        run_id: str,
        record_type: str,
        payload: dict[str, Any],
        artifact_refs: tuple[str, ...] = (),
    ) -> EvidenceLedgerEntry:
        path = self._path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0)
            existing = [
                EvidenceLedgerEntry.model_validate_json(line)
                for line in stream.read().splitlines()
                if line.strip()
            ]
            self._verify(existing, run_id)
            sequence_no = len(existing)
            previous = existing[-1].entry_sha256 if existing else None
            identity = hashlib.sha256(
                canonical_json(
                    {
                        "run_id": run_id,
                        "sequence_no": sequence_no,
                        "record_type": record_type,
                        "payload": payload,
                        "artifact_refs": artifact_refs,
                        "previous": previous,
                    }
                )
            ).hexdigest()
            provisional = EvidenceLedgerEntry(
                entry_id=f"ev_{identity}",
                run_id=run_id,
                sequence_no=sequence_no,
                record_type=record_type,
                payload=payload,
                artifact_refs=artifact_refs,
                previous_entry_sha256=previous,
                entry_sha256="0" * 64,
            )
            data = provisional.model_dump(mode="json")
            data["entry_sha256"] = _entry_hash(data)
            entry = EvidenceLedgerEntry.model_validate(data)
            stream.seek(0, os.SEEK_END)
            stream.write(entry.model_dump_json() + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return entry

    def read(self, run_id: str, *, verify: bool = True) -> tuple[EvidenceLedgerEntry, ...]:
        path = self._path(run_id)
        if not path.exists():
            return ()
        entries = tuple(
            EvidenceLedgerEntry.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()
        )
        if verify:
            self._verify(entries, run_id)
        return entries

    @staticmethod
    def _verify(entries, run_id: str) -> None:
        previous = None
        for sequence_no, entry in enumerate(entries):
            if entry.run_id != run_id or entry.sequence_no != sequence_no:
                raise EvidenceLedgerIntegrityError("evidence ledger identity or sequence mismatch")
            if entry.previous_entry_sha256 != previous:
                raise EvidenceLedgerIntegrityError("broken evidence ledger hash chain")
            if _entry_hash(entry.model_dump(mode="json")) != entry.entry_sha256:
                raise EvidenceLedgerIntegrityError("evidence ledger entry hash mismatch")
            previous = entry.entry_sha256

    def _path(self, run_id: str) -> Path:
        if not SAFE_IDENTIFIER.fullmatch(run_id):
            raise ValueError(f"unsafe run identifier: {run_id!r}")
        return self.root / run_id / "evidence.jsonl"

