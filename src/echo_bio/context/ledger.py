"""Append-only sidecar ledger for source/candidate/delivered context decisions."""

from __future__ import annotations

import json
import os
from pathlib import Path

from echo_bio.context.contracts import ContextProjectionRecord


class ContextProjectionLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: ContextProjectionRecord) -> None:
        payload = json.dumps(record.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n"
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    def read_all(self) -> tuple[ContextProjectionRecord, ...]:
        if not self.path.exists():
            return ()
        records = []
        for line in self.path.read_text().splitlines():
            if line.strip():
                records.append(ContextProjectionRecord.model_validate_json(line))
        return tuple(records)
