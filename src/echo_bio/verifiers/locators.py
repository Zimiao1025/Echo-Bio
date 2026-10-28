"""Exact readback for typed evidence locators."""

from __future__ import annotations

import csv
import hashlib
import io
import json

from echo_bio.artifacts.store import LocalArtifactStore
from echo_bio.evidence.contracts import EvidenceLocator


def read_locator(store: LocalArtifactStore, locator: EvidenceLocator) -> str:
    ref = store.resolve(locator.artifact_uri)
    content = store.get_text(ref)
    if locator.kind == "artifact":
        return content[:500]
    if locator.kind == "exact_quote":
        excerpt = content[locator.start_char : locator.end_char]
        if not excerpt or hashlib.sha256(excerpt.encode()).hexdigest() != locator.quote_sha256:
            raise ValueError("exact-quote locator does not read back to its frozen hash")
        return excerpt
    if locator.kind == "table_cell":
        rows = list(csv.DictReader(io.StringIO(content)))
        if locator.row >= len(rows) or locator.column not in rows[locator.row]:
            raise ValueError("table-cell locator is outside its source")
        return str(rows[locator.row][locator.column])
    if locator.kind == "json_pointer":
        current = json.loads(content)
        for raw in locator.json_pointer.removeprefix("/").split("/"):
            key = raw.replace("~1", "/").replace("~0", "~")
            current = current[int(key)] if isinstance(current, list) else current[key]
        return json.dumps(current, sort_keys=True) if not isinstance(current, str) else current
    receipt = store.get_text(store.resolve(locator.execution_receipt_uri))
    return receipt[:500]
