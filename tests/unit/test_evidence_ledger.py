from __future__ import annotations

import json

import pytest

from echo_bio.evidence.ledger import EvidenceLedger, EvidenceLedgerIntegrityError


def test_evidence_ledger_is_content_chained_and_detects_tampering(tmp_path) -> None:
    ledger = EvidenceLedger(tmp_path)
    first = ledger.append(run_id="run-1", record_type="failure", payload={"stage": "one"})
    second = ledger.append(run_id="run-1", record_type="failure", payload={"stage": "two"})
    assert second.previous_entry_sha256 == first.entry_sha256
    assert ledger.read("run-1") == (first, second)

    path = tmp_path / "run-1" / "evidence.jsonl"
    lines = path.read_text().splitlines()
    value = json.loads(lines[0])
    value["payload"]["stage"] = "tampered"
    lines[0] = json.dumps(value)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(EvidenceLedgerIntegrityError, match="hash"):
        ledger.read("run-1")
