"""Content-chained append-only JSONL trace ledger."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from echo_bio.contracts.trace import (
    CostRecord,
    DataClass,
    ProviderUsage,
    RedactionMetadata,
    RunEnvelope,
    TraceError,
    TraceEvent,
    TraceEventType,
)
from echo_bio.telemetry.redaction import TraceRedactor, canonical_json

SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]+$")


class TraceIntegrityError(RuntimeError):
    pass


def _event_hash(event_data: dict[str, Any]) -> str:
    payload = dict(event_data)
    payload.pop("event_sha256", None)
    return hashlib.sha256(canonical_json(payload)).hexdigest()


class TraceLedger:
    def __init__(self, root: str | Path, *, redactor: TraceRedactor | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.redactor = redactor or TraceRedactor()

    def create_run(self, envelope: RunEnvelope) -> RunTraceWriter:
        run_dir = self._run_dir(envelope.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        envelope_path = run_dir / "run.json"
        serialized = json.dumps(envelope.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        try:
            with envelope_path.open("x", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            existing = RunEnvelope.model_validate_json(envelope_path.read_text())
            if existing != envelope:
                raise TraceIntegrityError(
                    f"run envelope already exists with different content: {envelope.run_id}"
                ) from exc
        return RunTraceWriter(self, envelope)

    def open_run(self, run_id: str) -> RunTraceWriter:
        run_dir = self._run_dir(run_id)
        envelope = RunEnvelope.model_validate_json((run_dir / "run.json").read_text())
        return RunTraceWriter(self, envelope)

    def read_envelope(self, run_id: str) -> RunEnvelope:
        return RunEnvelope.model_validate_json((self._run_dir(run_id) / "run.json").read_text())

    def read_events(self, run_id: str, *, verify: bool = True) -> list[TraceEvent]:
        path = self._run_dir(run_id) / "events.jsonl"
        if not path.exists():
            return []
        events = [TraceEvent.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
        if verify:
            self._verify_events(self.read_envelope(run_id), events)
        return events

    def list_runs(self) -> list[str]:
        return sorted(path.name for path in self.root.iterdir() if path.is_dir() and (path / "run.json").exists())

    def _verify_events(self, envelope: RunEnvelope, events: list[TraceEvent]) -> None:
        previous_hash = None
        for expected_sequence, event in enumerate(events):
            if event.run_id != envelope.run_id or event.attempt_id != envelope.attempt_id:
                raise TraceIntegrityError(f"identity mismatch at sequence {expected_sequence}")
            if event.sequence_no != expected_sequence:
                raise TraceIntegrityError(f"non-contiguous sequence at {event.sequence_no}")
            if event.previous_event_sha256 != previous_hash:
                raise TraceIntegrityError(f"broken event chain at sequence {event.sequence_no}")
            calculated = _event_hash(event.model_dump(mode="json"))
            if calculated != event.event_sha256:
                raise TraceIntegrityError(f"event hash mismatch at sequence {event.sequence_no}")
            previous_hash = event.event_sha256

    def _run_dir(self, run_id: str) -> Path:
        if not SAFE_IDENTIFIER.fullmatch(run_id):
            raise ValueError(f"unsafe run identifier: {run_id!r}")
        return self.root / run_id


class RunTraceWriter:
    def __init__(self, ledger: TraceLedger, envelope: RunEnvelope) -> None:
        self.ledger = ledger
        self.envelope = envelope
        self.path = ledger._run_dir(envelope.run_id) / "events.jsonl"

    def append(
        self,
        event_type: TraceEventType,
        payload: dict[str, Any],
        *,
        data_class: DataClass | None = None,
        input_refs: tuple[str, ...] = (),
        output_refs: tuple[str, ...] = (),
        provider_usage: ProviderUsage | None = None,
        local_token_estimates: dict[str, int] | None = None,
        cost: CostRecord | None = None,
        error: TraceError | None = None,
    ) -> TraceEvent:
        selected_class = data_class or self.envelope.data_class
        redacted = self.ledger.redactor.sanitize(payload, data_class=selected_class)
        selected_error = error
        redaction_metadata = redacted.metadata
        if error is not None:
            sanitized_error = self.ledger.redactor.sanitize(
                {"message": error.message},
                data_class=selected_class,
            )
            sanitized_message = sanitized_error.value.get(
                "message",
                f"[WITHHELD:{selected_class}]",
            )
            selected_error = error.model_copy(update={"message": sanitized_message})
            redaction_metadata = RedactionMetadata(
                data_class=selected_class,
                rules_applied=tuple(
                    sorted(set(redacted.metadata.rules_applied) | set(sanitized_error.metadata.rules_applied))
                ),
                fields_withheld=(redacted.metadata.fields_withheld + sanitized_error.metadata.fields_withheld),
                secret_matches=(redacted.metadata.secret_matches + sanitized_error.metadata.secret_matches),
                original_character_count=(
                    redacted.metadata.original_character_count + sanitized_error.metadata.original_character_count
                ),
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            lines = [line for line in handle.read().splitlines() if line.strip()]
            previous_hash = None
            if lines:
                existing_events = [TraceEvent.model_validate_json(line) for line in lines]
                self.ledger._verify_events(self.envelope, existing_events)
                previous = existing_events[-1]
                previous_hash = previous.event_sha256
            sequence_no = len(lines)
            event_data = {
                "event_id": f"evt_{uuid4().hex}",
                "run_id": self.envelope.run_id,
                "attempt_id": self.envelope.attempt_id,
                "sequence_no": sequence_no,
                "event_type": event_type,
                "input_refs": input_refs,
                "output_refs": output_refs,
                "provider_usage": provider_usage,
                "local_token_estimates": local_token_estimates or {},
                "cost": cost,
                "error": selected_error,
                "redaction": redaction_metadata,
                "payload": redacted.value,
                "payload_sha256": redacted.original_sha256,
                "previous_event_sha256": previous_hash,
                "event_sha256": "0" * 64,
            }
            provisional = TraceEvent.model_validate(event_data)
            event_data = provisional.model_dump(mode="json")
            event_data["event_sha256"] = _event_hash(event_data)
            event = TraceEvent.model_validate(event_data)
            handle.seek(0, os.SEEK_END)
            handle.write(event.model_dump_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return event
