"""Deterministic execution-ledger audit helpers.

These helpers only inspect values that crossed the execution hook.  They never
infer execution from model-authored ``<execute>`` or ``<observation>`` text.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from echo_bio.contracts.trace import TraceError, TraceEvent

_EXECUTION_ERROR_PATTERNS = (
    re.compile(r"(?im)^Error:\s*.+$"),
    re.compile(r"(?im)^Error running (?:R code|Bash script)\b.*$"),
    re.compile(r"(?im)^Traceback \(most recent call last\):\s*$"),
    re.compile(r"(?im)^(?:ERROR:\s*)?.*\b(?:execution|code) timed out\b.*$"),
)


def execution_error_from_result(raw_result: str) -> TraceError | None:
    """Recognize runtime failures even when successful stdout precedes them."""

    matches = [match for pattern in _EXECUTION_ERROR_PATTERNS if (match := pattern.search(raw_result)) is not None]
    if not matches:
        return None
    first = min(matches, key=lambda match: match.start())
    excerpt = raw_result[max(0, first.start() - 200) : first.start() + 500].strip()
    return TraceError(
        error_type="execution_error",
        message=excerpt,
        retryable=None,
    )


def build_executed_code_bundle(events: Sequence[TraceEvent]) -> dict[str, Any]:
    """Build provenance only from durable execution ledger events.

    Older traces did not persist raw code bytes.  They still yield an honest
    hash-only entry instead of reconstructing code from untrusted model text.
    New traces attach a content-addressed code artifact to ``execution_started``.
    """

    starts = [event for event in events if event.event_type == "execution_started"]
    finishes = [event for event in events if event.event_type == "execution_finished"]
    unused_finishes = set(range(len(finishes)))
    executions: list[dict[str, Any]] = []
    for sequence, start in enumerate(starts, start=1):
        execution_key = str(start.payload.get("execution_key", ""))
        matched_index = next(
            (
                index
                for index in sorted(unused_finishes)
                if finishes[index].payload.get("execution_key") == execution_key
            ),
            None,
        )
        finish = finishes[matched_index] if matched_index is not None else None
        if matched_index is not None:
            unused_finishes.remove(matched_index)
        entry: dict[str, Any] = {
            "sequence": sequence,
            "language": start.payload.get("language"),
            "code_sha256": start.payload.get("code_sha256"),
            "code_character_count": start.payload.get("code_character_count"),
            "execution_key": execution_key,
            "code_artifact_uri": start.output_refs[0] if start.output_refs else None,
            "observation_artifact_uris": list(finish.output_refs) if finish else [],
            "status": ("incomplete" if finish is None else "error" if finish.error is not None else "completed"),
        }
        executions.append(entry)
    return {
        "schema_version": "echo.executed-code-bundle.v2",
        "source": "execution_ledger_only",
        "execution_started_count": len(starts),
        "execution_finished_count": len(finishes),
        "unmatched_execution_finished_count": len(unused_finishes),
        "executions": executions,
    }
