"""Shared finding builders and structured-artifact readers."""

from __future__ import annotations

import json
from typing import Any

from echo_bio.artifacts.store import LocalArtifactStore
from echo_bio.verifiers.contracts import VerificationFinding


def read_json_artifact(store: LocalArtifactStore, uri: str) -> dict[str, Any]:
    value = json.loads(store.get_text(store.resolve(uri)))
    if not isinstance(value, dict):
        raise ValueError("analysis artifact must contain a JSON object")
    return value


def finding(
    *,
    verifier_id: str,
    level: str,
    domain: str,
    criterion_id: str,
    passed: bool | None,
    weight: float,
    hard_gate: bool,
    message: str,
    evidence_refs: tuple[str, ...] = (),
    automated: bool = True,
    warn: bool = False,
) -> VerificationFinding:
    if passed is None:
        status = "uncertain"
        score = None
        coverage = "partial"
    elif passed:
        status = "warn" if warn else "pass"
        score = 0.5 if warn else 1.0
        coverage = "full"
    else:
        status = "fail"
        score = 0.0
        coverage = "full"
    return VerificationFinding(
        verifier_id=verifier_id,
        level=level,
        domain=domain,
        criterion_id=criterion_id,
        status=status,
        score=score,
        rubric_weight=weight,
        coverage=coverage,
        hard_gate=hard_gate,
        automated=automated,
        message=message,
        evidence_refs=evidence_refs,
    )

