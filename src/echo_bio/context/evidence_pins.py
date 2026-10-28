"""Canonical scientific state snapshot and evidence-completeness checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from echo_bio.contracts.plans import ResearchPlan


@dataclass(frozen=True)
class EvidenceCompleteness:
    complete: bool
    preserved_active_step: bool
    preserved_open_issues: bool
    preserved_evidence_pins: bool
    errors: tuple[str, ...]


def build_context_snapshot(
    *,
    task: str,
    plan: ResearchPlan,
    resource_slice: tuple[str, ...] = (),
) -> dict[str, Any]:
    active = next((step for step in plan.steps if step.step_id == plan.active_step_id), None)
    completed = [
        {
            "step_id": step.step_id,
            "evidence": [item.model_dump(mode="json") for item in step.completion_evidence],
        }
        for step in plan.steps
        if step.status == "completed"
    ]
    return {
        "schema_version": "echo.context-snapshot.v1",
        "task": task,
        "plan_id": plan.plan_id,
        "plan_revision": plan.revision,
        "objective": plan.objective,
        "active_step": active.model_dump(mode="json") if active else None,
        "steps": [
            {
                "step_id": step.step_id,
                "title": step.title,
                "status": step.status,
                "dependencies": list(step.dependencies),
            }
            for step in plan.steps
        ],
        "resource_slice": list(resource_slice),
        "evidence_pins": [pin.model_dump(mode="json") for pin in plan.evidence_pins],
        "unresolved_issues": [
            issue.model_dump(mode="json") for issue in plan.unresolved_issues if issue.status == "open"
        ],
        "completed_step_evidence": completed,
    }


def canonical_snapshot_bytes(snapshot: dict[str, Any]) -> bytes:
    return json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def snapshot_sha256(snapshot: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_snapshot_bytes(snapshot)).hexdigest()


def validate_snapshot(
    snapshot: dict[str, Any],
    plan: ResearchPlan,
    *,
    source_message_count: int,
) -> EvidenceCompleteness:
    errors: list[str] = []
    expected_active = plan.active_step_id
    observed_active = (snapshot.get("active_step") or {}).get("step_id")
    active_ok = expected_active == observed_active
    if not active_ok:
        errors.append("active step differs")

    expected_issues = {issue.issue_id for issue in plan.unresolved_issues if issue.status == "open"}
    observed_issues = {issue.get("issue_id") for issue in snapshot.get("unresolved_issues", [])}
    issues_ok = expected_issues == observed_issues
    if not issues_ok:
        errors.append("open issue set differs")

    expected_pins = {pin.pin_id for pin in plan.evidence_pins if pin.required}
    observed_pins = {pin.get("pin_id") for pin in snapshot.get("evidence_pins", [])}
    pins_ok = expected_pins.issubset(observed_pins)
    if not pins_ok:
        errors.append("required evidence pin set differs")
    for pin in plan.evidence_pins:
        if pin.required and pin.source_message_index is not None and pin.source_message_index >= source_message_count:
            errors.append(f"evidence pin {pin.pin_id} points outside source history")
    completed_by_id = {
        item.get("step_id"): item.get("evidence", []) for item in snapshot.get("completed_step_evidence", [])
    }
    for step in plan.steps:
        if step.status == "completed" and not completed_by_id.get(step.step_id):
            errors.append(f"completed step {step.step_id} lost its completion evidence")
    return EvidenceCompleteness(
        complete=not errors,
        preserved_active_step=active_ok,
        preserved_open_issues=issues_ok,
        preserved_evidence_pins=pins_ok,
        errors=tuple(errors),
    )
