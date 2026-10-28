"""Typed research-plan contracts and deterministic transition validation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PlanContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CompletionEvidence(PlanContract):
    schema_version: Literal["echo.completion-evidence.v1"] = "echo.completion-evidence.v1"
    evidence_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    kind: Literal["artifact", "execution", "table", "citation", "verifier"]
    ref: str = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    description: str = Field(min_length=1)


class EvidencePin(PlanContract):
    schema_version: Literal["echo.evidence-pin.v1"] = "echo.evidence-pin.v1"
    pin_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    label: str = Field(min_length=1)
    ref: str = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    reason: str = Field(min_length=1)
    source_message_index: int | None = Field(default=None, ge=0)
    required: bool = True


class UnresolvedIssue(PlanContract):
    schema_version: Literal["echo.unresolved-issue.v1"] = "echo.unresolved-issue.v1"
    issue_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    description: str = Field(min_length=1)
    severity: Literal["low", "medium", "high", "critical"]
    status: Literal["open", "resolved"] = "open"
    resolution_evidence: tuple[CompletionEvidence, ...] = ()

    @model_validator(mode="after")
    def validate_resolution(self) -> UnresolvedIssue:
        if self.status == "resolved" and not self.resolution_evidence:
            raise ValueError("resolved issue requires resolution evidence")
        return self


StepStatus = Literal["pending", "in_progress", "completed", "failed", "blocked"]


class ResearchStep(PlanContract):
    schema_version: Literal["echo.research-step.v1"] = "echo.research-step.v1"
    step_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    title: str = Field(min_length=1)
    status: StepStatus = "pending"
    dependencies: tuple[str, ...] = ()
    required_evidence_kinds: tuple[
        Literal["artifact", "execution", "table", "citation", "verifier"], ...
    ] = ()
    completion_evidence: tuple[CompletionEvidence, ...] = ()
    notes: str | None = None

    @model_validator(mode="after")
    def validate_completion(self) -> ResearchStep:
        if self.status == "completed":
            if not self.completion_evidence:
                raise ValueError("completed step requires completion evidence")
            supplied = {item.kind for item in self.completion_evidence}
            missing = set(self.required_evidence_kinds) - supplied
            if missing:
                raise ValueError(f"completed step is missing required evidence kinds: {sorted(missing)}")
        if self.status in {"failed", "blocked"} and not self.notes:
            raise ValueError(f"{self.status} step requires notes")
        return self


class ResearchPlan(PlanContract):
    schema_version: Literal["echo.research-plan.v1"] = "echo.research-plan.v1"
    plan_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    run_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    objective: str = Field(min_length=1)
    revision: int = Field(default=0, ge=0)
    steps: tuple[ResearchStep, ...] = Field(min_length=1)
    active_step_id: str | None = None
    evidence_pins: tuple[EvidencePin, ...] = ()
    unresolved_issues: tuple[UnresolvedIssue, ...] = ()

    @model_validator(mode="after")
    def validate_graph(self) -> ResearchPlan:
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("step IDs must be unique")
        step_map = {step.step_id: step for step in self.steps}
        for step in self.steps:
            missing = set(step.dependencies) - set(step_map)
            if missing:
                raise ValueError(f"step {step.step_id} has unknown dependencies: {sorted(missing)}")
            if step.step_id in step.dependencies:
                raise ValueError("step cannot depend on itself")
            if step.status in {"in_progress", "completed"}:
                incomplete = [dep for dep in step.dependencies if step_map[dep].status != "completed"]
                if incomplete:
                    raise ValueError(f"step {step.step_id} has incomplete dependencies: {incomplete}")
        self._validate_acyclic(step_map)
        active = [step.step_id for step in self.steps if step.status == "in_progress"]
        if len(active) > 1:
            raise ValueError("at most one step may be in progress")
        if self.active_step_id != (active[0] if active else None):
            raise ValueError("active_step_id must match the in-progress step")
        for label, values in (
            ("evidence pin", [pin.pin_id for pin in self.evidence_pins]),
            ("unresolved issue", [issue.issue_id for issue in self.unresolved_issues]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} IDs must be unique")
        return self

    @staticmethod
    def _validate_acyclic(step_map: dict[str, ResearchStep]) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visiting:
                raise ValueError(f"research plan contains a dependency cycle at {step_id}")
            if step_id in visited:
                return
            visiting.add(step_id)
            for dependency in step_map[step_id].dependencies:
                visit(dependency)
            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in step_map:
            visit(step_id)


class PlanTransition(PlanContract):
    schema_version: Literal["echo.plan-transition.v1"] = "echo.plan-transition.v1"
    step_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    expected_revision: int = Field(ge=0)
    target_status: Literal["in_progress", "completed", "failed", "blocked"]
    completion_evidence: tuple[CompletionEvidence, ...] = ()
    notes: str | None = None


class PlanTransitionError(ValueError):
    """A requested plan update violates the frozen state machine."""


class ResearchPlanStateMachine:
    ALLOWED: dict[StepStatus, set[StepStatus]] = {
        "pending": {"in_progress", "blocked"},
        "in_progress": {"completed", "failed", "blocked"},
        "failed": {"in_progress"},
        "blocked": {"in_progress"},
        "completed": set(),
    }

    def apply(self, plan: ResearchPlan, transition: PlanTransition) -> ResearchPlan:
        if transition.expected_revision != plan.revision:
            raise PlanTransitionError(
                f"stale plan revision {transition.expected_revision}; current revision is {plan.revision}"
            )
        step_map = {step.step_id: step for step in plan.steps}
        if transition.step_id not in step_map:
            raise PlanTransitionError(f"unknown step: {transition.step_id}")
        current = step_map[transition.step_id]
        if transition.target_status not in self.ALLOWED[current.status]:
            raise PlanTransitionError(f"invalid transition: {current.status} -> {transition.target_status}")
        if transition.target_status == "in_progress":
            if plan.active_step_id not in {None, current.step_id}:
                raise PlanTransitionError(f"another step is already active: {plan.active_step_id}")
            incomplete = [dep for dep in current.dependencies if step_map[dep].status != "completed"]
            if incomplete:
                raise PlanTransitionError(f"cannot start before dependencies complete: {incomplete}")
        if transition.target_status == "completed" and not transition.completion_evidence:
            raise PlanTransitionError("completion requires evidence")
        if transition.target_status in {"failed", "blocked"} and not transition.notes:
            raise PlanTransitionError(f"{transition.target_status} transition requires notes")
        updated_payload = current.model_dump(mode="python")
        updated_payload.update(
            status=transition.target_status,
            completion_evidence=transition.completion_evidence,
            notes=transition.notes,
        )
        try:
            updated = ResearchStep.model_validate(updated_payload)
        except ValueError as error:
            raise PlanTransitionError(str(error)) from error
        steps = tuple(updated if step.step_id == updated.step_id else step for step in plan.steps)
        active_step_id = updated.step_id if updated.status == "in_progress" else None
        try:
            payload = plan.model_dump(mode="python")
            payload.update(
                revision=plan.revision + 1,
                steps=steps,
                active_step_id=active_step_id,
            )
            return ResearchPlan.model_validate(payload)
        except ValueError as error:
            raise PlanTransitionError(str(error)) from error
