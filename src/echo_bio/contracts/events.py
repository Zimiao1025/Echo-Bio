"""Typed Phase 0 lifecycle contracts for the Biomni harness boundary."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class Contract(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)


class RunIdentity(Contract):
    schema_version: Literal["echo.run-identity.v1"] = "echo.run-identity.v1"
    run_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class RunStart(Contract):
    schema_version: Literal["echo.run-start.v1"] = "echo.run-start.v1"
    identity: RunIdentity
    task: str
    occurred_at: datetime = Field(default_factory=utc_now)


class ModelRequestDraft(Contract):
    schema_version: Literal["echo.model-request-draft.v1"] = "echo.model-request-draft.v1"
    identity: RunIdentity
    messages: tuple[Any, ...]
    occurred_at: datetime = Field(default_factory=utc_now)


class ContextProjection(Contract):
    schema_version: Literal["echo.context-projection.v1"] = "echo.context-projection.v1"
    messages: tuple[Any, ...]
    policy_id: str = "identity"


class ModelResponseObserved(Contract):
    schema_version: Literal["echo.model-response-observed.v1"] = "echo.model-response-observed.v1"
    identity: RunIdentity
    response: Any
    requested_messages: tuple[Any, ...]
    occurred_at: datetime = Field(default_factory=utc_now)


class ExecutionDraft(Contract):
    schema_version: Literal["echo.execution-draft.v1"] = "echo.execution-draft.v1"
    identity: RunIdentity
    language: Literal["python", "r", "bash", "cli"]
    code: str
    occurred_at: datetime = Field(default_factory=utc_now)


class ExecutionDecision(Contract):
    schema_version: Literal["echo.execution-decision.v1"] = "echo.execution-decision.v1"
    code: str
    allow: bool = True
    policy_id: str = "identity"


class RawExecutionObserved(Contract):
    schema_version: Literal["echo.raw-execution-observed.v1"] = "echo.raw-execution-observed.v1"
    identity: RunIdentity
    language: Literal["python", "r", "bash", "cli"]
    code: str
    raw_result: str
    occurred_at: datetime = Field(default_factory=utc_now)


class ObservationProjection(Contract):
    schema_version: Literal["echo.observation-projection.v1"] = "echo.observation-projection.v1"
    result: str
    policy_id: str = "identity"


class PlanUpdated(Contract):
    schema_version: Literal["echo.plan-updated.v1"] = "echo.plan-updated.v1"
    identity: RunIdentity
    plan: dict[str, Any]
    source: str
    occurred_at: datetime = Field(default_factory=utc_now)


class OutcomeDraft(Contract):
    schema_version: Literal["echo.outcome-draft.v1"] = "echo.outcome-draft.v1"
    identity: RunIdentity
    content: str
    occurred_at: datetime = Field(default_factory=utc_now)


class FinalizationDecision(Contract):
    schema_version: Literal["echo.finalization-decision.v1"] = "echo.finalization-decision.v1"
    content: str
    allow: bool = True
    policy_id: str = "identity"


class RunEnded(Contract):
    schema_version: Literal["echo.run-ended.v1"] = "echo.run-ended.v1"
    identity: RunIdentity
    status: Literal["completed", "failed", "cancelled"]
    final_content: str | None = None
    occurred_at: datetime = Field(default_factory=utc_now)


class HookFailure(Contract):
    schema_version: Literal["echo.hook-failure.v1"] = "echo.hook-failure.v1"
    hook_name: str
    error_type: str
    error_message: str
    identity: RunIdentity | None = None
    occurred_at: datetime = Field(default_factory=utc_now)
