"""Versioned contracts for append-only Phase 1 research traces."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class TraceContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


DataClass = Literal["public", "internal", "controlled", "secret"]
Split = Literal["pilot", "search_dev", "private", "external"]
TraceEventType = Literal[
    "run_started",
    "resource_retrieval",
    "shared_resource_receipt_bound",
    "model_request",
    "model_response",
    "execution_started",
    "execution_finished",
    "plan_updated",
    "evidence_state_updated",
    "termination_decided",
    "harness_policy_installed",
    "finalization",
    "verifier_finished",
    "hook_error",
    "error",
    "run_ended",
]


class RunEnvelope(TraceContract):
    schema_version: Literal["echo.run.v1"] = "echo.run.v1"
    run_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    split: Split
    harness_candidate_id: str = Field(min_length=1)
    model_snapshot: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    environment_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    data_manifest_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    data_class: DataClass = "internal"
    started_at: datetime = Field(default_factory=utc_now)
    parent_run_id: str | None = None


class ProviderUsage(TraceContract):
    schema_version: Literal["echo.provider-usage.v1"] = "echo.provider-usage.v1"
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_read_tokens: int | None = Field(default=None, ge=0)
    cache_write_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    source: Literal["provider", "local_estimate", "unavailable"] = "unavailable"


class CostRecord(TraceContract):
    schema_version: Literal["echo.cost.v1"] = "echo.cost.v1"
    amount: float | None = Field(default=None, ge=0)
    currency: str = "USD"
    price_table_version: str | None = None
    priced: bool = False


class TraceError(TraceContract):
    schema_version: Literal["echo.trace-error.v1"] = "echo.trace-error.v1"
    error_type: str
    message: str
    retryable: bool | None = None


class RedactionMetadata(TraceContract):
    schema_version: Literal["echo.redaction.v1"] = "echo.redaction.v1"
    data_class: DataClass
    rules_applied: tuple[str, ...] = ()
    fields_withheld: int = Field(default=0, ge=0)
    secret_matches: int = Field(default=0, ge=0)
    original_character_count: int = Field(default=0, ge=0)


class TokenComponent(TraceContract):
    schema_version: Literal["echo.token-component.v1"] = "echo.token-component.v1"
    component: Literal["system", "tool_resource", "plan", "history", "observation", "evidence"]
    message_index: int = Field(ge=0)
    role: str
    token_count: int = Field(ge=0)
    character_count: int = Field(ge=0)
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class TraceEvent(TraceContract):
    schema_version: Literal["echo.trace-event.v1"] = "echo.trace-event.v1"
    event_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    sequence_no: int = Field(ge=0)
    event_type: TraceEventType
    occurred_at: datetime = Field(default_factory=utc_now)
    input_refs: tuple[str, ...] = ()
    output_refs: tuple[str, ...] = ()
    provider_usage: ProviderUsage | None = None
    local_token_estimates: dict[str, int] = Field(default_factory=dict)
    cost: CostRecord | None = None
    error: TraceError | None = None
    redaction: RedactionMetadata
    payload: dict[str, Any] = Field(default_factory=dict)
    payload_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    previous_event_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    event_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("local_token_estimates")
    @classmethod
    def validate_token_estimates(cls, value: dict[str, int]) -> dict[str, int]:
        if any(tokens < 0 for tokens in value.values()):
            raise ValueError("token estimates must be non-negative")
        return value
