"""Versioned contracts for Phase 3 context projection decisions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ContextContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ContextCompilerConfig(ContextContract):
    schema_version: Literal["echo.context-compiler-config.v1"] = "echo.context-compiler-config.v1"
    mode: Literal["shadow", "treatment"] = "shadow"
    keep_recent_messages: int = Field(default=6, ge=1)
    estimated_remaining_requests: int = Field(default=3, ge=1)
    summary_cost_tokens: int = Field(default=0, ge=0)
    cache_write_cost_tokens: int = Field(default=0, ge=0)
    cache_read_loss_tokens: int = Field(default=0, ge=0)
    minimum_net_savings_tokens: int = Field(default=1, ge=0)
    context_window_tokens: int = Field(default=128_000, ge=1)


class CompactionEconomics(ContextContract):
    schema_version: Literal["echo.compaction-economics.v1"] = "echo.compaction-economics.v1"
    source_tokens: int = Field(ge=0)
    candidate_tokens: int = Field(ge=0)
    estimated_remaining_requests: int = Field(ge=1)
    per_request_savings_tokens: int
    gross_future_savings_tokens: int
    summary_cost_tokens: int = Field(ge=0)
    cache_write_cost_tokens: int = Field(ge=0)
    cache_read_loss_tokens: int = Field(ge=0)
    total_compaction_cost_tokens: int = Field(ge=0)
    net_savings_tokens: int
    source_window_utilization: float = Field(ge=0)
    candidate_window_utilization: float = Field(ge=0)
    window_risk_reduction: float
    economically_eligible: bool


DecisionReason = Literal[
    "applied",
    "shadow_only",
    "evidence_incomplete",
    "state_mismatch",
    "no_reduction",
    "negative_economics",
    "compiler_error",
]


class CompactionDecision(ContextContract):
    schema_version: Literal["echo.compaction-decision.v1"] = "echo.compaction-decision.v1"
    applied: bool
    candidate_eligible: bool
    reason: DecisionReason
    detail: str


class ContextProjectionRecord(ContextContract):
    schema_version: Literal["echo.context-projection-record.v1"] = "echo.context-projection-record.v1"
    plan_id: str
    plan_revision: int = Field(ge=0)
    mode: Literal["shadow", "treatment"]
    source_message_count: int = Field(ge=0)
    candidate_message_count: int = Field(ge=0)
    delivered_message_count: int = Field(ge=0)
    source_tokens: int = Field(ge=0)
    candidate_tokens: int = Field(ge=0)
    delivered_tokens: int = Field(ge=0)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    delivered_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    snapshot_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    included_source_indices: tuple[int, ...] = ()
    dropped_source_indices: tuple[int, ...] = ()
    preserved_active_step: bool
    preserved_open_issues: bool
    preserved_evidence_pins: bool
    evidence_complete: bool
    state_equivalent: bool
    economics: CompactionEconomics
    decision: CompactionDecision

