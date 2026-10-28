"""Versioned contracts for deterministic scientific pre-review."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from echo_bio.artifacts import ArtifactRef


class ReviewContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReadinessCriterion(ReviewContract):
    schema_version: Literal["echo.readiness-criterion.v1"] = "echo.readiness-criterion.v1"
    criterion_id: str = Field(min_length=1)
    status: Literal["pass", "warn", "fail", "not_assessed"]
    score: float | None = Field(default=None, ge=0, le=1)
    hard_gate: bool = False
    details: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = ()
    automated: bool = True


class ScientificReviewPacket(ReviewContract):
    """Reviewable evidence package; it is not a scientific truth certificate."""

    schema_version: Literal["echo.scientific-review-packet.v1"] = "echo.scientific-review-packet.v1"
    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    domain: Literal["scrna", "bulk_rnaseq", "gwas", "proteomics"]
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    outcome_artifact: ArtifactRef
    code_artifact: ArtifactRef
    execution_artifacts: tuple[ArtifactRef, ...]
    criteria: tuple[ReadinessCriterion, ...]
    readiness_score: float = Field(ge=0, le=1)
    ready_for_blinded_review: bool
    scientific_correctness_assessed: Literal[False] = False
    required_independent_review: tuple[str, ...] = (
        "biological validity",
        "statistical validity",
        "claim-evidence entailment",
        "task-specific rubric satisfaction",
    )
