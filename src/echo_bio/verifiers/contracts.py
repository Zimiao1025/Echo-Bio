"""Closed contracts shared by L0-L3 scientific verifiers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from echo_bio.artifacts.contracts import ArtifactRef
from echo_bio.contracts.plans import ResearchPlan
from echo_bio.evidence.contracts import ClaimEvidenceGraph, ScientificExecutionReceipt


class VerifierContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


Domain = Literal["scrna", "bulk_rnaseq", "gwas"]
VerificationStatus = Literal["pass", "warn", "fail", "uncertain", "not_assessed"]


class CleanReplayManifest(VerifierContract):
    schema_version: Literal["echo.clean-replay-manifest.v1"] = "echo.clean-replay-manifest.v1"
    run_id: str = Field(min_length=1)
    code_artifacts: tuple[ArtifactRef, ...] = Field(min_length=1)
    input_artifacts: tuple[ArtifactRef, ...] = Field(min_length=1)
    output_artifacts: tuple[ArtifactRef, ...] = Field(min_length=1)
    environment_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    data_manifest_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    random_seeds: tuple[int, ...] = ()
    replay_command: tuple[str, ...] = Field(min_length=1)
    network_required: bool = False


class VerificationBundle(VerifierContract):
    schema_version: Literal["echo.verification-bundle.v1"] = "echo.verification-bundle.v1"
    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    domain: Domain
    split: Literal["pilot", "search_dev", "private", "external"]
    data_manifest_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    environment_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_artifact: ArtifactRef
    analysis_artifact: ArtifactRef
    claim_graph: ClaimEvidenceGraph
    scientific_receipts: tuple[ScientificExecutionReceipt, ...] = Field(min_length=1)
    research_plan: ResearchPlan
    replay_manifest: CleanReplayManifest


class VerificationFinding(VerifierContract):
    schema_version: Literal["echo.verification-finding.v1"] = "echo.verification-finding.v1"
    verifier_id: str = Field(min_length=1)
    level: Literal["L0", "L1", "L2", "L3"]
    domain: Literal["generic", "scrna", "bulk_rnaseq", "gwas"]
    criterion_id: str = Field(min_length=1)
    status: VerificationStatus
    score: float | None = Field(default=None, ge=0, le=1)
    rubric_weight: float = Field(default=0, ge=0, le=1)
    coverage: Literal["full", "partial", "none"]
    hard_gate: bool
    automated: bool
    message: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = ()


class ClaimAssessment(VerifierContract):
    schema_version: Literal["echo.claim-assessment.v1"] = "echo.claim-assessment.v1"
    claim_id: str = Field(pattern=r"^claim_[a-f0-9]{64}$")
    status: Literal["supported", "limited", "refuted", "uncertain", "unlinked", "overreach"]
    locator_ids: tuple[str, ...] = ()
    evidence_excerpts: tuple[str, ...] = ()
    conflict_detected: bool = False
    causal_overreach: bool = False
    human_review_required: bool
    rationale: str = Field(min_length=1)


class RubricAssessment(VerifierContract):
    schema_version: Literal["echo.rubric-assessment.v1"] = "echo.rubric-assessment.v1"
    criterion_id: Literal[
        "method_selection",
        "biological_interpretation",
        "scientific_reasoning",
        "source_reliability",
    ]
    judge_id: str = Field(min_length=1)
    generator_id: str = Field(min_length=1)
    score: float | None = Field(default=None, ge=0, le=1)
    status: VerificationStatus
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence_refs: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_independence(self) -> RubricAssessment:
        if self.judge_id == self.generator_id:
            raise ValueError("rubric judge must be independent from the generator")
        return self


class AuditRoutingDecision(VerifierContract):
    schema_version: Literal["echo.audit-routing-decision.v1"] = "echo.audit-routing-decision.v1"
    required: bool
    reasons: tuple[
        Literal["private_sample", "near_threshold", "judge_disagreement", "causal_claim", "uncertain_claim"],
        ...,
    ] = ()
    blind_review: bool
    minimum_reviewers: int = Field(default=0, ge=0)


class VerifierReport(VerifierContract):
    schema_version: Literal["echo.verifier-report.v1"] = "echo.verifier-report.v1"
    run_id: str
    task_id: str
    domain: Domain
    findings: tuple[VerificationFinding, ...]
    claim_assessments: tuple[ClaimAssessment, ...]
    rubric_assessments: tuple[RubricAssessment, ...]
    audit_routing: AuditRoutingDecision
    hard_gate_passed: bool
    weighted_score: float = Field(ge=0, le=1)
    l0_l1_covered_weight: float = Field(ge=0, le=1)
    total_positive_rubric_weight: float = Field(default=1.0, gt=0)
    l0_l1_coverage_fraction: float = Field(ge=0, le=1)
    important_claim_l2_coverage: float = Field(ge=0, le=1)
    scientific_truth_established: Literal[False] = False


class CapabilityDecision(VerifierContract):
    schema_version: Literal["echo.capability-decision.v1"] = "echo.capability-decision.v1"
    run_id: str
    decision: Literal["pass", "fail", "uncertain"]
    hard_gate_passed: bool
    coverage_passed: bool
    score_passed: bool
    human_review_required: bool
    reasons: tuple[str, ...]
    gate_config_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class GateConfig(VerifierContract):
    schema_version: Literal["echo.gate-config.v1"] = "echo.gate-config.v1"
    config_id: str = "gate-config-v1"
    minimum_l0_l1_coverage: float = Field(default=0.60, ge=0, le=1)
    minimum_weighted_score: float = Field(default=0.75, ge=0, le=1)
    soft_noninferiority_margin: float = Field(default=0.03, ge=0, le=1)
    uncertain_policy: Literal["fail", "human_review"] = "human_review"
    private_blind_audit_fraction: float = Field(default=0.20, ge=0.20, le=1)
    forced_audit_conditions: tuple[str, ...] = (
        "near_threshold",
        "judge_disagreement",
        "causal_claim",
        "uncertain_claim",
    )
    hard_gate_levels: tuple[Literal["L0", "L1"], ...] = ("L0", "L1")
    verifier_and_gate_candidate_writable: Literal[False] = False
