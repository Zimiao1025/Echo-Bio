"""Route one verification bundle through L0, domain L1, claim L2, and rubric L3."""

from __future__ import annotations

from collections.abc import Mapping

from echo_bio.artifacts.store import LocalArtifactStore
from echo_bio.verifiers.base import finding
from echo_bio.verifiers.bulk_rnaseq import BulkRnaSeqVerifier
from echo_bio.verifiers.claim_entailment import ClaimVerifier
from echo_bio.verifiers.contracts import VerificationBundle, VerifierReport
from echo_bio.verifiers.generic import GenericProvenanceVerifier
from echo_bio.verifiers.gwas import GwasVerifier
from echo_bio.verifiers.rubric import AuditRouter, RecordedRubricJudge
from echo_bio.verifiers.scrna import ScrnaVerifier


class VerifierRouter:
    def __init__(
        self,
        store: LocalArtifactStore,
        *,
        rubric_judge: RecordedRubricJudge | None = None,
        audit_router: AuditRouter | None = None,
    ) -> None:
        self.store = store
        self.l0 = GenericProvenanceVerifier(store)
        self.l1 = {
            "scrna": ScrnaVerifier(store),
            "bulk_rnaseq": BulkRnaSeqVerifier(store),
            "gwas": GwasVerifier(store),
        }
        self.l2 = ClaimVerifier(store)
        self.rubric_judge = rubric_judge or RecordedRubricJudge()
        self.audit_router = audit_router or AuditRouter()

    def verify(
        self,
        bundle: VerificationBundle,
        *,
        generator_id: str,
        rubric_scores: Mapping[str, float | None],
        private_attempt_index: int | None = None,
        expert_labels: Mapping[str, str] | None = None,
    ) -> VerifierReport:
        l0_findings = self.l0.verify(bundle)
        l1_findings = self.l1[bundle.domain].verify(bundle)
        claim_assessments, l2_findings = self.l2.verify(bundle.claim_graph)
        rubric = self.rubric_judge.assess(
            generator_id=generator_id,
            scores=rubric_scores,
            evidence_refs=(bundle.analysis_artifact.uri,),
        )
        l3_findings = tuple(
            finding(
                verifier_id=item.judge_id,
                level="L3",
                domain="generic",
                criterion_id=item.criterion_id,
                passed=None if item.status == "not_assessed" else item.status in {"pass", "warn"},
                weight=0.15 / len(rubric),
                hard_gate=False,
                message=item.rationale,
                evidence_refs=item.evidence_refs,
                automated=False,
                warn=item.status == "warn",
            )
            for item in rubric
        )
        audit = self.audit_router.route(
            bundle=bundle,
            graph=bundle.claim_graph,
            claims=claim_assessments,
            rubric=rubric,
            private_attempt_index=private_attempt_index,
            expert_labels=expert_labels,
        )
        findings = (*l0_findings, *l1_findings, *l2_findings, *l3_findings)
        hard_pass = not any(item.hard_gate and item.status == "fail" for item in findings)
        scored_weight = sum(item.rubric_weight for item in findings if item.score is not None)
        weighted_score = (
            sum(item.rubric_weight * item.score for item in findings if item.score is not None) / scored_weight
            if scored_weight
            else 0.0
        )
        l0_l1_covered = sum(
            item.rubric_weight for item in findings if item.level in {"L0", "L1"} and item.coverage == "full"
        )
        important_ids = {
            claim.claim_id for claim in bundle.claim_graph.claims if claim.importance == "important"
        }
        assessed_important = {item.claim_id for item in claim_assessments if item.claim_id in important_ids}
        important_coverage = len(assessed_important) / len(important_ids) if important_ids else 1.0
        return VerifierReport(
            run_id=bundle.run_id,
            task_id=bundle.task_id,
            domain=bundle.domain,
            findings=findings,
            claim_assessments=claim_assessments,
            rubric_assessments=rubric,
            audit_routing=audit,
            hard_gate_passed=hard_pass,
            weighted_score=weighted_score,
            l0_l1_covered_weight=l0_l1_covered,
            total_positive_rubric_weight=1.0,
            l0_l1_coverage_fraction=l0_l1_covered,
            important_claim_l2_coverage=important_coverage,
        )
