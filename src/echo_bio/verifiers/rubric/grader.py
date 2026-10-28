"""Independent rubric-judge boundary and mandatory expert-audit routing."""

from __future__ import annotations

from collections.abc import Mapping

from echo_bio.evidence.contracts import ClaimEvidenceGraph
from echo_bio.verifiers.contracts import (
    AuditRoutingDecision,
    ClaimAssessment,
    RubricAssessment,
    VerificationBundle,
)

RUBRIC_CRITERIA = (
    "method_selection",
    "biological_interpretation",
    "scientific_reasoning",
    "source_reliability",
)


class RecordedRubricJudge:
    """Offline judge fixture implementing the same independent-judge contract as a model gateway."""

    def __init__(self, *, judge_id: str = "recorded-independent-judge-v1") -> None:
        self.judge_id = judge_id

    def assess(
        self,
        *,
        generator_id: str,
        scores: Mapping[str, float | None],
        evidence_refs: tuple[str, ...],
    ) -> tuple[RubricAssessment, ...]:
        assessments = []
        for criterion in RUBRIC_CRITERIA:
            score = scores.get(criterion)
            if score is None:
                status = "not_assessed"
                confidence = None
                rationale = "Recorded judge fixture has no assessment for this criterion."
            elif score >= 0.8:
                status = "pass"
                confidence = 0.85
                rationale = "Recorded independent judge rates the criterion above the pass threshold."
            elif score >= 0.6:
                status = "warn"
                confidence = 0.65
                rationale = "Recorded independent judge places the criterion in the review band."
            else:
                status = "fail"
                confidence = 0.8
                rationale = "Recorded independent judge rates the criterion below the review threshold."
            assessments.append(
                RubricAssessment(
                    criterion_id=criterion,
                    judge_id=self.judge_id,
                    generator_id=generator_id,
                    score=score,
                    status=status,
                    confidence=confidence,
                    evidence_refs=evidence_refs,
                    rationale=rationale,
                )
            )
        return tuple(assessments)


class AuditRouter:
    def __init__(self, *, threshold: float = 0.75, near_margin: float = 0.05) -> None:
        self.threshold = threshold
        self.near_margin = near_margin

    def route(
        self,
        *,
        bundle: VerificationBundle,
        graph: ClaimEvidenceGraph,
        claims: tuple[ClaimAssessment, ...],
        rubric: tuple[RubricAssessment, ...],
        private_attempt_index: int | None = None,
        expert_labels: Mapping[str, str] | None = None,
    ) -> AuditRoutingDecision:
        reasons = set()
        if bundle.split == "private" and private_attempt_index is not None and private_attempt_index % 5 == 0:
            reasons.add("private_sample")
        if any(claim.claim_type == "causal" for claim in graph.claims):
            reasons.add("causal_claim")
        if any(item.status in {"uncertain", "unlinked", "overreach"} for item in claims):
            reasons.add("uncertain_claim")
        scored = [item.score for item in rubric if item.score is not None]
        average = sum(scored) / len(scored) if scored else None
        if average is not None and abs(average - self.threshold) <= self.near_margin:
            reasons.add("near_threshold")
        if expert_labels and any(
            expert_labels.get(item.criterion_id) not in {None, item.status} for item in rubric
        ):
            reasons.add("judge_disagreement")
        ordered = tuple(
            reason
            for reason in (
                "private_sample",
                "near_threshold",
                "judge_disagreement",
                "causal_claim",
                "uncertain_claim",
            )
            if reason in reasons
        )
        required = bool(ordered)
        two_reviewers = bool(reasons.intersection({"judge_disagreement", "causal_claim"}))
        return AuditRoutingDecision(
            required=required,
            reasons=ordered,
            blind_review=required,
            minimum_reviewers=2 if two_reviewers else 1 if required else 0,
        )


def calibration_metrics(judge_labels: list[bool], expert_labels: list[bool]) -> dict[str, float | int]:
    if len(judge_labels) != len(expert_labels) or not judge_labels:
        raise ValueError("judge and expert calibration labels must be non-empty and aligned")
    tp = sum(judge and expert for judge, expert in zip(judge_labels, expert_labels, strict=True))
    tn = sum(not judge and not expert for judge, expert in zip(judge_labels, expert_labels, strict=True))
    fp = sum(judge and not expert for judge, expert in zip(judge_labels, expert_labels, strict=True))
    fn = sum(not judge and expert for judge, expert in zip(judge_labels, expert_labels, strict=True))
    negatives = tn + fp
    positives = tp + fn
    return {
        "count": len(judge_labels),
        "agreement": (tp + tn) / len(judge_labels),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "false_positive_rate": fp / negatives if negatives else 0.0,
        "false_negative_rate": fn / positives if positives else 0.0,
    }
