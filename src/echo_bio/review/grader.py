"""Programmatic evidence-readiness checks before independent scientific review."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from echo_bio.artifacts import ArtifactRef, LocalArtifactStore
from echo_bio.contracts.trace import TraceEvent
from echo_bio.review.contracts import ReadinessCriterion, ScientificReviewPacket

NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[-+]?\d+)?%?", re.IGNORECASE)
LIMITATION_PATTERN = re.compile(
    r"(?i)\b(limit(?:ation|ed|s)?|cannot|uncertain|small sample|synthetic|not (?:establish|infer|prove)|"
    r"requires? (?:validation|replication)|caution)\b"
)

DOMAIN_CONCEPTS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "scrna": (
        ("biological replicate", ("donor", "replicate")),
        ("identifier alignment", ("cell", "donor")),
        ("QC threshold", ("threshold", "qc")),
    ),
    "bulk_rnaseq": (
        ("biological replicate", ("sample", "replicate")),
        ("design/contrast", ("design", "contrast", "condition")),
        ("effect/multiplicity", ("effect", "fold", "fdr", "multiple", "adjust")),
    ),
    "gwas": (
        ("allele alignment", ("allele", "effect direction", "beta")),
        ("QC threshold", ("threshold", "qc")),
        ("non-causal interpretation", ("association", "caus", "cannot")),
    ),
    "proteomics": (
        ("protein identifier alignment", ("protein", "identifier")),
        ("dual threshold", ("log2", "fold", "q_value", "q-value", "threshold")),
        ("non-causal interpretation", ("differential abundance", "caus", "cannot")),
    ),
}


class ScientificReadinessGrader:
    """Check durable evidence closure without claiming scientific correctness."""

    def __init__(self, store: LocalArtifactStore) -> None:
        self.store = store

    def grade(
        self,
        *,
        run_id: str,
        task: dict,
        outcome: str,
        outcome_ref: ArtifactRef,
        code_ref: ArtifactRef,
        execution_refs: Sequence[ArtifactRef],
        trace_events: Sequence[TraceEvent],
    ) -> ScientificReviewPacket:
        refs = (outcome_ref, code_ref, *execution_refs)
        integrity_errors: list[str] = []
        for ref in refs:
            try:
                self.store.verify(ref)
            except Exception as error:  # pragma: no cover - exercised through store contract tests
                integrity_errors.append(f"{ref.artifact_id}:{type(error).__name__}")

        criteria = [
            ReadinessCriterion(
                criterion_id="durable_artifact_integrity",
                status="pass" if not integrity_errors else "fail",
                score=1.0 if not integrity_errors else 0.0,
                hard_gate=True,
                details=(
                    f"Verified {len(refs)} content-addressed artifacts by hash and size."
                    if not integrity_errors
                    else f"Artifact verification failures: {', '.join(integrity_errors)}"
                ),
                evidence_refs=tuple(ref.uri for ref in refs),
            ),
            self._execution_criterion(execution_refs, trace_events),
            self._trace_consistency_criterion(outcome, code_ref, execution_refs),
            self._quantitative_criterion(outcome, outcome_ref),
            self._limitation_criterion(outcome, outcome_ref),
            self._invariant_criterion(task, outcome, outcome_ref),
        ]
        scored = [criterion.score for criterion in criteria if criterion.score is not None]
        readiness_score = sum(scored) / len(scored) if scored else 0.0
        hard_gate_failed = any(criterion.hard_gate and criterion.status == "fail" for criterion in criteria)
        return ScientificReviewPacket(
            run_id=run_id,
            task_id=str(task["task_id"]),
            domain=task["domain"],
            manifest_sha256=str(task["data"]["sha256"]),
            outcome_artifact=outcome_ref,
            code_artifact=code_ref,
            execution_artifacts=tuple(execution_refs),
            criteria=tuple(criteria),
            readiness_score=readiness_score,
            ready_for_blinded_review=not hard_gate_failed,
        )

    def _trace_consistency_criterion(
        self,
        outcome: str,
        code_ref: ArtifactRef,
        execution_refs: Sequence[ArtifactRef],
    ) -> ReadinessCriterion:
        forbidden = re.findall(r"</?(?:execute|observation)\b", outcome, re.IGNORECASE)
        try:
            code_bundle = json.loads(self.store.get_text(code_ref))
            code_count = len(code_bundle.get("executions", []))
        except (ValueError, TypeError, AttributeError):
            code_count = -1
        actual_count = len(execution_refs)
        aligned = not forbidden and code_count == actual_count and actual_count > 0
        details = (
            f"Final answer is action-pure and {code_count} archived code actions align with "
            f"{actual_count} execution observations."
            if aligned
            else f"Action/evidence mismatch: forbidden final-answer control tags={len(forbidden)}, "
            f"code actions={code_count}, execution observations={actual_count}."
        )
        return ReadinessCriterion(
            criterion_id="action_execution_trace_consistency",
            status="pass" if aligned else "fail",
            score=1.0 if aligned else 0.0,
            hard_gate=True,
            details=details,
            evidence_refs=(code_ref.uri, *(ref.uri for ref in execution_refs)),
        )

    @staticmethod
    def _execution_criterion(
        execution_refs: Sequence[ArtifactRef], trace_events: Sequence[TraceEvent]
    ) -> ReadinessCriterion:
        finished = [event for event in trace_events if event.event_type == "execution_finished"]
        unresolved = [event for event in finished if event.error is not None]
        if not finished or not execution_refs:
            return ReadinessCriterion(
                criterion_id="executed_evidence_present",
                status="fail",
                score=0.0,
                hard_gate=True,
                details="No durable execution observation is attached to the outcome.",
            )
        status = "warn" if unresolved else "pass"
        score = 0.5 if unresolved else 1.0
        details = (
            f"Archived {len(execution_refs)} execution observations; {len(unresolved)} were marked as errors."
            if unresolved
            else f"Archived {len(execution_refs)} execution observations with no telemetry-marked error."
        )
        return ReadinessCriterion(
            criterion_id="executed_evidence_present",
            status=status,
            score=score,
            hard_gate=True,
            details=details,
            evidence_refs=tuple(ref.uri for ref in execution_refs),
        )

    @staticmethod
    def _quantitative_criterion(outcome: str, outcome_ref: ArtifactRef) -> ReadinessCriterion:
        values = NUMBER_PATTERN.findall(outcome)
        passed = len(values) >= 3
        return ReadinessCriterion(
            criterion_id="quantitative_result_presence",
            status="pass" if passed else "warn",
            score=1.0 if passed else 0.0,
            details=f"Detected {len(values)} numeric values in the final answer; this does not validate them.",
            evidence_refs=(outcome_ref.uri,),
        )

    @staticmethod
    def _limitation_criterion(outcome: str, outcome_ref: ArtifactRef) -> ReadinessCriterion:
        passed = bool(LIMITATION_PATTERN.search(outcome))
        return ReadinessCriterion(
            criterion_id="limitations_disclosed",
            status="pass" if passed else "fail",
            score=1.0 if passed else 0.0,
            details=(
                "The final answer contains an explicit limitation/caution statement."
                if passed
                else "The final answer contains no detectable limitation/caution statement."
            ),
            evidence_refs=(outcome_ref.uri,),
        )

    @staticmethod
    def _invariant_criterion(task: dict, outcome: str, outcome_ref: ArtifactRef) -> ReadinessCriterion:
        normalized = outcome.casefold()
        configured = task.get("readiness_concepts")
        concepts = (
            tuple((str(item["name"]), tuple(str(term) for term in item["terms"])) for item in configured)
            if configured
            else DOMAIN_CONCEPTS[str(task["domain"])]
        )
        covered = [name for name, terms in concepts if any(term in normalized for term in terms)]
        coverage = len(covered) / len(concepts)
        status = "pass" if coverage == 1 else "warn" if coverage > 0 else "fail"
        return ReadinessCriterion(
            criterion_id="manifest_invariant_mentions",
            status=status,
            score=coverage,
            details=(
                f"Detected textual coverage for {len(covered)}/{len(concepts)} domain invariant groups: "
                f"{', '.join(covered) if covered else 'none'}. Textual coverage is not semantic validation."
            ),
            evidence_refs=(outcome_ref.uri,),
        )
