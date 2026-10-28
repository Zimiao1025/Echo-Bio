"""Locator-grounded L2 checks with explicit uncertainty and causal-overreach detection."""

from __future__ import annotations

import re

from echo_bio.artifacts.store import LocalArtifactStore
from echo_bio.evidence.contracts import ClaimEvidenceGraph
from echo_bio.verifiers.base import finding
from echo_bio.verifiers.contracts import ClaimAssessment, VerificationFinding
from echo_bio.verifiers.locators import read_locator

NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[-+]?\d+)?", re.IGNORECASE)
BOUNDED_PATTERN = re.compile(
    r"(?i)(not (?:established|proven|supported)|cannot establish|uncertain|hypothesis|association only|"
    r"does not establish|requires validation)"
)


class ClaimVerifier:
    verifier_id = "claim-locator-v1"

    def __init__(self, store: LocalArtifactStore) -> None:
        self.store = store

    def verify(
        self, graph: ClaimEvidenceGraph
    ) -> tuple[tuple[ClaimAssessment, ...], tuple[VerificationFinding, ...]]:
        locator_map = {locator.locator_id: locator for locator in graph.locators}
        links_by_claim = {
            claim.claim_id: [link for link in graph.links if link.claim_id == claim.claim_id] for claim in graph.claims
        }
        important_count = max(1, sum(claim.importance == "important" for claim in graph.claims))
        assessments = []
        findings = []
        for claim in graph.claims:
            links = links_by_claim[claim.claim_id]
            assessment = self._assess(claim.text, claim.claim_type, links, locator_map)
            assessment = assessment.model_copy(update={"claim_id": claim.claim_id})
            assessments.append(assessment)
            passed = assessment.status in {"supported", "limited"}
            uncertain = assessment.status == "uncertain"
            hard_gate = claim.importance == "important" and assessment.status in {
                "refuted",
                "unlinked",
                "overreach",
            }
            weight = 0.15 / important_count if claim.importance == "important" else 0.0
            item = finding(
                verifier_id=self.verifier_id,
                level="L2",
                domain="generic",
                criterion_id=f"claim:{claim.claim_id}",
                passed=None if uncertain else passed,
                weight=weight,
                hard_gate=hard_gate,
                message=assessment.rationale,
                evidence_refs=tuple(locator_map[link.locator_id].artifact_uri for link in links),
            )
            findings.append(item)
        return tuple(assessments), tuple(findings)

    def _assess(self, text, claim_type, links, locator_map) -> ClaimAssessment:
        if not links:
            return ClaimAssessment(
                claim_id="claim_" + "0" * 64,
                status="unlinked",
                human_review_required=True,
                rationale="No evidence locator is linked to this claim.",
            )
        relations = {link.relation for link in links}
        conflict = "supports" in relations and "refutes" in relations
        locator_ids = tuple(link.locator_id for link in links)
        try:
            excerpts = tuple(read_locator(self.store, locator_map[link.locator_id]) for link in links)
        except Exception as error:
            return ClaimAssessment(
                claim_id="claim_" + "0" * 64,
                status="refuted",
                locator_ids=locator_ids,
                human_review_required=True,
                rationale=f"Evidence locator readback failed: {type(error).__name__}.",
            )
        if conflict:
            return ClaimAssessment(
                claim_id="claim_" + "0" * 64,
                status="uncertain",
                locator_ids=locator_ids,
                evidence_excerpts=excerpts,
                conflict_detected=True,
                human_review_required=True,
                rationale="The claim has both supporting and refuting evidence links.",
            )
        if claim_type == "causal":
            bounded = bool(BOUNDED_PATTERN.search(text))
            return ClaimAssessment(
                claim_id="claim_" + "0" * 64,
                status="limited" if bounded else "overreach",
                locator_ids=locator_ids,
                evidence_excerpts=excerpts,
                causal_overreach=not bounded,
                human_review_required=True,
                rationale=(
                    "The causal statement is explicitly bounded and still requires expert review."
                    if bounded
                    else "Positive causal language is unsupported by a causal-design verifier."
                ),
            )
        if "refutes" in relations:
            return ClaimAssessment(
                claim_id="claim_" + "0" * 64,
                status="refuted",
                locator_ids=locator_ids,
                evidence_excerpts=excerpts,
                human_review_required=True,
                rationale="The only directional evidence link refutes the claim.",
            )
        if claim_type == "statistical":
            claim_numbers = [float(value) for value in NUMBER_PATTERN.findall(text)]
            evidence_numbers = [float(value) for excerpt in excerpts for value in NUMBER_PATTERN.findall(excerpt)]
            supported = bool(claim_numbers) and all(
                any(
                    abs(value - observed) <= 1e-10 * max(1.0, abs(value), abs(observed))
                    for observed in evidence_numbers
                )
                for value in claim_numbers
            )
            return ClaimAssessment(
                claim_id="claim_" + "0" * 64,
                status="supported" if supported else "refuted",
                locator_ids=locator_ids,
                evidence_excerpts=excerpts,
                human_review_required=not supported,
                rationale=(
                    "Every numeric value in the claim reads back from typed evidence."
                    if supported
                    else "At least one claimed numeric value does not read back from typed evidence."
                ),
            )
        if claim_type == "mechanistic":
            bounded = bool(BOUNDED_PATTERN.search(text))
            return ClaimAssessment(
                claim_id="claim_" + "0" * 64,
                status="limited" if bounded else "uncertain",
                locator_ids=locator_ids,
                evidence_excerpts=excerpts,
                human_review_required=True,
                rationale=(
                    "The mechanistic interpretation is explicitly bounded."
                    if bounded
                    else "A locator is present, but no deterministic mechanism-entailment rule applies."
                ),
            )
        normalized_claim = " ".join(text.casefold().split())
        exact_support = any(" ".join(excerpt.casefold().split()) in normalized_claim for excerpt in excerpts)
        return ClaimAssessment(
            claim_id="claim_" + "0" * 64,
            status="supported" if exact_support else "uncertain",
            locator_ids=locator_ids,
            evidence_excerpts=excerpts,
            human_review_required=not exact_support,
            rationale=(
                "The exact source excerpt appears in the factual claim."
                if exact_support
                else "The artifact is reachable, but deterministic factual entailment is unavailable."
            ),
        )
