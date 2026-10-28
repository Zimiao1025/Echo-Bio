"""Frozen capability-floor evaluation rules."""

from __future__ import annotations

import hashlib

from echo_bio.telemetry.redaction import canonical_json
from echo_bio.verifiers.contracts import CapabilityDecision, GateConfig, VerifierReport


class CapabilityGate:
    def __init__(self, config: GateConfig) -> None:
        self.config = config
        self.config_sha256 = hashlib.sha256(canonical_json(config.model_dump(mode="json"))).hexdigest()

    def evaluate(self, report: VerifierReport) -> CapabilityDecision:
        hard_gate_passed = report.hard_gate_passed
        coverage_passed = report.l0_l1_coverage_fraction >= self.config.minimum_l0_l1_coverage
        score_passed = report.weighted_score >= self.config.minimum_weighted_score
        human_required = report.audit_routing.required or any(
            assessment.human_review_required for assessment in report.claim_assessments
        )
        reasons = []
        if not hard_gate_passed:
            reasons.append("hard verifier gate failed")
        if not coverage_passed:
            reasons.append("L0/L1 rubric coverage below minimum")
        if not score_passed:
            reasons.append("weighted verifier score below minimum")
        if human_required:
            reasons.append("mandatory human audit pending")
        if not hard_gate_passed or not coverage_passed or not score_passed:
            decision = "fail"
        elif human_required and self.config.uncertain_policy == "human_review":
            decision = "uncertain"
        else:
            decision = "pass"
        return CapabilityDecision(
            run_id=report.run_id,
            decision=decision,
            hard_gate_passed=hard_gate_passed,
            coverage_passed=coverage_passed,
            score_passed=score_passed,
            human_review_required=human_required,
            reasons=tuple(reasons),
            gate_config_sha256=self.config_sha256,
        )
