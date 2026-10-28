"""Generic L0 provenance, integrity, policy, completeness, and replay checks."""

from __future__ import annotations

from echo_bio.artifacts.store import LocalArtifactStore
from echo_bio.evidence.scientific_receipt import ScientificReceiptValidator
from echo_bio.verifiers.base import finding
from echo_bio.verifiers.contracts import VerificationBundle, VerificationFinding
from echo_bio.verifiers.locators import read_locator


class GenericProvenanceVerifier:
    verifier_id = "generic-provenance-v1"

    def __init__(self, store: LocalArtifactStore) -> None:
        self.store = store

    def verify(self, bundle: VerificationBundle) -> tuple[VerificationFinding, ...]:
        refs = [bundle.input_artifact, bundle.analysis_artifact]
        for receipt in bundle.scientific_receipts:
            refs.extend(
                (
                    receipt.code_artifact,
                    *receipt.input_artifacts,
                    *receipt.output_artifacts,
                    receipt.diagnostic_receipt_artifact,
                )
            )
        locator_refs = []
        schema_ok = True
        try:
            for locator in bundle.claim_graph.locators:
                locator_refs.append(self.store.resolve(locator.artifact_uri))
                read_locator(self.store, locator)
            for uri in bundle.claim_graph.scientific_receipt_uris:
                locator_refs.append(self.store.resolve(uri))
        except Exception:
            schema_ok = False
        findings = [
            finding(
                verifier_id=self.verifier_id,
                level="L0",
                domain="generic",
                criterion_id="schema_and_reference_authenticity",
                passed=schema_ok,
                weight=0.03,
                hard_gate=True,
                message=(
                    "All typed contracts and referenced artifact URIs resolve."
                    if schema_ok
                    else "A claim or receipt reference does not resolve to durable metadata."
                ),
                evidence_refs=tuple(locator.artifact_uri for locator in bundle.claim_graph.locators),
            )
        ]
        integrity_errors = []
        for ref in (*refs, *locator_refs):
            try:
                self.store.verify(ref)
            except Exception as error:
                integrity_errors.append(f"{ref.uri}:{type(error).__name__}")
        for receipt in bundle.scientific_receipts:
            try:
                ScientificReceiptValidator(self.store).validate(receipt)
            except Exception as error:
                integrity_errors.append(f"{receipt.receipt_id}:{type(error).__name__}")
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L0",
                domain="generic",
                criterion_id="artifact_hash_integrity",
                passed=not integrity_errors,
                weight=0.04,
                hard_gate=True,
                message=(
                    f"Verified {len(refs) + len(locator_refs)} referenced artifacts and all execution receipts."
                    if not integrity_errors
                    else "Integrity failures: " + "; ".join(integrity_errors)
                ),
                evidence_refs=tuple(dict.fromkeys(ref.uri for ref in (*refs, *locator_refs))),
            )
        )
        lineage_ok = all(
            receipt.run_id == bundle.run_id
            and receipt.environment_digest == bundle.environment_digest
            and receipt.data_manifest_digest == bundle.data_manifest_digest
            for receipt in bundle.scientific_receipts
        ) and bundle.claim_graph.run_id == bundle.run_id
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L0",
                domain="generic",
                criterion_id="run_data_environment_lineage",
                passed=lineage_ok,
                weight=0.04,
                hard_gate=True,
                message=(
                    "Run, data-manifest, and environment identities are consistent."
                    if lineage_ok
                    else "Run, data-manifest, or environment identity mismatch."
                ),
            )
        )
        classes = {ref.data_class for ref in (*refs, *locator_refs)}
        policy_ok = bundle.split != "external" or not classes.intersection({"controlled", "secret"})
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L0",
                domain="generic",
                criterion_id="data_class_policy",
                passed=policy_ok,
                weight=0.02,
                hard_gate=True,
                message=(
                    f"Data classes {sorted(classes)} comply with split={bundle.split}."
                    if policy_ok
                    else "Controlled or secret artifacts cannot enter the external evaluation split."
                ),
            )
        )
        steps_complete = all(
            step.status == "completed" and step.completion_evidence for step in bundle.research_plan.steps
        )
        required_artifacts = all(
            receipt.code_artifact and receipt.output_artifacts and receipt.diagnostic_receipt_artifact
            for receipt in bundle.scientific_receipts
        )
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L0",
                domain="generic",
                criterion_id="required_steps_and_artifacts",
                passed=steps_complete and required_artifacts,
                weight=0.04,
                hard_gate=True,
                message=(
                    "All planned steps are evidence-complete and required execution artifacts exist."
                    if steps_complete and required_artifacts
                    else "A planned step or required execution artifact is incomplete."
                ),
            )
        )
        replay = bundle.replay_manifest
        receipt_code = {receipt.code_artifact.uri for receipt in bundle.scientific_receipts}
        receipt_inputs = {ref.uri for receipt in bundle.scientific_receipts for ref in receipt.input_artifacts}
        receipt_outputs = {ref.uri for receipt in bundle.scientific_receipts for ref in receipt.output_artifacts}
        replay_ok = (
            replay.run_id == bundle.run_id
            and replay.environment_digest == bundle.environment_digest
            and replay.data_manifest_digest == bundle.data_manifest_digest
            and receipt_code.issubset({ref.uri for ref in replay.code_artifacts})
            and receipt_inputs.issubset({ref.uri for ref in replay.input_artifacts})
            and receipt_outputs.issubset({ref.uri for ref in replay.output_artifacts})
            and not replay.network_required
        )
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L0",
                domain="generic",
                criterion_id="clean_replay_manifest",
                passed=replay_ok,
                weight=0.03,
                hard_gate=True,
                message=(
                    "Replay manifest closes code/input/output/environment lineage without network access."
                    if replay_ok
                    else "Replay manifest is incomplete, inconsistent, or requires network access."
                ),
                evidence_refs=tuple(
                    ref.uri for ref in (*replay.code_artifacts, *replay.input_artifacts, *replay.output_artifacts)
                ),
            )
        )
        return tuple(findings)
