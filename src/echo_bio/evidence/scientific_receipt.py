"""Build and validate scientific execution receipts from immutable artifacts."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from typing import Any

from echo_bio.artifacts.contracts import ArtifactRef
from echo_bio.artifacts.store import LocalArtifactStore
from echo_bio.evidence.contracts import (
    DiagnosticReceipt,
    RandomSeedRecord,
    ScientificExecutionReceipt,
    ScientificReceiptCandidate,
)
from echo_bio.evidence.diagnostic_receipt import DiagnosticReceiptValidator, ReceiptValidationError
from echo_bio.telemetry.redaction import canonical_json

ARTIFACT_URI_PATTERN = re.compile(r"artifact://[A-Za-z0-9_.-]+/art_[a-f0-9]{64}\?sha256=[a-f0-9]{64}")
SEED_PATTERNS = (
    ("python.random.seed", re.compile(r"\brandom\.seed\(\s*(-?\d+)\s*\)")),
    ("numpy.random.seed", re.compile(r"\b(?:np|numpy)\.random\.seed\(\s*(-?\d+)\s*\)")),
    ("r.set.seed", re.compile(r"\bset\.seed\(\s*(-?\d+)\s*\)")),
    ("cli.--seed", re.compile(r"(?:^|\s)--seed(?:=|\s+)(-?\d+)(?:\s|$)")),
)


def extract_random_seeds(code: str) -> tuple[RandomSeedRecord, ...]:
    records = []
    for source, pattern in SEED_PATTERNS:
        records.extend(RandomSeedRecord(source=source, value=int(match.group(1))) for match in pattern.finditer(code))
    return tuple(records)


def extract_parameters(code: str, language: str) -> dict[str, str | int | float | bool | None]:
    parameters: dict[str, str | int | float | bool | None] = {}
    if language == "python":
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return parameters
        for call_index, node in enumerate(item for item in ast.walk(tree) if isinstance(item, ast.Call)):
            for keyword in node.keywords:
                if keyword.arg is None:
                    continue
                try:
                    value = ast.literal_eval(keyword.value)
                except (ValueError, TypeError):
                    continue
                if value is None or isinstance(value, (str, int, float, bool)):
                    parameters[f"call{call_index}.{keyword.arg}"] = value
    else:
        pattern = re.compile(r"(?:--)?([A-Za-z][A-Za-z0-9_.-]*)\s*(?:=|\s+)\s*(['\"]?[^,\s)]+['\"]?)")
        for index, match in enumerate(pattern.finditer(code)):
            raw = match.group(2).strip("'\"")
            value: str | int | float | bool | None = raw
            if re.fullmatch(r"-?\d+", raw):
                value = int(raw)
            elif re.fullmatch(r"-?\d+(?:\.\d+)", raw):
                value = float(raw)
            elif raw.lower() in {"true", "false"}:
                value = raw.lower() == "true"
            parameters[f"arg{index}.{match.group(1)}"] = value
    return parameters


def _receipt_id(payload: dict[str, Any]) -> str:
    return "sci_" + hashlib.sha256(canonical_json(payload)).hexdigest()


class ScientificReceiptBuilder:
    def __init__(self, store: LocalArtifactStore) -> None:
        self.store = store

    def build(
        self,
        *,
        execution_id: str,
        run_id: str,
        attempt_id: str,
        language: str,
        code_artifact: ArtifactRef,
        output_artifacts: tuple[ArtifactRef, ...],
        diagnostic_receipt_artifact: ArtifactRef,
        environment_digest: str,
        data_manifest_digest: str,
        source_trace_event_id: str,
        candidate: ScientificReceiptCandidate | None = None,
    ) -> ScientificExecutionReceipt:
        code = self.store.get_text(code_artifact)
        extracted_parameters = extract_parameters(code, language)
        if candidate is not None and any(
            key not in extracted_parameters or extracted_parameters[key] != value
            for key, value in candidate.parameters.items()
        ):
            raise ReceiptValidationError("candidate parameters conflict with deterministic code extraction")
        inputs = []
        for uri in dict.fromkeys(ARTIFACT_URI_PATTERN.findall(code)):
            inputs.append(self.store.resolve(uri))
        diagnostic = DiagnosticReceipt.model_validate_json(self.store.get_text(diagnostic_receipt_artifact))
        DiagnosticReceiptValidator(self.store).validate(diagnostic)
        declared = diagnostic.observed_exit_state
        if candidate is not None and candidate.declared_exit_state != "unknown":
            declared = candidate.declared_exit_state
        identity_payload = {
            "execution_id": execution_id,
            "code_sha256": code_artifact.sha256,
            "input_sha256": [ref.sha256 for ref in inputs],
            "output_sha256": [ref.sha256 for ref in output_artifacts],
            "diagnostic_sha256": diagnostic_receipt_artifact.sha256,
            "environment_digest": environment_digest,
            "data_manifest_digest": data_manifest_digest,
        }
        receipt = ScientificExecutionReceipt(
            receipt_id=_receipt_id(identity_payload),
            execution_id=execution_id,
            run_id=run_id,
            attempt_id=attempt_id,
            language=language,
            code_artifact=code_artifact,
            input_artifacts=tuple(inputs),
            output_artifacts=output_artifacts,
            diagnostic_receipt_artifact=diagnostic_receipt_artifact,
            parameters=extracted_parameters,
            random_seeds=extract_random_seeds(code),
            environment_digest=environment_digest,
            data_manifest_digest=data_manifest_digest,
            observed_exit_state=diagnostic.observed_exit_state,
            declared_exit_state=declared,
            source_trace_event_id=source_trace_event_id,
        )
        ScientificReceiptValidator(self.store).validate(receipt)
        return receipt


class ScientificReceiptValidator:
    def __init__(self, store: LocalArtifactStore) -> None:
        self.store = store

    def validate(self, receipt: ScientificExecutionReceipt) -> None:
        code = self.store.get_text(receipt.code_artifact)
        for ref in (*receipt.input_artifacts, *receipt.output_artifacts):
            self.store.verify(ref)
        diagnostic = DiagnosticReceipt.model_validate_json(self.store.get_text(receipt.diagnostic_receipt_artifact))
        DiagnosticReceiptValidator(self.store).validate(diagnostic)
        if diagnostic.execution_id != receipt.execution_id:
            raise ReceiptValidationError("diagnostic and scientific receipts refer to different executions")
        if receipt.observed_exit_state != diagnostic.observed_exit_state:
            raise ReceiptValidationError("scientific receipt observed exit state mismatches source diagnostics")
        if receipt.declared_exit_state != receipt.observed_exit_state:
            raise ReceiptValidationError("declared exit state mismatches observed exit state")
        if receipt.parameters != extract_parameters(code, receipt.language):
            raise ReceiptValidationError("scientific receipt parameters mismatch deterministic code extraction")
        if receipt.random_seeds != extract_random_seeds(code):
            raise ReceiptValidationError("scientific receipt random seeds mismatch deterministic code extraction")
        expected_inputs = tuple(self.store.resolve(uri) for uri in dict.fromkeys(ARTIFACT_URI_PATTERN.findall(code)))
        if receipt.input_artifacts != expected_inputs:
            raise ReceiptValidationError("scientific receipt input artifacts do not match code references")
        identity_payload = {
            "execution_id": receipt.execution_id,
            "code_sha256": receipt.code_artifact.sha256,
            "input_sha256": [ref.sha256 for ref in receipt.input_artifacts],
            "output_sha256": [ref.sha256 for ref in receipt.output_artifacts],
            "diagnostic_sha256": receipt.diagnostic_receipt_artifact.sha256,
            "environment_digest": receipt.environment_digest,
            "data_manifest_digest": receipt.data_manifest_digest,
        }
        if receipt.receipt_id != _receipt_id(identity_payload):
            raise ReceiptValidationError("scientific receipt identity mismatch")


def persist_receipt(store: LocalArtifactStore, run_id: str, receipt: DiagnosticReceipt | ScientificExecutionReceipt):
    return store.put_text(
        run_id=run_id,
        text=json.dumps(receipt.model_dump(mode="json"), sort_keys=True, separators=(",", ":")),
        media_type="application/json",
        data_class=receipt.source_artifact.data_class
        if isinstance(receipt, DiagnosticReceipt)
        else receipt.code_artifact.data_class,
    )
