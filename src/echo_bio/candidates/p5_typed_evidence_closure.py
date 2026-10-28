"""P5: model-external, typed evidence closure for paired search-dev runs.

P5 is an independent successor to the rejected P4 prompt treatment.  It does
not add instructions to the provider-visible prompt.  Instead it freezes one
validated resource-selection receipt for both arms and reduces public task
contracts plus executed artifacts into a deterministic termination decision.

Only public task fields are compiled.  Reference answers, private evaluator
values and post-hoc rubric outcomes are rejected at the compiler boundary.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from echo_bio.pilot.resource_abi import CanonicalResourceSelection

MECHANISM_ID = "P5"
POLICY_ID = "p5_shared_receipt_typed_evidence_closure_v1"

_FORBIDDEN_TASK_KEYS = {
    "answer_regex",
    "evaluator",
    "evaluators",
    "expected_value",
    "expected_values",
    "private_rubric",
    "reference_answer",
    "reference_result",
    "rubric_result",
}


class P5Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PublicEvidenceBoundaryViolation(ValueError):
    """A task attempted to expose answer-bearing evaluation material."""


class SharedResourceReceiptViolation(ValueError):
    """Paired arms did not receive the same validated resource receipt."""


class SharedResourceReceipt(P5Contract):
    schema_version: Literal["echo.p5-shared-resource-receipt.v1"] = (
        "echo.p5-shared-resource-receipt.v1"
    )
    task_id: str = Field(min_length=1)
    retrieval_prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    selection: dict[str, tuple[int, ...]]
    canonical_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    protocol_valid: Literal[True] = True

    @model_validator(mode="after")
    def validate_canonical_selection(self) -> SharedResourceReceipt:
        expected_keys = {"tools", "data_lake", "libraries", "know_how"}
        if set(self.selection) != expected_keys:
            raise ValueError("resource selection keys are not exact")
        if any(
            value < 0 or isinstance(value, bool)
            for values in self.selection.values()
            for value in values
        ):
            raise ValueError("resource indices must be non-negative integers")
        if any(len(values) != len(set(values)) for values in self.selection.values()):
            raise ValueError("resource indices must be unique within each category")
        document = {key: list(self.selection[key]) for key in sorted(self.selection)}
        canonical = json.dumps(document, separators=(",", ":"), sort_keys=True)
        if _sha(canonical) != self.canonical_sha256:
            raise ValueError("resource receipt canonical digest mismatch")
        return self


class PairedActionContextBinding(P5Contract):
    schema_version: Literal["echo.p5-paired-action-context.v1"] = (
        "echo.p5-paired-action-context.v1"
    )
    task_id: str = Field(min_length=1)
    resource_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    task_prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    base_action_system_prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    pre_treatment_context_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


EvidencePhase = Literal["precondition", "execution", "finalization"]
EvidenceKind = Literal[
    "input_artifact",
    "successful_execution",
    "durable_output",
    "machine_result_schema",
    "declared_limitation",
    "evidence_locator",
]


class EvidenceRequirement(P5Contract):
    requirement_id: str = Field(min_length=1)
    kind: EvidenceKind
    phase: EvidencePhase
    locator: str | None = None
    source_pointer: str = Field(min_length=1)


class PublicEvidenceContract(P5Contract):
    schema_version: Literal["echo.p5-public-evidence-contract.v1"] = (
        "echo.p5-public-evidence-contract.v1"
    )
    task_id: str = Field(min_length=1)
    task_contract_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    public_only: Literal[True] = True
    requirements: tuple[EvidenceRequirement, ...]


RequirementStatusValue = Literal["satisfied", "missing", "invalid", "deferred"]


class EvidenceRequirementStatus(P5Contract):
    requirement_id: str
    status: RequirementStatusValue
    detail_code: str
    observed_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class ExecutionObservation(P5Contract):
    execution_key: str = Field(min_length=1)
    successful: bool
    language: str = Field(min_length=1)
    code_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class EvidenceCompletionState(P5Contract):
    schema_version: Literal["echo.p5-evidence-completion-state.v1"] = (
        "echo.p5-evidence-completion-state.v1"
    )
    task_id: str
    contract_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    statuses: tuple[EvidenceRequirementStatus, ...]
    execution_count: int = Field(ge=0)
    successful_execution_count: int = Field(ge=0)
    duplicate_execution_count: int = Field(ge=0)
    remaining_execution_actions: int = Field(ge=0)
    state_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class TerminationDecision(P5Contract):
    schema_version: Literal["echo.p5-termination-decision.v1"] = (
        "echo.p5-termination-decision.v1"
    )
    action: Literal["continue_batch", "finalize_now", "fail_closed"]
    reason_code: Literal[
        "execution_evidence_complete",
        "execution_evidence_missing",
        "execution_budget_exhausted",
    ]
    unresolved_requirement_ids: tuple[str, ...] = ()
    state_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def freeze_shared_resource_receipt(
    *,
    task_id: str,
    retrieval_prompt: str,
    selection: CanonicalResourceSelection,
) -> SharedResourceReceipt:
    """Freeze one qualified selection to be replayed into both paired arms."""

    document = selection.as_document()
    return SharedResourceReceipt(
        task_id=task_id,
        retrieval_prompt_sha256=_sha(retrieval_prompt),
        selection={key: tuple(value) for key, value in document.items()},
        canonical_sha256=selection.sha256,
    )


def bind_pre_treatment_action_context(
    *,
    task_id: str,
    task_prompt: str,
    base_action_system_prompt: str,
    receipt: SharedResourceReceipt,
) -> PairedActionContextBinding:
    """Hash the exact context that must be identical before arm treatment."""

    if receipt.task_id != task_id:
        raise SharedResourceReceiptViolation("resource receipt task identity mismatch")
    task_prompt_sha256 = _sha(task_prompt)
    if task_prompt_sha256 != receipt.retrieval_prompt_sha256:
        raise SharedResourceReceiptViolation("resource receipt retrieval prompt mismatch")
    base_sha = _sha(base_action_system_prompt)
    context = {
        "base_action_system_prompt_sha256": base_sha,
        "resource_receipt_sha256": receipt.canonical_sha256,
        "task_prompt_sha256": task_prompt_sha256,
    }
    return PairedActionContextBinding(
        task_id=task_id,
        resource_receipt_sha256=receipt.canonical_sha256,
        task_prompt_sha256=task_prompt_sha256,
        base_action_system_prompt_sha256=base_sha,
        pre_treatment_context_sha256=_canonical_sha(context),
    )


def assert_paired_context_identical(
    control: PairedActionContextBinding,
    candidate: PairedActionContextBinding,
) -> None:
    if control != candidate:
        raise SharedResourceReceiptViolation(
            "paired arms do not share an exact pre-treatment action context"
        )


def compile_public_evidence_contract(task: dict[str, Any]) -> PublicEvidenceContract:
    """Compile content-free evidence requirements from an allowlisted task view."""

    leaked = sorted(_find_forbidden_keys(task))
    if leaked:
        raise PublicEvidenceBoundaryViolation(
            f"answer-bearing task keys are forbidden: {', '.join(leaked)}"
        )
    task_id = str(task.get("task_id") or "")
    if not task_id:
        raise ValueError("public task requires task_id")
    data = _require_mapping(task, "data")
    execution = _require_mapping(task, "execution_contract")
    schema = _require_mapping(task, "machine_readable_result_schema")
    data_path = str(data.get("path") or "")
    if not data_path:
        raise ValueError("public task requires data.path")
    machine_path = str(execution.get("machine_readable_result_path") or "")
    if not machine_path:
        raise ValueError("execution contract requires machine_readable_result_path")
    required_outputs = execution.get("required_output_paths")
    if not isinstance(required_outputs, list) or not all(
        isinstance(value, str) and value for value in required_outputs
    ):
        raise ValueError("execution contract requires string required_output_paths")

    requirements: list[EvidenceRequirement] = [
        EvidenceRequirement(
            requirement_id="input:primary",
            kind="input_artifact",
            phase="precondition",
            locator=f"/app/data/{Path(data_path).name}",
            source_pointer="/data/path",
        ),
        EvidenceRequirement(
            requirement_id="execution:successful",
            kind="successful_execution",
            phase="execution",
            source_pointer="/execution_contract",
        ),
    ]
    for index, locator in enumerate(required_outputs):
        requirements.append(
            EvidenceRequirement(
                requirement_id=f"output:{index}",
                kind="durable_output",
                phase="execution",
                locator=locator,
                source_pointer=f"/execution_contract/required_output_paths/{index}",
            )
        )
    requirements.append(
        EvidenceRequirement(
            requirement_id="result:schema",
            kind="machine_result_schema",
            phase="execution",
            locator=machine_path,
            source_pointer="/machine_readable_result_schema",
        )
    )
    required_evidence = task.get("required_evidence", [])
    if not isinstance(required_evidence, list):
        raise ValueError("required_evidence must be a list")
    if "limitations" in required_evidence:
        requirements.append(
            EvidenceRequirement(
                requirement_id="finalization:limitations",
                kind="declared_limitation",
                phase="finalization",
                source_pointer="/required_evidence",
            )
        )
    for label in required_evidence:
        if isinstance(label, str) and "locator" in label:
            requirements.append(
                EvidenceRequirement(
                    requirement_id=f"finalization:locator:{len(requirements)}",
                    kind="evidence_locator",
                    phase="finalization",
                    source_pointer="/required_evidence",
                )
            )

    public_projection = {
        "task_id": task_id,
        "data": data,
        "execution_contract": execution,
        "machine_readable_result_schema": schema,
        "required_evidence": required_evidence,
    }
    return PublicEvidenceContract(
        task_id=task_id,
        task_contract_sha256=_canonical_sha(public_projection),
        requirements=tuple(requirements),
    )


def reduce_evidence_state(
    *,
    contract: PublicEvidenceContract,
    task: dict[str, Any],
    data_root: Path,
    workspace: Path,
    executions: tuple[ExecutionObservation, ...],
    remaining_execution_actions: int,
) -> EvidenceCompletionState:
    """Project public files and execution receipts into one immutable state."""

    if remaining_execution_actions < 0:
        raise ValueError("remaining execution actions cannot be negative")
    if contract.task_id != task.get("task_id"):
        raise ValueError("evidence contract task identity mismatch")
    schema = _require_mapping(task, "machine_readable_result_schema")
    statuses: list[EvidenceRequirementStatus] = []
    successful = sum(observation.successful for observation in executions)
    for requirement in contract.requirements:
        if requirement.phase == "finalization":
            statuses.append(
                EvidenceRequirementStatus(
                    requirement_id=requirement.requirement_id,
                    status="deferred",
                    detail_code="checked_by_typed_finalizer",
                )
            )
            continue
        if requirement.kind == "successful_execution":
            statuses.append(
                EvidenceRequirementStatus(
                    requirement_id=requirement.requirement_id,
                    status="satisfied" if successful else "missing",
                    detail_code=(
                        "successful_execution_observed"
                        if successful
                        else "no_successful_execution"
                    ),
                )
            )
            continue
        assert requirement.locator is not None
        resolved = _resolve_container_locator(
            requirement.locator, data_root=data_root, workspace=workspace
        )
        if not resolved.is_file():
            statuses.append(
                EvidenceRequirementStatus(
                    requirement_id=requirement.requirement_id,
                    status="missing",
                    detail_code="artifact_not_found",
                )
            )
            continue
        digest = _file_sha(resolved)
        if requirement.kind != "machine_result_schema":
            statuses.append(
                EvidenceRequirementStatus(
                    requirement_id=requirement.requirement_id,
                    status="satisfied",
                    detail_code="artifact_resolved",
                    observed_sha256=digest,
                )
            )
            continue
        try:
            result = json.loads(resolved.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            valid, detail = False, "machine_result_invalid_json"
        else:
            valid, detail = _matches_schema(result, schema)
        statuses.append(
            EvidenceRequirementStatus(
                requirement_id=requirement.requirement_id,
                status="satisfied" if valid else "invalid",
                detail_code="machine_result_schema_valid" if valid else detail,
                observed_sha256=digest,
            )
        )

    duplicate_count = sum(count - 1 for count in Counter(
        observation.execution_key for observation in executions
    ).values() if count > 1)
    core = {
        "contract_sha256": contract.task_contract_sha256,
        "duplicate_execution_count": duplicate_count,
        "execution_count": len(executions),
        "remaining_execution_actions": remaining_execution_actions,
        "statuses": [status.model_dump(mode="json") for status in statuses],
        "successful_execution_count": successful,
        "task_id": contract.task_id,
    }
    return EvidenceCompletionState(
        task_id=contract.task_id,
        contract_sha256=contract.task_contract_sha256,
        statuses=tuple(statuses),
        execution_count=len(executions),
        successful_execution_count=successful,
        duplicate_execution_count=duplicate_count,
        remaining_execution_actions=remaining_execution_actions,
        state_sha256=_canonical_sha(core),
    )


def decide_termination(state: EvidenceCompletionState) -> TerminationDecision:
    """Make one deterministic, model-free decision for all missing evidence."""

    unresolved = tuple(
        status.requirement_id
        for status in state.statuses
        if status.status in {"missing", "invalid"}
    )
    if not unresolved:
        return TerminationDecision(
            action="finalize_now",
            reason_code="execution_evidence_complete",
            state_sha256=state.state_sha256,
        )
    if state.remaining_execution_actions == 0:
        return TerminationDecision(
            action="fail_closed",
            reason_code="execution_budget_exhausted",
            unresolved_requirement_ids=unresolved,
            state_sha256=state.state_sha256,
        )
    return TerminationDecision(
        action="continue_batch",
        reason_code="execution_evidence_missing",
        unresolved_requirement_ids=unresolved,
        state_sha256=state.state_sha256,
    )


def _find_forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_TASK_KEYS:
                found.add(str(key))
            found.update(_find_forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_find_forbidden_keys(child))
    return found


def _require_mapping(document: dict[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"public task requires object field: {key}")
    return value


def _resolve_container_locator(locator: str, *, data_root: Path, workspace: Path) -> Path:
    for prefix, root in (("/app/data/", data_root), ("/app/work/", workspace)):
        if locator.startswith(prefix):
            relative = Path(locator.removeprefix(prefix))
            if relative.is_absolute() or ".." in relative.parts:
                break
            resolved_root = root.resolve()
            candidate = (resolved_root / relative).resolve()
            if candidate == resolved_root or resolved_root in candidate.parents:
                return candidate
    raise PublicEvidenceBoundaryViolation(f"locator crosses public runtime roots: {locator}")


def _matches_schema(value: Any, schema: dict[str, Any], pointer: str = "") -> tuple[bool, str]:
    if not isinstance(value, dict):
        return False, f"schema_type_mismatch:{pointer or '/'}"
    if set(value) != set(schema):
        return False, f"schema_keys_mismatch:{pointer or '/'}"
    for key, expected in schema.items():
        child = value[key]
        child_pointer = f"{pointer}/{key}"
        if isinstance(expected, dict):
            valid, detail = _matches_schema(child, expected, child_pointer)
        else:
            valid = _matches_type(child, str(expected))
            detail = f"schema_type_mismatch:{child_pointer}"
        if not valid:
            return False, detail
    return True, "machine_result_schema_valid"


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected.startswith("array[") and expected.endswith("]"):
        element_type = expected[6:-1]
        return isinstance(value, list) and all(
            _matches_type(element, element_type) for element in value
        )
    return False


def _canonical_sha(value: Any) -> str:
    return _sha(json.dumps(value, separators=(",", ":"), sort_keys=True))


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
