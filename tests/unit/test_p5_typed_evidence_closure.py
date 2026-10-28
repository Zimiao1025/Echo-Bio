from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from echo_bio.candidates.p5_typed_evidence_closure import (
    ExecutionObservation,
    PublicEvidenceBoundaryViolation,
    SharedResourceReceipt,
    SharedResourceReceiptViolation,
    assert_paired_context_identical,
    bind_pre_treatment_action_context,
    compile_public_evidence_contract,
    decide_termination,
    freeze_shared_resource_receipt,
    reduce_evidence_state,
)
from echo_bio.contracts.trace import RedactionMetadata, TraceEvent
from echo_bio.pilot.resource_abi import CanonicalResourceSelection


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _task() -> dict[str, object]:
    return {
        "task_id": "public-task-1",
        "data": {"path": "fixtures/data.csv", "sha256": "0" * 64},
        "execution_contract": {
            "machine_readable_result_path": "/app/work/result.json",
            "required_output_paths": [
                "/app/work/analysis.py",
                "/app/work/result.json",
            ],
        },
        "machine_readable_result_schema": {
            "count": "integer",
            "labels": "array[string]",
            "nested": {"score": "number"},
        },
        "required_evidence": [
            "input-data locator",
            "executed analysis-code locator",
            "machine-readable result locator",
            "limitations",
        ],
        "scientific_rubric": ["public invariant"],
    }


def _observation(*, key: str = "execution-1", successful: bool = True) -> ExecutionObservation:
    return ExecutionObservation(
        execution_key=key,
        successful=successful,
        language="python",
        code_sha256=_sha(key),
    )


def test_shared_receipt_is_content_addressed_and_context_is_exactly_paired() -> None:
    selection = CanonicalResourceSelection(
        tools=(1,),
        data_lake=(),
        libraries=(2, 3),
        know_how=(),
        transport="json_schema_v1",
        normalization="exact_json_object",
    )
    receipt = freeze_shared_resource_receipt(
        task_id="public-task-1",
        retrieval_prompt="same task prompt",
        selection=selection,
    )
    control = bind_pre_treatment_action_context(
        task_id="public-task-1",
        task_prompt="same task prompt",
        base_action_system_prompt="same system and resource projection",
        receipt=receipt,
    )
    candidate = bind_pre_treatment_action_context(
        task_id="public-task-1",
        task_prompt="same task prompt",
        base_action_system_prompt="same system and resource projection",
        receipt=receipt,
    )

    assert receipt.canonical_sha256 == selection.sha256
    assert_paired_context_identical(control, candidate)

    changed = candidate.model_copy(
        update={"base_action_system_prompt_sha256": "f" * 64}
    )
    with pytest.raises(SharedResourceReceiptViolation):
        assert_paired_context_identical(control, changed)


def test_shared_receipt_rejects_digest_or_prompt_mismatch() -> None:
    with pytest.raises(ValidationError):
        SharedResourceReceipt(
            task_id="public-task-1",
            retrieval_prompt_sha256="0" * 64,
            selection={"tools": (1,), "data_lake": (), "libraries": (), "know_how": ()},
            canonical_sha256="0" * 64,
        )


@pytest.mark.parametrize("forbidden", ["reference_result", "answer_regex", "private_rubric"])
def test_public_compiler_rejects_answer_bearing_material(forbidden: str) -> None:
    task = _task()
    task[forbidden] = {"count": 7}
    with pytest.raises(PublicEvidenceBoundaryViolation):
        compile_public_evidence_contract(task)


def test_compiler_creates_execution_and_deferred_finalization_requirements() -> None:
    contract = compile_public_evidence_contract(_task())

    assert contract.public_only is True
    assert {requirement.kind for requirement in contract.requirements} >= {
        "input_artifact",
        "successful_execution",
        "durable_output",
        "machine_result_schema",
        "declared_limitation",
        "evidence_locator",
    }
    assert all(
        requirement.phase == "finalization"
        for requirement in contract.requirements
        if requirement.kind in {"declared_limitation", "evidence_locator"}
    )


def test_reducer_finalizes_from_real_artifacts_without_provider_text(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    workspace = tmp_path / "work"
    data_root.mkdir()
    workspace.mkdir()
    (data_root / "data.csv").write_text("sample,value\na,1\n")
    (workspace / "analysis.py").write_text("print('executed')\n")
    (workspace / "result.json").write_text(
        json.dumps({"count": 1, "labels": ["a"], "nested": {"score": 1.0}})
    )
    task = _task()
    contract = compile_public_evidence_contract(task)
    state = reduce_evidence_state(
        contract=contract,
        task=task,
        data_root=data_root,
        workspace=workspace,
        executions=(_observation(),),
        remaining_execution_actions=2,
    )
    decision = decide_termination(state)

    assert decision.action == "finalize_now"
    assert decision.reason_code == "execution_evidence_complete"
    assert all(status.status != "missing" for status in state.statuses)
    assert any(status.status == "deferred" for status in state.statuses)


def test_reducer_batches_missing_requirements_and_fails_closed_at_budget(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    workspace = tmp_path / "work"
    data_root.mkdir()
    workspace.mkdir()
    (data_root / "data.csv").write_text("x\n")
    task = _task()
    contract = compile_public_evidence_contract(task)

    continuing = reduce_evidence_state(
        contract=contract,
        task=task,
        data_root=data_root,
        workspace=workspace,
        executions=(),
        remaining_execution_actions=1,
    )
    exhausted = continuing.model_copy(update={"remaining_execution_actions": 0})

    continue_decision = decide_termination(continuing)
    fail_decision = decide_termination(exhausted)
    assert continue_decision.action == "continue_batch"
    assert len(continue_decision.unresolved_requirement_ids) == 4
    assert fail_decision.action == "fail_closed"
    assert fail_decision.unresolved_requirement_ids == continue_decision.unresolved_requirement_ids


def test_reducer_marks_wrong_machine_schema_invalid_and_counts_duplicates(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    workspace = tmp_path / "work"
    data_root.mkdir()
    workspace.mkdir()
    (data_root / "data.csv").write_text("x\n")
    (workspace / "analysis.py").write_text("pass\n")
    (workspace / "result.json").write_text(
        json.dumps({"count": "one", "labels": ["a"], "nested": {"score": 1.0}})
    )
    task = _task()
    state = reduce_evidence_state(
        contract=compile_public_evidence_contract(task),
        task=task,
        data_root=data_root,
        workspace=workspace,
        executions=(_observation(), _observation()),
        remaining_execution_actions=1,
    )

    assert state.duplicate_execution_count == 1
    invalid = next(status for status in state.statuses if status.requirement_id == "result:schema")
    assert invalid.status == "invalid"
    assert invalid.detail_code == "schema_type_mismatch:/count"


@pytest.mark.parametrize(
    "event_type",
    [
        "shared_resource_receipt_bound",
        "evidence_state_updated",
        "termination_decided",
    ],
)
def test_trace_contract_accepts_p5_control_events(event_type: str) -> None:
    TraceEvent(
        event_id="evt-1",
        run_id="run-1",
        attempt_id="attempt-1",
        sequence_no=0,
        event_type=event_type,
        redaction=RedactionMetadata(data_class="public"),
        payload_sha256="0" * 64,
        event_sha256="1" * 64,
    )
