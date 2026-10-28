"""Compile a bounded provider view without changing the immutable source history."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from echo_bio.context.contracts import (
    CompactionDecision,
    ContextCompilerConfig,
    ContextProjectionRecord,
)
from echo_bio.context.economics import evaluate_compaction
from echo_bio.context.evidence_pins import (
    build_context_snapshot,
    canonical_snapshot_bytes,
    snapshot_sha256,
    validate_snapshot,
)
from echo_bio.contracts.plans import ResearchPlan
from echo_bio.telemetry.token_attribution import TokenEstimator

SNAPSHOT_OPEN = "<echo_context_snapshot>"
SNAPSHOT_CLOSE = "</echo_context_snapshot>"


@dataclass(frozen=True)
class _FallbackSystemMessage:
    content: str
    type: str = "system"


@dataclass(frozen=True)
class CompiledContext:
    messages: tuple[Any, ...]
    candidate_messages: tuple[Any, ...]
    record: ContextProjectionRecord


def _text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text") or block.get("content") or "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def _role(message: Any) -> str:
    return str(getattr(message, "type", getattr(message, "role", type(message).__name__))).lower()


def _digest_messages(messages: tuple[Any, ...]) -> str:
    payload = [{"role": _role(message), "content": _text(message)} for message in messages]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _default_message_factory(content: str) -> Any:
    try:
        from langchain_core.messages import SystemMessage

        return SystemMessage(content=content)
    except ImportError:
        return _FallbackSystemMessage(content=content)


class ContextCompiler:
    """Evidence-gated compiler with shadow-first, fail-open delivery semantics."""

    def __init__(
        self,
        config: ContextCompilerConfig | None = None,
        *,
        estimator: TokenEstimator | None = None,
        message_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config or ContextCompilerConfig()
        self.estimator = estimator or TokenEstimator()
        self.message_factory = message_factory or _default_message_factory

    def compile(
        self,
        messages: list[Any] | tuple[Any, ...],
        *,
        task: str,
        plan: ResearchPlan,
        resource_slice: tuple[str, ...] = (),
    ) -> CompiledContext:
        source = tuple(messages)
        try:
            return self._compile(source, task=task, plan=plan, resource_slice=resource_slice)
        except Exception as error:
            return self._compiler_failure(source, plan, error)

    def _compile(
        self,
        source: tuple[Any, ...],
        *,
        task: str,
        plan: ResearchPlan,
        resource_slice: tuple[str, ...],
    ) -> CompiledContext:
        snapshot = build_context_snapshot(task=task, plan=plan, resource_slice=resource_slice)
        snapshot_json = canonical_snapshot_bytes(snapshot).decode()
        snapshot_message = self.message_factory(f"{SNAPSHOT_OPEN}\n{snapshot_json}\n{SNAPSHOT_CLOSE}")

        included = self._included_indices(source, plan)
        system_indices = [index for index in included if _role(source[index]) == "system"]
        other_indices = [index for index in included if index not in system_indices]
        candidate = tuple(source[index] for index in system_indices) + (snapshot_message,) + tuple(
            source[index] for index in other_indices
        )
        dropped = tuple(index for index in range(len(source)) if index not in included)

        parsed_snapshot = self._parse_snapshot(snapshot_message)
        expected_digest = snapshot_sha256(snapshot)
        observed_digest = snapshot_sha256(parsed_snapshot)
        state_equivalent = expected_digest == observed_digest
        completeness = validate_snapshot(parsed_snapshot, plan, source_message_count=len(source))
        source_tokens = self._count_messages(source)
        candidate_tokens = self._count_messages(candidate)
        economics = evaluate_compaction(source_tokens, candidate_tokens, self.config)

        if not completeness.complete:
            decision = CompactionDecision(
                applied=False,
                candidate_eligible=False,
                reason="evidence_incomplete",
                detail="; ".join(completeness.errors),
            )
        elif not state_equivalent:
            decision = CompactionDecision(
                applied=False,
                candidate_eligible=False,
                reason="state_mismatch",
                detail="serialized snapshot did not round-trip to the source plan state",
            )
        elif economics.per_request_savings_tokens <= 0:
            decision = CompactionDecision(
                applied=False,
                candidate_eligible=False,
                reason="no_reduction",
                detail="candidate context is not smaller than the source context",
            )
        elif not economics.economically_eligible:
            decision = CompactionDecision(
                applied=False,
                candidate_eligible=False,
                reason="negative_economics",
                detail="projected future savings do not cover compaction and cache costs",
            )
        elif self.config.mode == "shadow":
            decision = CompactionDecision(
                applied=False,
                candidate_eligible=True,
                reason="shadow_only",
                detail="eligible candidate measured but source context delivered",
            )
        else:
            decision = CompactionDecision(
                applied=True,
                candidate_eligible=True,
                reason="applied",
                detail="evidence, state-equivalence, and economics gates passed",
            )

        delivered = candidate if decision.applied else source
        record = ContextProjectionRecord(
            plan_id=plan.plan_id,
            plan_revision=plan.revision,
            mode=self.config.mode,
            source_message_count=len(source),
            candidate_message_count=len(candidate),
            delivered_message_count=len(delivered),
            source_tokens=source_tokens,
            candidate_tokens=candidate_tokens,
            delivered_tokens=self._count_messages(delivered),
            source_sha256=_digest_messages(source),
            candidate_sha256=_digest_messages(candidate),
            delivered_sha256=_digest_messages(delivered),
            snapshot_sha256=expected_digest,
            included_source_indices=tuple(sorted(included)),
            dropped_source_indices=dropped,
            preserved_active_step=completeness.preserved_active_step,
            preserved_open_issues=completeness.preserved_open_issues,
            preserved_evidence_pins=completeness.preserved_evidence_pins,
            evidence_complete=completeness.complete,
            state_equivalent=state_equivalent,
            economics=economics,
            decision=decision,
        )
        return CompiledContext(messages=delivered, candidate_messages=candidate, record=record)

    def _compiler_failure(
        self,
        source: tuple[Any, ...],
        plan: ResearchPlan,
        error: Exception,
    ) -> CompiledContext:
        source_tokens = self._count_messages(source)
        economics = evaluate_compaction(source_tokens, source_tokens, self.config)
        digest = _digest_messages(source)
        record = ContextProjectionRecord(
            plan_id=plan.plan_id,
            plan_revision=plan.revision,
            mode=self.config.mode,
            source_message_count=len(source),
            candidate_message_count=len(source),
            delivered_message_count=len(source),
            source_tokens=source_tokens,
            candidate_tokens=source_tokens,
            delivered_tokens=source_tokens,
            source_sha256=digest,
            candidate_sha256=digest,
            delivered_sha256=digest,
            snapshot_sha256=snapshot_sha256({}),
            included_source_indices=tuple(range(len(source))),
            dropped_source_indices=(),
            preserved_active_step=False,
            preserved_open_issues=False,
            preserved_evidence_pins=False,
            evidence_complete=False,
            state_equivalent=False,
            economics=economics,
            decision=CompactionDecision(
                applied=False,
                candidate_eligible=False,
                reason="compiler_error",
                detail=f"{type(error).__name__}: {error}",
            ),
        )
        return CompiledContext(messages=source, candidate_messages=source, record=record)

    def _included_indices(self, source: tuple[Any, ...], plan: ResearchPlan) -> set[int]:
        included = {index for index, message in enumerate(source) if _role(message) == "system"}
        first_human = next(
            (index for index, message in enumerate(source) if _role(message) in {"human", "user"}),
            None,
        )
        if first_human is not None:
            included.add(first_human)
        start = max(0, len(source) - self.config.keep_recent_messages)
        included.update(range(start, len(source)))
        included.update(
            pin.source_message_index
            for pin in plan.evidence_pins
            if pin.source_message_index is not None and pin.source_message_index < len(source)
        )
        return included

    def _count_messages(self, messages: tuple[Any, ...]) -> int:
        return sum(self.estimator.count(_text(message)) for message in messages)

    @staticmethod
    def _parse_snapshot(message: Any) -> dict[str, Any]:
        content = _text(message)
        start = content.index(SNAPSHOT_OPEN) + len(SNAPSHOT_OPEN)
        end = content.index(SNAPSHOT_CLOSE, start)
        return json.loads(content[start:end].strip())
