"""Identity hooks that append sanitized research events to the Phase 1 ledger."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from echo_bio.adapters.biomni.hooks import NoOpHarnessHooks
from echo_bio.contracts import (
    ContextProjection,
    ExecutionDecision,
    ExecutionDraft,
    FinalizationDecision,
    ModelRequestDraft,
    ModelResponseObserved,
    ObservationProjection,
    OutcomeDraft,
    PlanUpdated,
    RawExecutionObserved,
    RunEnded,
    RunStart,
)
from echo_bio.contracts.trace import DataClass, ProviderUsage, RunEnvelope, Split, TraceError
from echo_bio.telemetry.execution_audit import execution_error_from_result
from echo_bio.telemetry.ledger import RunTraceWriter, TraceLedger
from echo_bio.telemetry.prompt_attribution_v2 import PromptContextAttributorV2
from echo_bio.telemetry.provider_traffic import (
    CacheWriteAccounting,
    InputTokenAccounting,
    normalize_provider_traffic,
)
from echo_bio.telemetry.redaction import canonical_json
from echo_bio.telemetry.token_attribution import ContextAttributor, TokenEstimator, extract_provider_usage

TAIL_SIGNAL_PATTERN = re.compile(
    r"(?i)(error|exception|warning|failed|quality control|\bqc\b|adjusted p|false discovery|confidence interval)"
)
GROUNDING_PATTERN = re.compile(r"(?i)(artifact[_:-]|evidence[_:-]|PMID\s*:?\s*\d+|doi:\s*10\.|https?://|<citation>)")


def text_content(value: Any) -> str:
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                part = block.get("text") or block.get("content")
                if isinstance(part, str):
                    parts.append(part)
        return "".join(parts)
    return str(content)


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class TelemetryConfig:
    task_id: str
    split: Split
    harness_candidate_id: str
    model_snapshot: str
    provider: str
    environment_digest: str
    data_manifest_digest: str
    data_class: DataClass = "internal"
    capture_content: bool = False
    input_token_accounting: InputTokenAccounting = "unknown"
    cache_write_accounting: CacheWriteAccounting = "unknown"


class TelemetryHarnessHooks(NoOpHarnessHooks):
    """Append-only observer; every return remains identical to its input."""

    def __init__(
        self,
        ledger: TraceLedger,
        config: TelemetryConfig,
        *,
        attributor: ContextAttributor | None = None,
    ) -> None:
        self.ledger = ledger
        self.config = config
        self.attributor = attributor or ContextAttributor()
        self.prompt_attributor_v2 = PromptContextAttributorV2(self.attributor)
        self.output_estimator = TokenEstimator(config.model_snapshot)
        self.writer: RunTraceWriter | None = None

    def _active_writer(self) -> RunTraceWriter:
        if self.writer is None:
            raise RuntimeError("telemetry event emitted outside an active run")
        return self.writer

    def on_run_start(self, event: RunStart) -> None:
        envelope = RunEnvelope(
            run_id=event.identity.run_id,
            attempt_id=event.identity.attempt_id,
            session_id=event.identity.session_id,
            task_id=self.config.task_id,
            split=self.config.split,
            harness_candidate_id=self.config.harness_candidate_id,
            model_snapshot=self.config.model_snapshot,
            provider=self.config.provider,
            environment_digest=self.config.environment_digest,
            data_manifest_digest=self.config.data_manifest_digest,
            data_class=self.config.data_class,
            started_at=event.occurred_at,
        )
        self.writer = self.ledger.create_run(envelope)
        payload = self._text_summary("task", event.task)
        self.writer.append("run_started", payload)

    def before_model_request(self, event: ModelRequestDraft) -> ContextProjection:
        components, totals = self.attributor.attribute(event.messages)
        components_v2, totals_v2 = self.prompt_attributor_v2.upgrade(components)
        messages = []
        for index, message in enumerate(event.messages):
            content = text_content(message)
            summary = {
                "index": index,
                "role": str(getattr(message, "type", type(message).__name__)),
                "character_count": len(content),
                "content_sha256": digest_text(content.strip()),
            }
            if self.config.capture_content:
                summary["content"] = content
            messages.append(summary)
        self._active_writer().append(
            "model_request",
            {
                "messages": messages,
                "components": [component.model_dump(mode="json") for component in components],
                "prompt_components_v2": [
                    component.model_dump(mode="json") for component in components_v2
                ],
            },
            local_token_estimates={
                **totals,
                **{f"prompt_v2.{key}": value for key, value in totals_v2.items()},
            },
        )
        return ContextProjection(messages=event.messages)

    def after_model_response(self, event: ModelResponseObserved) -> None:
        content = text_content(event.response)
        usage = ProviderUsage(**extract_provider_usage(event.response))
        payload = self._text_summary("response", content)
        payload["finish_reason"] = self._finish_reason(event.response)
        payload["normalized_provider_traffic"] = normalize_provider_traffic(
            usage,
            input_accounting=self.config.input_token_accounting,
            cache_write_accounting=self.config.cache_write_accounting,
        ).model_dump(mode="json")
        self._active_writer().append(
            "model_response",
            payload,
            provider_usage=usage,
            local_token_estimates={"output": self.output_estimator.count(content)},
        )

    def before_execution(self, event: ExecutionDraft) -> ExecutionDecision:
        execution_key = digest_text(f"{event.language}\0{event.code.strip()}\0{self.config.environment_digest}")
        payload = {
            "language": event.language,
            "code_character_count": len(event.code),
            "code_sha256": digest_text(event.code),
            "execution_key": execution_key,
        }
        if self.config.capture_content:
            payload["code"] = event.code
        self._active_writer().append("execution_started", payload)
        return ExecutionDecision(code=event.code)

    def after_execution(self, event: RawExecutionObserved) -> ObservationProjection:
        tail = event.raw_result[10000:] if len(event.raw_result) > 10000 else ""
        error = execution_error_from_result(event.raw_result)
        execution_key = digest_text(f"{event.language}\0{event.code.strip()}\0{self.config.environment_digest}")
        payload = {
            "language": event.language,
            "execution_key": execution_key,
            "raw_result_character_count": len(event.raw_result),
            "raw_result_sha256": digest_text(event.raw_result.strip()),
            "legacy_projection_would_truncate": len(event.raw_result) > 10000,
            "tail_signal_detected": bool(TAIL_SIGNAL_PATTERN.search(tail)),
            "tail_character_count": len(tail),
        }
        if self.config.capture_content:
            payload["raw_result"] = event.raw_result
        self._active_writer().append("execution_finished", payload, error=error)
        return ObservationProjection(result=event.raw_result)

    def on_plan_update(self, event: PlanUpdated) -> None:
        self._active_writer().append(
            "plan_updated",
            {"source": event.source, "plan": event.plan},
        )

    def before_finalize(self, event: OutcomeDraft) -> FinalizationDecision:
        claims = self._claims(event.content)
        payload = self._text_summary("outcome", event.content)
        payload.update(
            {
                "claim_count": len(claims),
                "important_claim_count": len(claims),
                "grounded_claim_count": sum(bool(GROUNDING_PATTERN.search(claim)) for claim in claims),
                "claim_digests": [digest_text(claim) for claim in claims],
            }
        )
        self._active_writer().append("finalization", payload)
        return FinalizationDecision(content=event.content)

    def on_run_end(self, event: RunEnded) -> None:
        writer = self._active_writer()
        if event.status == "failed":
            writer.append(
                "error",
                {"source": "run", "detail_available": False},
                error=TraceError(error_type="run_failed", message="Run failed; upstream exception detail unavailable."),
            )
        payload = {"status": event.status}
        if event.final_content is not None:
            payload.update(self._text_summary("final", event.final_content))
        writer.append("run_ended", payload)
        self.writer = None

    def record_hook_failure(self, hook_name: str, error_type: str, error_message: str) -> None:
        self._active_writer().append(
            "hook_error",
            {"hook_name": hook_name},
            error=TraceError(error_type=error_type, message=error_message),
        )

    def record_resource_retrieval(
        self,
        *,
        prompt: str,
        selected: dict[str, Any] | None,
        candidate_counts: dict[str, int],
    ) -> None:
        selected = selected or {}
        selected_names = {
            category: [self._resource_name(item) for item in items]
            for category, items in selected.items()
            if isinstance(items, list)
        }
        self._active_writer().append(
            "resource_retrieval",
            {
                "prompt_sha256": digest_text(prompt),
                "prompt_character_count": len(prompt),
                "candidate_counts": candidate_counts,
                "selected": selected_names,
                "selected_counts": {key: len(value) for key, value in selected_names.items()},
            },
        )

    def _text_summary(self, label: str, text: str) -> dict[str, Any]:
        result = {
            f"{label}_character_count": len(text),
            f"{label}_sha256": digest_text(text.strip()),
        }
        if self.config.capture_content:
            result[label] = text
        return result

    @staticmethod
    def _finish_reason(response: Any) -> str | None:
        metadata = getattr(response, "response_metadata", None) or {}
        if isinstance(metadata, dict):
            return metadata.get("finish_reason") or metadata.get("stop_reason")
        return None

    @staticmethod
    def _claims(content: str) -> list[str]:
        plain = re.sub(r"</?(?:solution|think|execute)[^>]*>", " ", content, flags=re.IGNORECASE)
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", plain) if len(part.strip()) >= 12]

    @staticmethod
    def _resource_name(item: Any) -> str:
        if isinstance(item, dict):
            return str(item.get("name") or item.get("id") or "unknown")
        return str(item)


def manifest_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()
