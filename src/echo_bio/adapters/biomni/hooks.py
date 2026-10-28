"""Fail-open hook bus used by the Phase 0 Biomni adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeVar, runtime_checkable

from echo_bio.contracts import (
    ContextProjection,
    ExecutionDecision,
    ExecutionDraft,
    FinalizationDecision,
    HookFailure,
    ModelRequestDraft,
    ModelResponseObserved,
    ObservationProjection,
    OutcomeDraft,
    PlanUpdated,
    RawExecutionObserved,
    RunEnded,
    RunStart,
)


@runtime_checkable
class HarnessHooks(Protocol):
    def on_run_start(self, event: RunStart) -> None: ...

    def before_model_request(self, event: ModelRequestDraft) -> ContextProjection: ...

    def after_model_response(self, event: ModelResponseObserved) -> None: ...

    def before_execution(self, event: ExecutionDraft) -> ExecutionDecision: ...

    def after_execution(self, event: RawExecutionObserved) -> ObservationProjection: ...

    def on_plan_update(self, event: PlanUpdated) -> None: ...

    def before_finalize(self, event: OutcomeDraft) -> FinalizationDecision: ...

    def on_run_end(self, event: RunEnded) -> None: ...


class NoOpHarnessHooks:
    """Identity implementation that preserves every provider-visible value."""

    def on_run_start(self, event: RunStart) -> None:
        return None

    def before_model_request(self, event: ModelRequestDraft) -> ContextProjection:
        return ContextProjection(messages=event.messages)

    def after_model_response(self, event: ModelResponseObserved) -> None:
        return None

    def before_execution(self, event: ExecutionDraft) -> ExecutionDecision:
        return ExecutionDecision(code=event.code)

    def after_execution(self, event: RawExecutionObserved) -> ObservationProjection:
        return ObservationProjection(result=event.raw_result)

    def on_plan_update(self, event: PlanUpdated) -> None:
        return None

    def before_finalize(self, event: OutcomeDraft) -> FinalizationDecision:
        return FinalizationDecision(content=event.content)

    def on_run_end(self, event: RunEnded) -> None:
        return None


T = TypeVar("T")


class SafeHarnessHookBus:
    """Validate hook returns and fail open while retaining an auditable failure list."""

    def __init__(self, hooks: HarnessHooks | None = None) -> None:
        self.hooks: HarnessHooks = hooks or NoOpHarnessHooks()
        self.failures: list[HookFailure] = []

    def _failure(self, hook_name: str, error: Exception, event: Any) -> None:
        self.failures.append(
            HookFailure(
                hook_name=hook_name,
                error_type=type(error).__name__,
                error_message=str(error),
                identity=getattr(event, "identity", None),
            )
        )

    def _project(
        self,
        hook_name: str,
        callback: Callable[[Any], Any],
        event: Any,
        expected_type: type[T],
        fallback: T,
    ) -> T:
        try:
            result = callback(event)
            if not isinstance(result, expected_type):
                raise TypeError(f"{hook_name} returned {type(result).__name__}; expected {expected_type.__name__}")
            return result
        except Exception as error:
            self._failure(hook_name, error, event)
            return fallback

    def _observe(self, hook_name: str, callback: Callable[[Any], Any], event: Any) -> None:
        try:
            callback(event)
        except Exception as error:
            self._failure(hook_name, error, event)

    def on_run_start(self, event: RunStart) -> None:
        self._observe("on_run_start", self.hooks.on_run_start, event)

    def before_model_request(self, event: ModelRequestDraft) -> ContextProjection:
        return self._project(
            "before_model_request",
            self.hooks.before_model_request,
            event,
            ContextProjection,
            ContextProjection(messages=event.messages),
        )

    def after_model_response(self, event: ModelResponseObserved) -> None:
        self._observe("after_model_response", self.hooks.after_model_response, event)

    def before_execution(self, event: ExecutionDraft) -> ExecutionDecision:
        return self._project(
            "before_execution",
            self.hooks.before_execution,
            event,
            ExecutionDecision,
            ExecutionDecision(code=event.code),
        )

    def after_execution(self, event: RawExecutionObserved) -> ObservationProjection:
        return self._project(
            "after_execution",
            self.hooks.after_execution,
            event,
            ObservationProjection,
            ObservationProjection(result=event.raw_result),
        )

    def on_plan_update(self, event: PlanUpdated) -> None:
        self._observe("on_plan_update", self.hooks.on_plan_update, event)

    def before_finalize(self, event: OutcomeDraft) -> FinalizationDecision:
        return self._project(
            "before_finalize",
            self.hooks.before_finalize,
            event,
            FinalizationDecision,
            FinalizationDecision(content=event.content),
        )

    def on_run_end(self, event: RunEnded) -> None:
        self._observe("on_run_end", self.hooks.on_run_end, event)
