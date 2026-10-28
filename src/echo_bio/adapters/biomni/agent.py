"""Biomni A1 subclass translating its minimal seam into Echo-Bio contracts."""

from __future__ import annotations

from typing import Any

from biomni.agent.a1 import A1

from echo_bio.adapters.biomni.hooks import HarnessHooks, SafeHarnessHookBus
from echo_bio.contracts import (
    ExecutionDraft,
    ModelRequestDraft,
    ModelResponseObserved,
    OutcomeDraft,
    PlanUpdated,
    RawExecutionObserved,
    RunEnded,
    RunIdentity,
    RunStart,
)


class EchoBiomniAgent(A1):
    """A1 with typed hooks; the default hook bus is exactly identity behavior."""

    def __init__(self, *args: Any, harness_hooks: HarnessHooks | None = None, **kwargs: Any) -> None:
        self.harness = SafeHarnessHookBus(harness_hooks)
        self._echo_identity: RunIdentity | None = None
        super().__init__(*args, **kwargs)

    @property
    def hook_failures(self):
        return tuple(self.harness.failures)

    def _identity(self) -> RunIdentity:
        if self._echo_identity is None:
            raise RuntimeError("Biomni emitted a harness event outside an active run")
        return self._echo_identity

    def _on_run_start(self, run_identity: dict[str, str], prompt: str) -> None:
        self._echo_identity = RunIdentity(**run_identity)
        self.harness.on_run_start(RunStart(identity=self._echo_identity, task=prompt))

    def _before_model_request(self, messages):
        event = ModelRequestDraft(identity=self._identity(), messages=tuple(messages))
        return list(self.harness.before_model_request(event).messages)

    def _after_model_response(self, response: Any, requested_messages) -> None:
        self.harness.after_model_response(
            ModelResponseObserved(
                identity=self._identity(),
                response=response,
                requested_messages=tuple(requested_messages),
            )
        )

    def _before_execution(self, code: str, language: str) -> str:
        event = ExecutionDraft(identity=self._identity(), code=code, language=language)
        decision = self.harness.before_execution(event)
        if not decision.allow:
            # Phase 0 has no policy-denial route. Fail closed only when a future
            # non-identity hook explicitly requests denial.
            raise PermissionError(f"Execution denied by harness policy {decision.policy_id}")
        return decision.code

    def _after_execution(self, code: str, language: str, raw_result: str) -> str:
        event = RawExecutionObserved(
            identity=self._identity(),
            code=code,
            language=language,
            raw_result=raw_result,
        )
        return self.harness.after_execution(event).result

    def _before_finalize(self, content: str) -> str:
        decision = self.harness.before_finalize(OutcomeDraft(identity=self._identity(), content=content))
        if not decision.allow:
            raise PermissionError(f"Finalization denied by harness policy {decision.policy_id}")
        return decision.content

    def emit_plan_update(self, plan: dict[str, Any], *, source: str) -> None:
        """Explicit bridge for future typed planners; Biomni A1 has no plan node today."""
        self.harness.on_plan_update(PlanUpdated(identity=self._identity(), plan=plan, source=source))

    def _on_run_end(self, run_identity: dict[str, str], status: str, final_content: str | None) -> None:
        identity = self._echo_identity or RunIdentity(**run_identity)
        self.harness.on_run_end(RunEnded(identity=identity, status=status, final_content=final_content))
        self._echo_identity = None
