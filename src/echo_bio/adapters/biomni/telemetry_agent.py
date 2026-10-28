"""Biomni adapter variant that adds read-only Phase 1 telemetry."""

from __future__ import annotations

from typing import Any

from echo_bio.adapters.biomni.agent import EchoBiomniAgent
from echo_bio.telemetry.hooks import TelemetryConfig, TelemetryHarnessHooks
from echo_bio.telemetry.ledger import TraceLedger


class TelemetryBiomniAgent(EchoBiomniAgent):
    def __init__(
        self,
        *args: Any,
        trace_ledger: TraceLedger,
        telemetry_config: TelemetryConfig,
        **kwargs: Any,
    ) -> None:
        self.telemetry_hooks = TelemetryHarnessHooks(trace_ledger, telemetry_config)
        self._telemetry_failure_cursor = 0
        super().__init__(*args, harness_hooks=self.telemetry_hooks, **kwargs)

    def _prepare_resources_for_retrieval(self, prompt):
        selected = super()._prepare_resources_for_retrieval(prompt)
        candidate_counts = {
            "tools": len(getattr(getattr(self, "tool_registry", None), "tools", [])),
            "data_lake": len(getattr(self, "data_lake_dict", {})),
            "libraries": len(getattr(self, "library_content_dict", {})),
            "know_how": len(getattr(getattr(self, "know_how_loader", None), "documents", {})),
        }
        try:
            self.telemetry_hooks.record_resource_retrieval(
                prompt=prompt,
                selected=selected,
                candidate_counts=candidate_counts,
            )
        except Exception as error:
            # Retrieval itself must remain unaffected by read-only instrumentation.
            self.harness._failure("resource_retrieval", error, None)
        return selected

    def _on_run_end(self, run_identity: dict[str, str], status: str, final_content: str | None) -> None:
        failures = self.harness.failures[self._telemetry_failure_cursor :]
        for failure in failures:
            try:
                self.telemetry_hooks.record_hook_failure(
                    failure.hook_name,
                    failure.error_type,
                    failure.error_message,
                )
            except Exception:
                # The in-memory SafeHarnessHookBus remains the fallback audit record.
                break
        self._telemetry_failure_cursor = len(self.harness.failures)
        super()._on_run_end(run_identity, status, final_content)
