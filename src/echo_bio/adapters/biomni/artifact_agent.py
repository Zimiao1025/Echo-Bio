"""Biomni adapter with Phase 2 lossless observations and optional projections."""

from __future__ import annotations

from typing import Any

from echo_bio.adapters.biomni.agent import EchoBiomniAgent
from echo_bio.adapters.biomni.telemetry_agent import TelemetryBiomniAgent
from echo_bio.artifacts.hooks import ArtifactBackedTelemetryHooks
from echo_bio.artifacts.preview import TypedPreviewer
from echo_bio.artifacts.projection import ObservationProjector, ProjectionConfig
from echo_bio.artifacts.recall import ArtifactRecallService
from echo_bio.artifacts.store import LocalArtifactStore
from echo_bio.telemetry.hooks import TelemetryConfig
from echo_bio.telemetry.ledger import TraceLedger


class LosslessObservationBiomniAgent(TelemetryBiomniAgent):
    def __init__(
        self,
        *args: Any,
        trace_ledger: TraceLedger,
        telemetry_config: TelemetryConfig,
        artifact_store: LocalArtifactStore,
        projection_config: ProjectionConfig | None = None,
        **kwargs: Any,
    ) -> None:
        projector = ObservationProjector(TypedPreviewer(artifact_store), projection_config)
        hooks = ArtifactBackedTelemetryHooks(
            trace_ledger,
            telemetry_config,
            artifact_store=artifact_store,
            projector=projector,
        )
        self.telemetry_hooks = hooks
        self._telemetry_failure_cursor = 0
        self.artifact_store = artifact_store
        self.artifact_recall_service = ArtifactRecallService(artifact_store)
        # Intentionally bypass TelemetryBiomniAgent.__init__, which constructs Phase 1 hooks.
        EchoBiomniAgent.__init__(self, *args, harness_hooks=hooks, **kwargs)

    def _project_execution_result(self, result: str) -> str:
        if self.telemetry_hooks.projector.config.mode == "treatment":
            # Phase 2 already returned either a verified projection or, on failure,
            # the complete original observation. Do not apply the legacy truncation twice.
            return result
        return super()._project_execution_result(result)

    def artifact_recall(
        self,
        uri: str,
        *,
        mode: str = "lines",
        offset: int = 0,
        length: int | None = None,
        start_line: int = 1,
        line_count: int = 100,
        columns: tuple[str, ...] = (),
        keys: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Read a bounded slice; suitable for registration as a Biomni custom tool."""
        if mode == "bytes":
            result = self.artifact_recall_service.bytes(uri, offset=offset, length=length)
        elif mode == "lines":
            result = self.artifact_recall_service.lines(uri, start_line=start_line, line_count=line_count)
        elif mode == "table":
            result = self.artifact_recall_service.table(uri, columns=columns, offset=offset, limit=line_count)
        elif mode == "json_keys":
            result = self.artifact_recall_service.json_keys(uri, keys=keys)
        else:
            raise ValueError(f"unsupported artifact recall mode: {mode}")
        return result.model_dump(mode="json")
