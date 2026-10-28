"""Biomni adapter with Phase 4 receipt and claim-evidence recording."""

from __future__ import annotations

from typing import Any

from echo_bio.adapters.biomni.agent import EchoBiomniAgent
from echo_bio.adapters.biomni.artifact_agent import LosslessObservationBiomniAgent
from echo_bio.artifacts.preview import TypedPreviewer
from echo_bio.artifacts.projection import ObservationProjector, ProjectionConfig
from echo_bio.artifacts.recall import ArtifactRecallService
from echo_bio.artifacts.store import LocalArtifactStore
from echo_bio.evidence.hooks import EvidenceBackedTelemetryHooks
from echo_bio.evidence.ledger import EvidenceLedger
from echo_bio.telemetry.hooks import TelemetryConfig
from echo_bio.telemetry.ledger import TraceLedger


class EvidenceBiomniAgent(LosslessObservationBiomniAgent):
    def __init__(
        self,
        *args: Any,
        trace_ledger: TraceLedger,
        telemetry_config: TelemetryConfig,
        artifact_store: LocalArtifactStore,
        evidence_ledger: EvidenceLedger,
        projection_config: ProjectionConfig | None = None,
        **kwargs: Any,
    ) -> None:
        projector = ObservationProjector(TypedPreviewer(artifact_store), projection_config)
        hooks = EvidenceBackedTelemetryHooks(
            trace_ledger,
            telemetry_config,
            artifact_store=artifact_store,
            projector=projector,
            evidence_ledger=evidence_ledger,
        )
        self.telemetry_hooks = hooks
        self._telemetry_failure_cursor = 0
        self.artifact_store = artifact_store
        self.artifact_recall_service = ArtifactRecallService(artifact_store)
        self.evidence_ledger = evidence_ledger
        EchoBiomniAgent.__init__(self, *args, harness_hooks=hooks, **kwargs)
