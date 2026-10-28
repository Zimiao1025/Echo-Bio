from __future__ import annotations

import pytest

from echo_bio.artifacts.store import LocalArtifactStore
from echo_bio.evidence.claim_linker import ClaimEvidenceAssembler, LocatorDraft, StructuredClaimDraft
from echo_bio.evidence.diagnostic_receipt import ReceiptValidationError


def test_statistical_claim_resolves_to_table_cell_and_reports_coverage(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    table = store.put_text(
        run_id="run-claim",
        text="feature,effect,p_adj\nGENE1,1.2,0.004\nGENE2,-0.3,0.8\n",
        media_type="text/csv",
        data_class="public",
    )
    claims = (
        StructuredClaimDraft(
            text="GENE1 has adjusted p=0.004.",
            claim_type="statistical",
            scope="constructed differential-expression table",
            evidence=(LocatorDraft(artifact_uri=table.uri, kind="table_cell", row=0, column="p_adj"),),
        ),
        StructuredClaimDraft(
            text="The pathway interpretation remains uncertain.",
            claim_type="mechanistic",
            scope="constructed fixture",
            limitations=("No perturbation experiment",),
        ),
    )
    graph = ClaimEvidenceAssembler(store).assemble(run_id="run-claim", claims=claims)
    assert graph.important_claim_count == 2
    assert graph.linked_important_claim_count == 1
    assert graph.important_claim_linkage_coverage == 0.5
    assert graph.entailment_assessed is False


def test_numeric_claim_rejects_artifact_only_locator(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    table = store.put_text(run_id="run-claim", text="effect\n1.2\n", data_class="public")
    claim = StructuredClaimDraft(
        text="The effect was 1.2.",
        claim_type="statistical",
        scope="fixture",
        evidence=(LocatorDraft(artifact_uri=table.uri, kind="artifact"),),
    )
    with pytest.raises(ReceiptValidationError, match="statistical claims require"):
        ClaimEvidenceAssembler(store).assemble(run_id="run-claim", claims=(claim,))


def test_json_pointer_must_read_back(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    result = store.put_text(
        run_id="run-claim",
        text='{"metrics":{"fdr":0.01}}',
        media_type="application/json",
        data_class="public",
    )
    good = StructuredClaimDraft(
        text="The FDR was 0.01.",
        claim_type="statistical",
        scope="fixture",
        evidence=(LocatorDraft(artifact_uri=result.uri, kind="json_pointer", json_pointer="/metrics/fdr"),),
    )
    graph = ClaimEvidenceAssembler(store).assemble(run_id="run-claim", claims=(good,))
    assert graph.important_claim_linkage_coverage == 1.0
    bad = StructuredClaimDraft(
        text="The FDR was 0.02.",
        claim_type="statistical",
        scope="fixture",
        evidence=(LocatorDraft(artifact_uri=result.uri, kind="json_pointer", json_pointer="/missing"),),
    )
    with pytest.raises(ReceiptValidationError, match="not reachable"):
        ClaimEvidenceAssembler(store).assemble(run_id="run-claim", claims=(bad,))
