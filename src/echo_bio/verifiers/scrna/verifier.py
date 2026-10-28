"""Deterministic scRNA minimum vertical-slice verifier."""

from __future__ import annotations

import csv
import io

from echo_bio.artifacts.store import LocalArtifactStore
from echo_bio.verifiers.base import finding, read_json_artifact
from echo_bio.verifiers.contracts import VerificationBundle, VerificationFinding
from echo_bio.verifiers.statistics import benjamini_hochberg, close


class ScrnaVerifier:
    verifier_id = "scrna-l1-v1"

    def __init__(self, store: LocalArtifactStore) -> None:
        self.store = store

    def verify(self, bundle: VerificationBundle) -> tuple[VerificationFinding, ...]:
        rows = list(csv.DictReader(io.StringIO(self.store.get_text(bundle.input_artifact))))
        result = read_json_artifact(self.store, bundle.analysis_artifact.uri)
        refs = (bundle.input_artifact.uri, bundle.analysis_artifact.uri)
        required = {"cell_id", "donor", "condition", "batch", "cell_type", "counts_total", "genes_detected", "pct_mito"}
        mapping_ok = bool(rows) and required.issubset(rows[0]) and len({row["cell_id"] for row in rows}) == len(rows)
        mapping_ok = mapping_ok and all(row["donor"] and row["condition"] for row in rows)
        findings = [
            finding(
                verifier_id=self.verifier_id,
                level="L1",
                domain="scrna",
                criterion_id="cell_donor_sample_mapping",
                passed=mapping_ok,
                weight=0.06,
                hard_gate=True,
                message="Cell identifiers are unique and donor/condition mappings are complete."
                if mapping_ok
                else "Cell, donor, or condition mapping is incomplete or duplicated.",
                evidence_refs=refs,
            )
        ]
        thresholds = result.get("qc_thresholds", {})
        threshold_ok = {"counts_total_min", "genes_detected_min", "pct_mito_max"}.issubset(thresholds)
        expected_failed = []
        if threshold_ok:
            expected_failed = sorted(
                row["cell_id"]
                for row in rows
                if float(row["counts_total"]) < float(thresholds["counts_total_min"])
                or float(row["genes_detected"]) < float(thresholds["genes_detected_min"])
                or float(row["pct_mito"]) > float(thresholds["pct_mito_max"])
            )
        qc_ok = threshold_ok and sorted(result.get("failed_cells", [])) == expected_failed
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L1",
                domain="scrna",
                criterion_id="qc_thresholds_and_low_quality_cells",
                passed=qc_ok,
                weight=0.07,
                hard_gate=True,
                message=f"Recomputed {len(expected_failed)} failed cells from explicit QC thresholds."
                if qc_ok
                else "QC thresholds are missing or the failed-cell set does not recompute.",
                evidence_refs=refs,
            )
        )
        doublet = result.get("doublet_policy", {})
        doublet_ok = bool(doublet.get("method")) and isinstance(doublet.get("flagged_cells"), list)
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L1",
                domain="scrna",
                criterion_id="doublet_handling",
                passed=doublet_ok,
                weight=0.05,
                hard_gate=False,
                message="Doublet method and flagged-cell set are explicit."
                if doublet_ok
                else "Doublet handling is not explicitly represented.",
                evidence_refs=(bundle.analysis_artifact.uri,),
            )
        )
        donor_counts = {donor: sum(row["donor"] == donor for row in rows) for donor in {row["donor"] for row in rows}}
        donor_ok = result.get("analysis_unit") == "donor" and result.get("donor_cell_counts") == donor_counts
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L1",
                domain="scrna",
                criterion_id="donor_aware_no_pseudoreplication",
                passed=donor_ok,
                weight=0.07,
                hard_gate=True,
                message="Analysis unit is donor and donor cell counts recompute."
                if donor_ok
                else "Analysis unit is not donor or donor summaries do not recompute.",
                evidence_refs=refs,
            )
        )
        comparison = result.get("comparison", {})
        comparison_ok = all(comparison.get(key) for key in ("group_a", "group_b", "cell_type", "comparison_set"))
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L1",
                domain="scrna",
                criterion_id="comparison_set_prespecified",
                passed=comparison_ok,
                weight=0.05,
                hard_gate=True,
                message="Groups, cell type, and comparison set are explicit."
                if comparison_ok
                else "The comparison set is incomplete or unspecified.",
                evidence_refs=(bundle.analysis_artifact.uri,),
            )
        )
        statistics = result.get("statistics", [])
        p_values = [float(item["p_value"]) for item in statistics if "p_value" in item]
        expected_adjusted = benjamini_hochberg(p_values)
        fdr_ok = bool(statistics) and len(p_values) == len(statistics) and all(
            "p_adj" in item and close(float(item["p_adj"]), expected)
            for item, expected in zip(statistics, expected_adjusted, strict=True)
        )
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L1",
                domain="scrna",
                criterion_id="fdr_recomputation",
                passed=fdr_ok,
                weight=0.07,
                hard_gate=True,
                message="Adjusted p-values match Benjamini-Hochberg recomputation."
                if fdr_ok
                else "Adjusted p-values are absent or fail Benjamini-Hochberg recomputation.",
                evidence_refs=(bundle.analysis_artifact.uri,),
            )
        )
        effect_ok = bool(statistics) and all(
            {"effect", "ci_low", "ci_high"}.issubset(item)
            and float(item["ci_low"]) <= float(item["effect"]) <= float(item["ci_high"])
            for item in statistics
        )
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L1",
                domain="scrna",
                criterion_id="effect_and_interval_structure",
                passed=effect_ok,
                weight=0.07,
                hard_gate=True,
                message="Every statistical row includes an effect within its interval."
                if effect_ok
                else "An effect or uncertainty interval is missing or inconsistent.",
                evidence_refs=(bundle.analysis_artifact.uri,),
            )
        )
        annotation = result.get("annotation", {})
        annotation_ok = annotation.get("checked") is True and isinstance(annotation.get("ambiguities"), list)
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L1",
                domain="scrna",
                criterion_id="annotation_output_structure",
                passed=annotation_ok,
                weight=0.06,
                hard_gate=False,
                message="Annotation check and ambiguity list are explicit."
                if annotation_ok
                else "Annotation validation or ambiguity preservation is missing.",
                evidence_refs=(bundle.analysis_artifact.uri,),
            )
        )
        return tuple(findings)
