"""Deterministic bulk RNA-seq minimum vertical-slice verifier."""

from __future__ import annotations

import csv
import io
import statistics

from echo_bio.artifacts.store import LocalArtifactStore
from echo_bio.verifiers.base import finding, read_json_artifact
from echo_bio.verifiers.contracts import VerificationBundle, VerificationFinding
from echo_bio.verifiers.statistics import benjamini_hochberg, close


class BulkRnaSeqVerifier:
    verifier_id = "bulk-rnaseq-l1-v1"

    def __init__(self, store: LocalArtifactStore) -> None:
        self.store = store

    def verify(self, bundle: VerificationBundle) -> tuple[VerificationFinding, ...]:
        rows = list(csv.DictReader(io.StringIO(self.store.get_text(bundle.input_artifact))))
        result = read_json_artifact(self.store, bundle.analysis_artifact.uri)
        refs = (bundle.input_artifact.uri, bundle.analysis_artifact.uri)
        required = {"sample_id", "condition", "batch"}
        mapping_ok = bool(rows) and required.issubset(rows[0]) and len({row["sample_id"] for row in rows}) == len(rows)
        replicate_counts = {
            condition: sum(row["condition"] == condition for row in rows)
            for condition in {row["condition"] for row in rows}
        }
        mapping_ok = (
            mapping_ok
            and result.get("replicate_counts") == replicate_counts
            and min(replicate_counts.values()) >= 2
        )
        findings = [
            finding(
                verifier_id=self.verifier_id,
                level="L1",
                domain="bulk_rnaseq",
                criterion_id="sample_mapping_and_replicates",
                passed=mapping_ok,
                weight=0.06,
                hard_gate=True,
                message="Sample IDs are unique and condition replicate counts recompute."
                if mapping_ok
                else "Sample mapping or biological replicate counts are incomplete.",
                evidence_refs=refs,
            )
        ]
        gene_columns = [column for column in rows[0] if column not in {"sample_id", "condition", "batch", "sex"}]
        library_sizes = {row["sample_id"]: sum(float(row[gene]) for gene in gene_columns) for row in rows}
        median_size = statistics.median(library_sizes.values())
        expected_factors = {sample: value / median_size for sample, value in library_sizes.items()}
        observed_factors = result.get("normalization", {}).get("size_factors", {})
        normalization_ok = result.get("normalization", {}).get("method") == "library_size" and set(
            observed_factors
        ) == set(expected_factors)
        normalization_ok = normalization_ok and all(
            close(float(observed_factors[sample]), value) for sample, value in expected_factors.items()
        )
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L1",
                domain="bulk_rnaseq",
                criterion_id="normalization_recomputation",
                passed=normalization_ok,
                weight=0.07,
                hard_gate=True,
                message="Library-size normalization factors recompute from the count matrix."
                if normalization_ok
                else "Normalization method or size factors do not recompute.",
                evidence_refs=refs,
            )
        )
        design = result.get("design", {})
        design_ok = design.get("formula") == "~ batch + condition" and design.get("contrast") == "treated-control"
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L1",
                domain="bulk_rnaseq",
                criterion_id="design_formula_and_contrast",
                passed=design_ok,
                weight=0.07,
                hard_gate=True,
                message="Design and treated-versus-control contrast are explicit."
                if design_ok
                else "Design formula or contrast is missing or inconsistent.",
                evidence_refs=(bundle.analysis_artifact.uri,),
            )
        )
        covariate_ok = "batch" in design.get("covariates", []) and {row["batch"] for row in rows}
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L1",
                domain="bulk_rnaseq",
                criterion_id="batch_and_covariates",
                passed=bool(covariate_ok),
                weight=0.05,
                hard_gate=True,
                message="Batch is represented as an explicit model covariate."
                if covariate_ok
                else "Batch/covariate handling is missing.",
                evidence_refs=refs,
            )
        )
        diagnostics = result.get("diagnostics", {})
        diagnostics_ok = diagnostics.get("dispersion_assessed") is True and diagnostics.get("outlier_assessed") is True
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L1",
                domain="bulk_rnaseq",
                criterion_id="dispersion_and_outlier_diagnostics",
                passed=diagnostics_ok,
                weight=0.06,
                hard_gate=False,
                message="Dispersion and outlier diagnostics are explicitly recorded."
                if diagnostics_ok
                else "Dispersion or outlier assessment is missing.",
                evidence_refs=(bundle.analysis_artifact.uri,),
            )
        )
        statistics_rows = result.get("results", [])
        schema_ok = bool(statistics_rows) and all(
            {"gene", "log_fc", "se", "ci_low", "ci_high", "p_value", "p_adj"}.issubset(item)
            for item in statistics_rows
        )
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L1",
                domain="bulk_rnaseq",
                criterion_id="de_result_structure",
                passed=schema_ok,
                weight=0.06,
                hard_gate=True,
                message="DE results contain effect, uncertainty, raw p, and adjusted p fields."
                if schema_ok
                else "Differential-expression result fields are incomplete.",
                evidence_refs=(bundle.analysis_artifact.uri,),
            )
        )
        interval_ok = schema_ok and all(
            float(item["se"]) >= 0 and float(item["ci_low"]) <= float(item["log_fc"]) <= float(item["ci_high"])
            for item in statistics_rows
        )
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L1",
                domain="bulk_rnaseq",
                criterion_id="effect_se_interval_consistency",
                passed=interval_ok,
                weight=0.06,
                hard_gate=True,
                message="All effects, standard errors, and confidence intervals are internally consistent."
                if interval_ok
                else "Effect uncertainty fields are absent or inconsistent.",
                evidence_refs=(bundle.analysis_artifact.uri,),
            )
        )
        p_values = [float(item["p_value"]) for item in statistics_rows if "p_value" in item]
        adjusted = benjamini_hochberg(p_values)
        multiplicity_ok = schema_ok and len(p_values) == len(statistics_rows) and all(
            close(float(item["p_adj"]), expected)
            for item, expected in zip(statistics_rows, adjusted, strict=True)
        )
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L1",
                domain="bulk_rnaseq",
                criterion_id="multiplicity_recomputation",
                passed=multiplicity_ok,
                weight=0.07,
                hard_gate=True,
                message="Adjusted p-values match Benjamini-Hochberg recomputation."
                if multiplicity_ok
                else "Multiplicity correction fails deterministic recomputation.",
                evidence_refs=(bundle.analysis_artifact.uri,),
            )
        )
        return tuple(findings)
