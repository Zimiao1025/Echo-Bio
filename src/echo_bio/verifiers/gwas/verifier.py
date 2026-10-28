"""Deterministic GWAS minimum vertical-slice verifier."""

from __future__ import annotations

import csv
import io
import math

from echo_bio.artifacts.store import LocalArtifactStore
from echo_bio.verifiers.base import finding, read_json_artifact
from echo_bio.verifiers.contracts import VerificationBundle, VerificationFinding
from echo_bio.verifiers.statistics import close


class GwasVerifier:
    verifier_id = "gwas-l1-v1"

    def __init__(self, store: LocalArtifactStore) -> None:
        self.store = store

    def verify(self, bundle: VerificationBundle) -> tuple[VerificationFinding, ...]:
        rows = list(csv.DictReader(io.StringIO(self.store.get_text(bundle.input_artifact))))
        result = read_json_artifact(self.store, bundle.analysis_artifact.uri)
        refs = (bundle.input_artifact.uri, bundle.analysis_artifact.uri)
        required = {
            "variant_id",
            "effect_allele",
            "other_allele",
            "eaf",
            "beta",
            "se",
            "p_value",
            "info",
            "call_rate",
            "hwe_p",
        }
        alignment = result.get("allele_alignment", {})
        mapping_ok = bool(rows) and required.issubset(rows[0]) and all(
            alignment.get(row["variant_id"]) == {
                "effect_allele": row["effect_allele"],
                "other_allele": row["other_allele"],
                "beta": float(row["beta"]),
            }
            for row in rows
        )
        findings = [
            finding(
                verifier_id=self.verifier_id,
                level="L1",
                domain="gwas",
                criterion_id="variant_allele_effect_alignment",
                passed=mapping_ok,
                weight=0.07,
                hard_gate=True,
                message="Variant IDs, alleles, and signed effects align with source rows."
                if mapping_ok
                else "Variant/allele/effect alignment is incomplete or inconsistent.",
                evidence_refs=refs,
            )
        ]
        thresholds = result.get("qc_thresholds", {})
        threshold_ok = {"call_rate_min", "maf_min", "hwe_p_min", "info_min"}.issubset(thresholds)
        expected_failures = {"call_rate": [], "maf": [], "hwe": [], "info": []}
        if threshold_ok:
            for row in rows:
                maf = min(float(row["eaf"]), 1 - float(row["eaf"]))
                if float(row["call_rate"]) < float(thresholds["call_rate_min"]):
                    expected_failures["call_rate"].append(row["variant_id"])
                if maf < float(thresholds["maf_min"]):
                    expected_failures["maf"].append(row["variant_id"])
                if float(row["hwe_p"]) < float(thresholds["hwe_p_min"]):
                    expected_failures["hwe"].append(row["variant_id"])
                if float(row["info"]) < float(thresholds["info_min"]):
                    expected_failures["info"].append(row["variant_id"])
        qc_ok = threshold_ok and result.get("qc_failures") == expected_failures
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L1",
                domain="gwas",
                criterion_id="call_rate_maf_hwe_info_qc",
                passed=qc_ok,
                weight=0.08,
                hard_gate=True,
                message="Call-rate, MAF, HWE, and imputation-info failures recompute."
                if qc_ok
                else "QC thresholds or variant failure sets do not recompute.",
                evidence_refs=refs,
            )
        )
        population = result.get("population_control", {})
        population_ok = (
            int(population.get("ancestry_pc_count", 0)) >= 1
            and bool(population.get("relatedness_policy"))
            and bool(population.get("cohort"))
        )
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L1",
                domain="gwas",
                criterion_id="ancestry_relatedness_cohort",
                passed=population_ok,
                weight=0.06,
                hard_gate=True,
                message="Ancestry PCs, relatedness policy, and cohort are explicit."
                if population_ok
                else "Population structure or relatedness handling is incomplete.",
                evidence_refs=(bundle.analysis_artifact.uri,),
            )
        )
        association = result.get("association", {})
        model_ok = bool(association.get("model")) and "ancestry_pcs" in association.get("covariates", [])
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L1",
                domain="gwas",
                criterion_id="association_model_and_covariates",
                passed=model_ok,
                weight=0.06,
                hard_gate=True,
                message="Association model and ancestry covariates are declared."
                if model_ok
                else "Association model or covariates are missing.",
                evidence_refs=(bundle.analysis_artifact.uri,),
            )
        )
        statistics_rows = result.get("results", [])
        source_by_id = {row["variant_id"]: row for row in rows}
        stats_ok = len(statistics_rows) == len(rows) and all(
            item.get("variant_id") in source_by_id
            and close(float(item["beta"]), float(source_by_id[item["variant_id"]]["beta"]))
            and close(float(item["se"]), float(source_by_id[item["variant_id"]]["se"]))
            and close(float(item["p_value"]), float(source_by_id[item["variant_id"]]["p_value"]))
            for item in statistics_rows
        )
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L1",
                domain="gwas",
                criterion_id="association_statistic_readback",
                passed=stats_ok,
                weight=0.06,
                hard_gate=True,
                message="Beta, SE, and p-value fields read back exactly from source rows."
                if stats_ok
                else "Association statistic readback failed.",
                evidence_refs=refs,
            )
        )
        interval_ok = stats_ok and all(
            close(float(item["ci_low"]), float(item["beta"]) - 1.96 * float(item["se"]))
            and close(float(item["ci_high"]), float(item["beta"]) + 1.96 * float(item["se"]))
            and close(float(item["odds_ratio"]), math.exp(float(item["beta"])))
            for item in statistics_rows
        )
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L1",
                domain="gwas",
                criterion_id="effect_interval_or_recomputation",
                passed=interval_ok,
                weight=0.06,
                hard_gate=True,
                message="Confidence intervals and odds ratios recompute from beta and SE."
                if interval_ok
                else "Confidence intervals or odds ratios fail recomputation.",
                evidence_refs=(bundle.analysis_artifact.uri,),
            )
        )
        threshold = float(result.get("significance_threshold", -1))
        expected_significant = sorted(row["variant_id"] for row in rows if float(row["p_value"]) < threshold)
        multiplicity_ok = (
            close(threshold, 5e-8)
            and sorted(result.get("significant_variants", [])) == expected_significant
        )
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L1",
                domain="gwas",
                criterion_id="declared_significance_threshold",
                passed=multiplicity_ok,
                weight=0.06,
                hard_gate=True,
                message="Genome-wide threshold and significant-variant set recompute."
                if multiplicity_ok
                else "Significance threshold or variant set is inconsistent.",
                evidence_refs=refs,
            )
        )
        downstream = result.get("downstream_evidence", {})
        downstream_ok = all(
            downstream.get(key) in {"not_performed", "unavailable", "available"}
            for key in ("ld", "fine_mapping", "colocalization")
        )
        findings.append(
            finding(
                verifier_id=self.verifier_id,
                level="L1",
                domain="gwas",
                criterion_id="ld_finemapping_colocalization_state",
                passed=downstream_ok,
                weight=0.05,
                hard_gate=False,
                message="LD, fine-mapping, and colocalization evidence states are explicit."
                if downstream_ok
                else "Downstream locus-evidence status is missing.",
                evidence_refs=(bundle.analysis_artifact.uri,),
            )
        )
        return tuple(findings)
