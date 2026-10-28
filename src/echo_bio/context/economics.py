"""Deterministic compaction economics, separate from scientific validity gates."""

from __future__ import annotations

from echo_bio.context.contracts import CompactionEconomics, ContextCompilerConfig


def evaluate_compaction(
    source_tokens: int,
    candidate_tokens: int,
    config: ContextCompilerConfig,
) -> CompactionEconomics:
    per_request = source_tokens - candidate_tokens
    gross = per_request * config.estimated_remaining_requests
    total_cost = config.summary_cost_tokens + config.cache_write_cost_tokens + config.cache_read_loss_tokens
    net = gross - total_cost
    source_utilization = source_tokens / config.context_window_tokens
    candidate_utilization = candidate_tokens / config.context_window_tokens
    return CompactionEconomics(
        source_tokens=source_tokens,
        candidate_tokens=candidate_tokens,
        estimated_remaining_requests=config.estimated_remaining_requests,
        per_request_savings_tokens=per_request,
        gross_future_savings_tokens=gross,
        summary_cost_tokens=config.summary_cost_tokens,
        cache_write_cost_tokens=config.cache_write_cost_tokens,
        cache_read_loss_tokens=config.cache_read_loss_tokens,
        total_compaction_cost_tokens=total_cost,
        net_savings_tokens=net,
        source_window_utilization=source_utilization,
        candidate_window_utilization=candidate_utilization,
        window_risk_reduction=source_utilization - candidate_utilization,
        economically_eligible=per_request > 0 and net >= config.minimum_net_savings_tokens,
    )
