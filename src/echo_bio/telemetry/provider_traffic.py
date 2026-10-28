"""Provider-token traffic normalization without cache double counting."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from echo_bio.contracts.trace import ProviderUsage

InputTokenAccounting = Literal[
    "inclusive_cache_read",
    "exclusive_cache",
    "unknown",
]
CacheWriteAccounting = Literal["reported", "not_applicable", "unknown"]


class TrafficContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NormalizedProviderTraffic(TrafficContract):
    """A lossless view of usage plus a guarded SoL-Pi-style total.

    ``raw_input_tokens`` preserves the provider value.  ``uncached_input_tokens``
    is emitted only when input/cache semantics are explicit.  Missing cache-write
    usage is never silently treated as zero unless the API contract declares that
    category not applicable.
    """

    schema_version: Literal["echo.normalized-provider-traffic.v1"] = (
        "echo.normalized-provider-traffic.v1"
    )
    input_accounting: InputTokenAccounting
    cache_write_accounting: CacheWriteAccounting
    raw_input_tokens: int | None = Field(default=None, ge=0)
    uncached_input_tokens: int | None = Field(default=None, ge=0)
    cache_read_tokens: int | None = Field(default=None, ge=0)
    cache_write_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    total_recorded_traffic: int | None = Field(default=None, ge=0)
    lower_bound_traffic: int = Field(ge=0)
    sol_pi_comparable: bool
    limitations: tuple[str, ...] = ()


def normalize_provider_traffic(
    usage: ProviderUsage,
    *,
    input_accounting: InputTokenAccounting,
    cache_write_accounting: CacheWriteAccounting,
) -> NormalizedProviderTraffic:
    """Normalize one usage record under explicit, frozen accounting semantics."""

    raw_input = usage.input_tokens
    output = usage.output_tokens
    cache_read = usage.cache_read_tokens
    reported_cache_write = usage.cache_write_tokens
    limitations: list[str] = []

    uncached: int | None = None
    if input_accounting == "inclusive_cache_read":
        if raw_input is None or cache_read is None:
            limitations.append("inclusive input accounting requires input and cache-read usage")
        elif cache_read > raw_input:
            limitations.append("cache-read usage exceeds raw input; accounting is inconsistent")
        else:
            uncached = raw_input - cache_read
    elif input_accounting == "exclusive_cache":
        if raw_input is None:
            limitations.append("exclusive input accounting requires raw input usage")
        else:
            uncached = raw_input
    else:
        limitations.append("input/cache accounting semantics are unknown")

    cache_write: int | None
    if cache_write_accounting == "reported":
        cache_write = reported_cache_write
        if cache_write is None:
            limitations.append("cache-write accounting is reported but usage is missing")
    elif cache_write_accounting == "not_applicable":
        cache_write = 0
    else:
        cache_write = reported_cache_write
        if cache_write is None:
            limitations.append("cache-write accounting semantics are unknown")

    comparable = (
        uncached is not None
        and cache_read is not None
        and cache_write is not None
        and output is not None
        and not limitations
    )
    total = (
        uncached + cache_read + cache_write + output
        if comparable
        else None
    )
    # This is explicitly a lower bound over values the provider actually emitted.
    # For inclusive input accounting, raw input already includes cache reads.
    if input_accounting == "inclusive_cache_read":
        lower_bound = (raw_input or 0) + (reported_cache_write or 0) + (output or 0)
    else:
        lower_bound = (
            (raw_input or 0)
            + (cache_read or 0)
            + (reported_cache_write or 0)
            + (output or 0)
        )

    return NormalizedProviderTraffic(
        input_accounting=input_accounting,
        cache_write_accounting=cache_write_accounting,
        raw_input_tokens=raw_input,
        uncached_input_tokens=uncached,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        output_tokens=output,
        reasoning_tokens=usage.reasoning_tokens,
        total_recorded_traffic=total,
        lower_bound_traffic=lower_bound,
        sol_pi_comparable=comparable,
        limitations=tuple(limitations),
    )
