"""Local token estimates separated by provider-visible context component."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any

import tiktoken

from echo_bio.contracts.trace import TokenComponent

OBSERVATION_PATTERN = re.compile(r"<observation>(.*?)</observation>", re.DOTALL | re.IGNORECASE)
PLAN_PATTERN = re.compile(r"<(?:plan|research_plan)>(.*?)</(?:plan|research_plan)>", re.DOTALL | re.IGNORECASE)
EVIDENCE_PATTERN = re.compile(r"<(?:evidence|citation)[^>]*>(.*?)</(?:evidence|citation)>", re.DOTALL | re.IGNORECASE)
RESOURCE_MARKERS = ("Environment Resources:", "Function Dictionary:", "Biological data lake", "Software Library:")


def _content(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                part = block.get("text") or block.get("content")
                if isinstance(part, str):
                    parts.append(part)
        return "".join(parts)
    return str(content)


def _role(message: Any) -> str:
    return str(getattr(message, "type", getattr(message, "role", type(message).__name__))).lower()


class TokenEstimator:
    def __init__(self, model: str | None = None) -> None:
        try:
            self.encoding = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text))


class ContextAttributor:
    COMPONENTS = ("system", "tool_resource", "plan", "history", "observation", "evidence")

    def __init__(self, estimator: TokenEstimator | None = None) -> None:
        self.estimator = estimator or TokenEstimator()

    def attribute(self, messages: list[Any] | tuple[Any, ...]) -> tuple[list[TokenComponent], dict[str, int]]:
        components: list[TokenComponent] = []
        totals: defaultdict[str, int] = defaultdict(int)
        for index, message in enumerate(messages):
            role = _role(message)
            content = _content(message)
            for component, text in self._segments(role, content):
                normalized = text.strip()
                if not normalized:
                    continue
                record = TokenComponent(
                    component=component,
                    message_index=index,
                    role=role,
                    token_count=self.estimator.count(normalized),
                    character_count=len(normalized),
                    content_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
                )
                components.append(record)
                totals[component] += record.token_count
        for component in self.COMPONENTS:
            totals.setdefault(component, 0)
        totals["total"] = sum(totals[component] for component in self.COMPONENTS)
        return components, dict(totals)

    def _segments(self, role: str, content: str):
        if role == "system":
            positions = [content.find(marker) for marker in RESOURCE_MARKERS if marker in content]
            if positions:
                boundary = min(position for position in positions if position >= 0)
                return [("system", content[:boundary]), ("tool_resource", content[boundary:])]
            return [("system", content)]
        observation = OBSERVATION_PATTERN.search(content)
        if observation:
            return [("observation", observation.group(1))]
        plan = PLAN_PATTERN.search(content)
        if plan:
            return [("plan", plan.group(1))]
        evidence = EVIDENCE_PATTERN.search(content)
        if evidence:
            return [("evidence", evidence.group(1))]
        return [("history", content)]


def extract_provider_usage(response: Any) -> dict[str, int | str | None]:
    usage = getattr(response, "usage_metadata", None) or {}
    response_metadata = getattr(response, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage", {}) if isinstance(response_metadata, dict) else {}
    details = usage.get("input_token_details", {}) if isinstance(usage, dict) else {}
    output_details = usage.get("output_token_details", {}) if isinstance(usage, dict) else {}

    def first(*values):
        return next((value for value in values if isinstance(value, int) and value >= 0), None)

    input_tokens = first(usage.get("input_tokens"), token_usage.get("prompt_tokens"))
    output_tokens = first(usage.get("output_tokens"), token_usage.get("completion_tokens"))
    cache_read = first(details.get("cache_read"), token_usage.get("cache_read_input_tokens"))
    cache_write = first(details.get("cache_creation"), token_usage.get("cache_creation_input_tokens"))
    reasoning = first(output_details.get("reasoning"), token_usage.get("reasoning_tokens"))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "reasoning_tokens": reasoning,
        "source": "provider" if input_tokens is not None or output_tokens is not None else "unavailable",
    }
