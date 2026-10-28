"""Deterministic redaction and withholding rules for trace payloads."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from echo_bio.contracts.trace import DataClass, RedactionMetadata

SECRET_PATTERNS = {
    "bearer_token": re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{12,}"),
    "api_key_assignment": re.compile(r"(?i)\b(api[_-]?key|token|secret)\s*[:=]\s*['\"]?[^\s'\"]{8,}"),
    "openai_style_key": re.compile(r"\bsk-[A-Za-z0-9_-]{12,}"),
    "aws_access_key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()


@dataclass(frozen=True)
class RedactionResult:
    value: Any
    metadata: RedactionMetadata
    original_sha256: str


class TraceRedactor:
    """Withhold controlled/secret fields and scrub credential-shaped public text."""

    def sanitize(self, value: Any, *, data_class: DataClass) -> RedactionResult:
        original = canonical_json(value)
        counters = {"fields_withheld": 0, "secret_matches": 0, "characters": 0}
        rules: set[str] = set()

        if data_class in ("controlled", "secret"):
            counters["fields_withheld"] = self._count_leaves(value)
            counters["characters"] = self._count_characters(value)
            rules.add(f"withhold_{data_class}")
            sanitized: Any = {
                "withheld": True,
                "data_class": data_class,
                "original_sha256": hashlib.sha256(original).hexdigest(),
                "original_character_count": counters["characters"],
            }
        else:
            sanitized = self._scrub(value, counters, rules)

        metadata = RedactionMetadata(
            data_class=data_class,
            rules_applied=tuple(sorted(rules)),
            fields_withheld=counters["fields_withheld"],
            secret_matches=counters["secret_matches"],
            original_character_count=self._count_characters(value),
        )
        return RedactionResult(
            value=sanitized,
            metadata=metadata,
            original_sha256=hashlib.sha256(original).hexdigest(),
        )

    def _scrub(self, value: Any, counters: dict[str, int], rules: set[str]) -> Any:
        if isinstance(value, str):
            counters["characters"] += len(value)
            scrubbed = value
            for name, pattern in SECRET_PATTERNS.items():
                scrubbed, count = pattern.subn(f"[REDACTED:{name}]", scrubbed)
                if count:
                    counters["secret_matches"] += count
                    rules.add(name)
            return scrubbed
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if self._sensitive_key(str(key)):
                    counters["fields_withheld"] += 1
                    rules.add("sensitive_field_name")
                    result[key] = "[REDACTED:sensitive_field_name]"
                else:
                    result[key] = self._scrub(item, counters, rules)
            return result
        if isinstance(value, (list, tuple)):
            return [self._scrub(item, counters, rules) for item in value]
        return value

    @staticmethod
    def _sensitive_key(key: str) -> bool:
        normalized = key.lower().replace("-", "_")
        protected_names = {
            "api_key",
            "access_token",
            "auth_token",
            "bearer_token",
            "refresh_token",
            "hf_token",
            "password",
            "secret",
            "credential",
            "credentials",
        }
        return normalized in protected_names or normalized.endswith(
            ("_api_key", "_access_token", "_auth_token", "_password", "_secret", "_credential")
        )

    def _count_leaves(self, value: Any) -> int:
        if isinstance(value, dict):
            return sum(self._count_leaves(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(self._count_leaves(item) for item in value)
        return 1

    def _count_characters(self, value: Any) -> int:
        if isinstance(value, str):
            return len(value)
        if isinstance(value, dict):
            return sum(self._count_characters(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(self._count_characters(item) for item in value)
        return len(str(value)) if value is not None else 0
