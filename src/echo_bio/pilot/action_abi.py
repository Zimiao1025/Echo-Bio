"""Fail-closed provider-action transports for the Biomni execution seam.

This module deliberately separates three questions:

* did the provider use exactly one permitted transport envelope;
* does the envelope contain exactly one schema-valid action; and
* how is that action projected into Biomni's legacy XML seam.

No parser in this module repairs truncated JSON, concatenates actions, or
chooses between ambiguous candidates.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Literal

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["execute", "solution"]},
        "language": {"type": "string", "enum": ["python", "r", "bash", "none"]},
        "content": {"type": "string", "minLength": 1, "maxLength": 12000},
    },
    "required": ["action", "language", "content"],
    "additionalProperties": False,
}

Transport = Literal[
    "provider_native_function_v1",
    "json_schema_v1",
    "legacy_marker_v1",
]


class ActionAbiViolation(ValueError):
    """The response cannot be interpreted as exactly one safe action."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CanonicalAction:
    action: Literal["execute", "solution"]
    language: Literal["python", "r", "bash", "none"]
    content: str
    transport: Transport
    normalization: str

    def as_document(self) -> dict[str, str]:
        return {
            "action": self.action,
            "language": self.language,
            "content": self.content,
        }

    def as_json(self) -> str:
        return json.dumps(self.as_document(), separators=(",", ":"), sort_keys=True)

    def as_biomni_xml(self) -> str:
        content = self.content.strip()
        if self.action == "solution":
            return f"<solution>{content}</solution>"
        content = re.sub(
            r"^#!(?:BASH|R)\s*\n?", "", content, count=1, flags=re.IGNORECASE
        )
        prefix = {"python": "", "r": "#!R\n", "bash": "#!BASH\n"}[self.language]
        return f"<execute>{prefix}{content}</execute>"


def validate_action_document(
    document: Any,
    *,
    transport: Transport,
    normalization: str,
) -> CanonicalAction:
    """Validate the shared action schema and its cross-field invariants."""

    if not isinstance(document, dict):
        raise ActionAbiViolation("not_object", "action must be a JSON object")
    if set(document) != {"action", "language", "content"}:
        raise ActionAbiViolation("invalid_keys", "action keys are not exact")
    action = document["action"]
    language = document["language"]
    content = document["content"]
    if action not in {"execute", "solution"}:
        raise ActionAbiViolation("invalid_action", "unsupported action")
    if language not in {"python", "r", "bash", "none"}:
        raise ActionAbiViolation("invalid_language", "unsupported language")
    if not isinstance(content, str) or not content.strip():
        raise ActionAbiViolation("empty_content", "action content must be non-empty")
    if len(content) > 12000:
        raise ActionAbiViolation("content_too_long", "action content exceeds 12000 characters")
    if re.search(r"</?(?:execute|solution|observation)\b", content, re.IGNORECASE):
        raise ActionAbiViolation("nested_control_tag", "nested control tag in action content")
    if re.search(
        r"\[(?:BEGIN|END)_(?:EXECUTED_CODE|PRIOR_SOLUTION|TOOL_OBSERVATION)\]",
        content,
        re.IGNORECASE,
    ):
        raise ActionAbiViolation("nested_control_marker", "nested control marker in action content")
    if action == "execute" and language == "none":
        raise ActionAbiViolation("execute_without_language", "execute requires a language")
    if action == "solution" and language != "none":
        raise ActionAbiViolation("solution_with_language", "solution requires language=none")
    return CanonicalAction(
        action=action,
        language=language,
        content=content.strip(),
        transport=transport,
        normalization=normalization,
    )


def parse_json_schema_action(raw_content: str) -> CanonicalAction:
    """Replay the current JSON Schema transport exactly as deployed in H1 v4."""

    normalization = "exact_json_object"
    try:
        document = json.loads(raw_content)
    except json.JSONDecodeError:
        extracted = _single_json_after_bounded_prefix(raw_content)
        if extracted is None:
            raise ActionAbiViolation("invalid_json", "response is not one bounded JSON action") from None
        document, normalization = extracted
    return validate_action_document(
        document,
        transport="json_schema_v1",
        normalization=normalization,
    )


def extract_provider_native_arguments(
    response_metadata: dict[str, Any],
    content: str,
    *,
    function_name: str = "emit_biomni_action",
) -> str:
    """Require one function call, no message output, and byte-preserved arguments."""

    if response_metadata.get("provider_output_kind") != "function_call":
        raise ActionAbiViolation("native_not_function", "provider output is not function-call only")
    calls = response_metadata.get("provider_function_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ActionAbiViolation("native_call_count", "provider must return exactly one function call")
    call = calls[0]
    if not isinstance(call, dict) or call.get("name") != function_name:
        raise ActionAbiViolation("native_function_name", "provider returned the wrong function")
    arguments = call.get("arguments")
    if not isinstance(arguments, str) or arguments != content:
        raise ActionAbiViolation("native_arguments_changed", "function arguments were not preserved")
    if response_metadata.get("provider_message_output_text") not in {None, ""}:
        raise ActionAbiViolation("native_mixed_output", "provider mixed message text with a function call")
    return arguments


def parse_provider_native_action(
    response_metadata: dict[str, Any],
    content: str,
    *,
    function_name: str = "emit_biomni_action",
) -> CanonicalAction:
    arguments = extract_provider_native_arguments(
        response_metadata,
        content,
        function_name=function_name,
    )
    try:
        document = json.loads(arguments)
    except json.JSONDecodeError as error:
        raise ActionAbiViolation("native_invalid_arguments", "function arguments are not JSON") from error
    return validate_action_document(
        document,
        transport="provider_native_function_v1",
        normalization="exact_single_function_call",
    )


_EXECUTED_CODE = re.compile(
    r"(?P<prefix>.*?)\[BEGIN_EXECUTED_CODE\]\s*\n?"
    r"(?P<content>.*?)\n?\s*\[END_EXECUTED_CODE\]\s*",
    re.DOTALL | re.IGNORECASE,
)
_PRIOR_SOLUTION = re.compile(
    r"(?P<prefix>.*?)\[BEGIN_PRIOR_SOLUTION\]\s*\n?"
    r"(?P<content>.*?)\n?\s*\[END_PRIOR_SOLUTION\]\s*",
    re.DOTALL | re.IGNORECASE,
)


def canonicalize_legacy_marker_action(raw_content: str) -> CanonicalAction:
    """Canonicalize one complete legacy marker block, otherwise fail closed.

    Exact or bounded-prefix JSON remains accepted as a compatibility action.
    Marker recovery permits a bounded prose prefix but no suffix, second block,
    nesting, or inferred choice between multiple actions.
    """

    try:
        action = parse_json_schema_action(raw_content)
    except ActionAbiViolation:
        action = None
    if action is not None:
        return CanonicalAction(
            action=action.action,
            language=action.language,
            content=action.content,
            transport="legacy_marker_v1",
            normalization=f"json_compat:{action.normalization}",
        )

    control_markers = re.findall(
        r"\[(?:BEGIN|END)_(?:EXECUTED_CODE|PRIOR_SOLUTION|TOOL_OBSERVATION)\]",
        raw_content,
        re.IGNORECASE,
    )
    matches: list[tuple[re.Match[str], Literal["execute", "solution"]]] = []
    for pattern, action_name in (
        (_EXECUTED_CODE, "execute"),
        (_PRIOR_SOLUTION, "solution"),
    ):
        match = pattern.fullmatch(raw_content)
        if match is not None:
            matches.append((match, action_name))
    if len(matches) != 1 or len(control_markers) != 2:
        raise ActionAbiViolation("legacy_ambiguous", "response is not exactly one closed marker block")
    match, action_name = matches[0]
    prefix = match.group("prefix")
    if len(prefix) > 4000 or re.search(
        r"</?(?:execute|solution|observation)\b|\[(?:BEGIN|END)_",
        prefix,
        re.IGNORECASE,
    ):
        raise ActionAbiViolation("legacy_prefix", "legacy marker prefix is unsafe")
    content = match.group("content").strip()
    if action_name == "solution":
        language = "none"
    elif re.match(r"^#!R(?:\s|$)", content, re.IGNORECASE):
        language = "r"
    elif re.match(r"^#!BASH(?:\s|$)", content, re.IGNORECASE):
        language = "bash"
    else:
        language = "python"
    return validate_action_document(
        {"action": action_name, "language": language, "content": content},
        transport="legacy_marker_v1",
        normalization=(
            "bounded_prefix_plus_executed_code_marker"
            if action_name == "execute" and prefix.strip()
            else "executed_code_marker"
            if action_name == "execute"
            else "bounded_prefix_plus_prior_solution_marker"
            if prefix.strip()
            else "prior_solution_marker"
        ),
    )


def _single_json_after_bounded_prefix(raw_content: str) -> tuple[dict[str, Any], str] | None:
    start = raw_content.find("{")
    if start < 0:
        return None
    prefix = raw_content[:start]
    if len(prefix) > 4000 or re.search(
        r"</?(?:execute|solution|observation)\b|\[(?:BEGIN|END)_",
        prefix,
        re.IGNORECASE,
    ):
        return None
    try:
        document, consumed = json.JSONDecoder().raw_decode(raw_content[start:])
    except json.JSONDecodeError:
        return None
    suffix = raw_content[start + consumed :].strip()
    dsml = re.fullmatch(
        r"</｜｜DSML｜｜ parameter>\s*</｜｜DSML｜｜ invoke>\s*</｜｜DSML｜｜ calls>",
        suffix,
    )
    if suffix and dsml is None:
        return None
    if not isinstance(document, dict):
        return None
    if dsml is not None:
        name = "json_plus_deepseek_dsml_closing_suffix"
        if prefix.strip():
            name = "bounded_prefix_plus_" + name
    else:
        name = "single_json_object_after_bounded_prefix"
    return document, name


class ProviderActionAdapter:
    """Invoke one selected transport and return one Biomni-compatible action."""

    def __init__(self, provider: Any, transport: Transport) -> None:
        self.provider = provider
        self.transport = transport
        self.last_raw_content: str | None = None
        self.last_response: Any | None = None

    def invoke(self, messages: list[Any]) -> tuple[Any, CanonicalAction]:
        if self.transport == "provider_native_function_v1":
            response = self.provider.invoke_function(
                messages,
                function_name="emit_biomni_action",
                function_description="Emit exactly one Biomni execute or solution action.",
                parameters=ACTION_SCHEMA,
            )
            self.last_response = response
            self.last_raw_content = str(response.content)
            action = parse_provider_native_action(
                dict(response.response_metadata or {}), self.last_raw_content
            )
        elif self.transport == "json_schema_v1":
            response = self.provider.invoke(
                messages,
                response_schema=ACTION_SCHEMA,
                response_schema_name="biomni_agent_action",
            )
            self.last_response = response
            self.last_raw_content = str(response.content)
            action = parse_json_schema_action(self.last_raw_content)
        else:
            response = self.provider.invoke(messages)
            self.last_response = response
            self.last_raw_content = str(response.content)
            action = canonicalize_legacy_marker_action(self.last_raw_content)
        metadata = dict(response.response_metadata or {})
        metadata["echo_action_abi"] = {
            "transport": self.transport,
            "normalization": action.normalization,
            "canonical_action": action.as_document(),
            "raw_content_sha256": hashlib.sha256(
                self.last_raw_content.encode()
            ).hexdigest(),
        }
        return response.model_copy(
            update={"content": action.as_biomni_xml(), "response_metadata": metadata}
        ), action
