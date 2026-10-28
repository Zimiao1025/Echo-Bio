"""Provider-neutral resource-selection ABI for the Biomni retrieval seam.

The ABI keeps three representations separate:

* the provider request asks for the canonical JSON schema;
* one explicitly frozen acceptance profile decodes the single paid response;
* the validated canonical value is projected to Biomni's four-line format.

The compatibility profile is a deterministic decoder for one response.  It
does not issue a second request, retry a provider call, or choose between two
candidate selections.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import Any, Literal

RESOURCE_CATEGORIES = ("tools", "data_lake", "libraries", "know_how")

RESOURCE_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        category: {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
            "uniqueItems": True,
        }
        for category in RESOURCE_CATEGORIES
    },
    "required": list(RESOURCE_CATEGORIES),
    "additionalProperties": False,
}

ResourceSelectionTransport = Literal[
    "json_schema_v1",
    "biomni_four_line_v1",
    "json_schema_or_biomni_four_line_v1",
    "json_schema_or_biomni_four_line_with_decimal_recovery_v2",
]
ObservedResourceSelectionTransport = Literal[
    "json_schema_v1",
    "biomni_four_line_v1",
]


class ResourceSelectionAbiViolation(ValueError):
    """One provider response cannot be decoded as one safe selection."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CanonicalResourceSelection:
    tools: tuple[int, ...]
    data_lake: tuple[int, ...]
    libraries: tuple[int, ...]
    know_how: tuple[int, ...]
    transport: ObservedResourceSelectionTransport
    normalization: str

    def as_document(self) -> dict[str, list[int]]:
        return {
            "tools": list(self.tools),
            "data_lake": list(self.data_lake),
            "libraries": list(self.libraries),
            "know_how": list(self.know_how),
        }

    def as_json(self) -> str:
        return json.dumps(self.as_document(), separators=(",", ":"), sort_keys=True)

    def as_biomni_lines(self) -> str:
        document = self.as_document()
        return "\n".join(
            (
                f"TOOLS: {json.dumps(document['tools'])}",
                f"DATA_LAKE: {json.dumps(document['data_lake'])}",
                f"LIBRARIES: {json.dumps(document['libraries'])}",
                f"KNOW_HOW: {json.dumps(document['know_how'])}",
            )
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.as_json().encode()).hexdigest()


class ProviderResourceSelectionAdapter:
    """Invoke one provider once and return one canonical resource selection."""

    def __init__(self, provider: Any, transport: ResourceSelectionTransport) -> None:
        self.provider = provider
        self.transport = transport
        self.last_response: Any | None = None
        self.last_raw_content: str | None = None

    def invoke(self, messages: list[Any]) -> tuple[Any, CanonicalResourceSelection]:
        response = self.provider.invoke(
            messages,
            response_schema=RESOURCE_SELECTION_SCHEMA,
            response_schema_name="biomni_resource_selection",
        )
        self.last_response = response
        self.last_raw_content = str(response.content)
        selection = parse_resource_selection(
            self.last_raw_content,
            transport=self.transport,
        )
        return response, selection


def validate_resource_selection_document(
    document: Any,
    *,
    transport: ObservedResourceSelectionTransport,
    normalization: str,
) -> CanonicalResourceSelection:
    """Validate exact keys and unique, non-negative integer indices."""

    if not isinstance(document, dict):
        raise ResourceSelectionAbiViolation("not_object", "resource selection must be an object")
    if set(document) != set(RESOURCE_CATEGORIES):
        raise ResourceSelectionAbiViolation("invalid_keys", "resource selection keys are not exact")
    canonical: dict[str, tuple[int, ...]] = {}
    for category in RESOURCE_CATEGORIES:
        values = document[category]
        if not isinstance(values, list):
            raise ResourceSelectionAbiViolation(f"invalid_{category}_list", f"{category} must be a list")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
            raise ResourceSelectionAbiViolation(
                f"invalid_{category}_index",
                f"{category} indices must be non-negative integers",
            )
        if len(values) != len(set(values)):
            raise ResourceSelectionAbiViolation(f"duplicate_{category}_index", f"{category} indices must be unique")
        canonical[category] = tuple(values)
    return CanonicalResourceSelection(
        tools=canonical["tools"],
        data_lake=canonical["data_lake"],
        libraries=canonical["libraries"],
        know_how=canonical["know_how"],
        transport=transport,
        normalization=normalization,
    )


def parse_json_schema_resource_selection(raw_content: str) -> CanonicalResourceSelection:
    """Parse exactly one JSON object without duplicate keys or repair."""

    def exact_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in pairs:
            if key in document:
                raise ResourceSelectionAbiViolation("duplicate_key", f"duplicate JSON key: {key}")
            document[key] = value
        return document

    def reject_constant(value: str) -> None:
        raise ResourceSelectionAbiViolation("invalid_json_constant", f"invalid JSON constant: {value}")

    try:
        document = json.loads(
            raw_content,
            object_pairs_hook=exact_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ResourceSelectionAbiViolation(
            "invalid_json", "response is not exactly one JSON resource selection"
        ) from error
    return validate_resource_selection_document(
        document,
        transport="json_schema_v1",
        normalization="exact_json_object",
    )


def reconcile_single_premature_object_close(
    raw_content: str,
) -> CanonicalResourceSelection:
    """Recover exactly one unambiguous premature top-level close.

    This is an explicit recovery tool for an already-settled response. It is
    deliberately absent from :func:`parse_resource_selection`, rejects valid
    JSON, and cannot trigger a provider request or transport fallback.
    """

    try:
        parse_json_schema_resource_selection(raw_content)
    except ResourceSelectionAbiViolation as error:
        if error.code != "invalid_json":
            raise
    else:
        raise ResourceSelectionAbiViolation("recovery_not_required", "exact JSON must use the primary parser")
    if raw_content != raw_content.strip() or not raw_content.startswith("{") or not raw_content.endswith("}"):
        raise ResourceSelectionAbiViolation("invalid_recovery_envelope", "recovery requires one bare JSON-like object")
    candidates: list[CanonicalResourceSelection] = []
    for match in re.finditer(
        r'}(?=,\s*"(?:tools|data_lake|libraries|know_how)"\s*:)',
        raw_content,
    ):
        repaired = raw_content[: match.start()] + raw_content[match.end() :]
        try:
            selection = parse_json_schema_resource_selection(repaired)
        except ResourceSelectionAbiViolation:
            continue
        candidates.append(
            replace(
                selection,
                normalization="single_premature_object_close_recovery_v1",
            )
        )
    unique = {candidate.as_json(): candidate for candidate in candidates}
    if not unique:
        raise ResourceSelectionAbiViolation(
            "premature_close_not_recoverable",
            "no single premature top-level close yields the exact schema",
        )
    if len(unique) != 1:
        raise ResourceSelectionAbiViolation(
            "ambiguous_premature_close_recovery",
            "more than one repair yields a valid resource selection",
        )
    return next(iter(unique.values()))


def reconcile_canonical_decimal_string_indices(
    raw_content: str,
) -> CanonicalResourceSelection:
    """Recover one exact JSON selection whose indices are decimal strings.

    This is a zero-provider-call reconciliation profile for a settled response.
    It accepts only the canonical four-key object where every non-empty index is
    an unsigned decimal string. Mixed representations, signs, whitespace,
    leading zeroes, decimals, duplicate keys, and duplicate normalized indices
    fail closed. Valid integer JSON remains on the primary parser.
    """

    try:
        parse_json_schema_resource_selection(raw_content)
    except ResourceSelectionAbiViolation as error:
        if error.code not in {
            "invalid_tools_index",
            "invalid_data_lake_index",
            "invalid_libraries_index",
            "invalid_know_how_index",
        }:
            raise
    else:
        raise ResourceSelectionAbiViolation(
            "recovery_not_required",
            "valid integer JSON must use the primary parser",
        )

    def exact_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in pairs:
            if key in document:
                raise ResourceSelectionAbiViolation(
                    "duplicate_key", f"duplicate JSON key: {key}"
                )
            document[key] = value
        return document

    def reject_constant(value: str) -> None:
        raise ResourceSelectionAbiViolation(
            "invalid_json_constant", f"invalid JSON constant: {value}"
        )

    try:
        document = json.loads(
            raw_content,
            object_pairs_hook=exact_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ResourceSelectionAbiViolation(
            "invalid_json", "response is not exactly one JSON resource selection"
        ) from error
    if not isinstance(document, dict):
        raise ResourceSelectionAbiViolation(
            "not_object", "resource selection must be an object"
        )
    if set(document) != set(RESOURCE_CATEGORIES):
        raise ResourceSelectionAbiViolation(
            "invalid_keys", "resource selection keys are not exact"
        )

    normalized: dict[str, list[int]] = {}
    saw_decimal_string = False
    saw_integer = False
    for category in RESOURCE_CATEGORIES:
        values = document[category]
        if not isinstance(values, list):
            raise ResourceSelectionAbiViolation(
                f"invalid_{category}_list", f"{category} must be a list"
            )
        converted = []
        for value in values:
            if isinstance(value, int) and not isinstance(value, bool):
                saw_integer = True
                converted.append(value)
                continue
            if not isinstance(value, str) or re.fullmatch(
                r"0|[1-9][0-9]*", value
            ) is None:
                raise ResourceSelectionAbiViolation(
                    f"invalid_{category}_decimal_string",
                    f"{category} indices must be canonical unsigned decimal strings",
                )
            saw_decimal_string = True
            converted.append(int(value))
        normalized[category] = converted
    if saw_integer and saw_decimal_string:
        raise ResourceSelectionAbiViolation(
            "mixed_index_representations",
            "integer and decimal-string indices cannot be mixed",
        )
    if saw_integer or not saw_decimal_string:
        raise ResourceSelectionAbiViolation(
            "recovery_not_required",
            "decimal-string recovery requires at least one string and no integers",
        )

    return validate_resource_selection_document(
        normalized,
        transport="json_schema_v1",
        normalization="canonical_decimal_string_indices_v1",
    )


_FOUR_LINE = re.compile(
    r"TOOLS: (?P<tools>\[[^\r\n]*\])\n"
    r"DATA_LAKE: (?P<data_lake>\[[^\r\n]*\])\n"
    r"LIBRARIES: (?P<libraries>\[[^\r\n]*\])\n"
    r"KNOW_HOW: (?P<know_how>\[[^\r\n]*\])"
)


def parse_biomni_four_line_resource_selection(
    raw_content: str,
) -> CanonicalResourceSelection:
    """Parse Biomni's exact four-line representation, with no surrounding text."""

    match = _FOUR_LINE.fullmatch(raw_content)
    if match is None:
        raise ResourceSelectionAbiViolation(
            "invalid_four_line_envelope",
            "response is not exactly the four ordered Biomni resource lines",
        )
    document: dict[str, Any] = {}
    for category in RESOURCE_CATEGORIES:
        try:
            document[category] = json.loads(match.group(category))
        except json.JSONDecodeError as error:
            raise ResourceSelectionAbiViolation(f"invalid_{category}_json", f"{category} is not a JSON list") from error
    return validate_resource_selection_document(
        document,
        transport="biomni_four_line_v1",
        normalization="exact_ordered_four_line_envelope",
    )


def parse_resource_selection(
    raw_content: str,
    *,
    transport: ResourceSelectionTransport,
) -> CanonicalResourceSelection:
    """Decode one response under one preselected acceptance profile."""

    if transport == "json_schema_v1":
        return parse_json_schema_resource_selection(raw_content)
    if transport == "biomni_four_line_v1":
        return parse_biomni_four_line_resource_selection(raw_content)
    if transport == "json_schema_or_biomni_four_line_with_decimal_recovery_v2":
        try:
            return parse_resource_selection(
                raw_content,
                transport="json_schema_or_biomni_four_line_v1",
            )
        except ResourceSelectionAbiViolation as error:
            if error.code not in {
                "invalid_tools_index",
                "invalid_data_lake_index",
                "invalid_libraries_index",
                "invalid_know_how_index",
            }:
                raise
        return reconcile_canonical_decimal_string_indices(raw_content)
    if transport != "json_schema_or_biomni_four_line_v1":
        raise ValueError(f"unsupported resource selection transport: {transport}")

    # The first non-whitespace byte makes the two exact grammars disjoint.  This
    # is deterministic decoding of one response, not a retry or provider call.
    if raw_content.lstrip().startswith("{"):
        return parse_json_schema_resource_selection(raw_content)
    return parse_biomni_four_line_resource_selection(raw_content)
