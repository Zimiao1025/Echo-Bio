"""Role-aware prompt component attribution for residual harness research."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from echo_bio.telemetry.token_attribution import ContextAttributor, TokenComponent

PromptComponentName = Literal[
    "system_policy",
    "resource_context",
    "task",
    "plan",
    "action_history",
    "observation",
    "evidence",
    "other_history",
]


class PromptAttributionContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PromptTokenComponentV2(PromptAttributionContract):
    schema_version: Literal["echo.prompt-token-component.v2"] = (
        "echo.prompt-token-component.v2"
    )
    component: PromptComponentName
    message_index: int = Field(ge=0)
    role: str
    token_count: int = Field(ge=0)
    character_count: int = Field(ge=0)
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class PromptContextAttributorV2:
    """Refine the v1 six-way attribution using message roles.

    The v1 attributor remains the byte/token authority.  This class only refines
    its broad ``history`` bucket, so adding v2 telemetry cannot change provider-
    visible messages or token estimates.
    """

    COMPONENTS = (
        "system_policy",
        "resource_context",
        "task",
        "plan",
        "action_history",
        "observation",
        "evidence",
        "other_history",
    )

    def __init__(self, base: ContextAttributor | None = None) -> None:
        self.base = base or ContextAttributor()

    def attribute(
        self, messages: list[Any] | tuple[Any, ...]
    ) -> tuple[list[PromptTokenComponentV2], dict[str, int]]:
        components, _ = self.base.attribute(messages)
        return self.upgrade(components)

    def upgrade(
        self, components: list[TokenComponent] | tuple[TokenComponent, ...]
    ) -> tuple[list[PromptTokenComponentV2], dict[str, int]]:
        first_human_index = min(
            (
                component.message_index
                for component in components
                if _role_family(component.role) == "human"
            ),
            default=None,
        )
        upgraded: list[PromptTokenComponentV2] = []
        totals: defaultdict[str, int] = defaultdict(int)
        for component in components:
            name = self._component_name(component, first_human_index)
            row = PromptTokenComponentV2(
                component=name,
                message_index=component.message_index,
                role=component.role,
                token_count=component.token_count,
                character_count=component.character_count,
                content_sha256=component.content_sha256,
            )
            upgraded.append(row)
            totals[name] += row.token_count
        for name in self.COMPONENTS:
            totals.setdefault(name, 0)
        totals["total"] = sum(totals[name] for name in self.COMPONENTS)
        return upgraded, dict(totals)

    @staticmethod
    def _component_name(
        component: TokenComponent, first_human_index: int | None
    ) -> PromptComponentName:
        direct: dict[str, PromptComponentName] = {
            "system": "system_policy",
            "tool_resource": "resource_context",
            "plan": "plan",
            "observation": "observation",
            "evidence": "evidence",
        }
        if component.component in direct:
            return direct[component.component]
        family = _role_family(component.role)
        if family == "tool":
            return "observation"
        if family == "assistant":
            return "action_history"
        if family == "human":
            # Biomni's ReAct adapter uses the first HumanMessage for the task and
            # appends every execution observation as a later HumanMessage.  The
            # message index therefore carries the boundary that the generic v1
            # role label lost.  This rule is part of the Biomni telemetry ABI;
            # chat-style multi-user sessions require a different classifier.
            if component.message_index == first_human_index:
                return "task"
            return "observation"
        return "other_history"


def _role_family(role: str) -> Literal["system", "human", "assistant", "tool", "other"]:
    normalized = role.lower()
    if "system" in normalized:
        return "system"
    if normalized in {"human", "humanmessage", "user"} or "human" in normalized:
        return "human"
    if normalized in {"ai", "aimessage", "assistant"} or normalized.startswith("ai"):
        return "assistant"
    if normalized in {"tool", "toolmessage", "function"} or "tool" in normalized:
        return "tool"
    return "other"
