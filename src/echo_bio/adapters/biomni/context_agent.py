"""Biomni adapter with a typed plan tool and evidence-gated context compiler."""

from __future__ import annotations

from typing import Any

from echo_bio.adapters.biomni.artifact_agent import LosslessObservationBiomniAgent
from echo_bio.context.compiler import ContextCompiler
from echo_bio.context.ledger import ContextProjectionLedger
from echo_bio.contracts.plans import PlanTransition, ResearchPlan, ResearchPlanStateMachine

PLAN_TOOL_SCHEMA: dict[str, Any] = {
    "name": "update_research_plan",
    "description": (
        "Advance exactly one typed research-plan step. A completed step must include one or more "
        "completion_evidence records; failed or blocked steps must include notes."
    ),
    "module": "echo_bio.context",
    "required_parameters": ["step_id", "expected_revision", "target_status"],
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "step_id": {"type": "string"},
            "expected_revision": {"type": "integer", "minimum": 0},
            "target_status": {
                "type": "string",
                "enum": ["in_progress", "completed", "failed", "blocked"],
            },
            "completion_evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["evidence_id", "kind", "ref", "description"],
                    "properties": {
                        "evidence_id": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": ["artifact", "execution", "table", "citation", "verifier"],
                        },
                        "ref": {"type": "string"},
                        "sha256": {"type": ["string", "null"]},
                        "description": {"type": "string"},
                    },
                },
            },
            "notes": {"type": ["string", "null"]},
        },
    },
}


class ContextCompiledBiomniAgent(LosslessObservationBiomniAgent):
    def __init__(
        self,
        *args: Any,
        research_plan: ResearchPlan,
        context_compiler: ContextCompiler,
        context_projection_ledger: ContextProjectionLedger | None = None,
        **kwargs: Any,
    ) -> None:
        self.research_plan = research_plan
        self.context_compiler = context_compiler
        self.context_projection_ledger = context_projection_ledger
        self.context_projection_records = []
        self._context_task = research_plan.objective
        self._resource_slice: tuple[str, ...] = ()
        self._plan_machine = ResearchPlanStateMachine()
        super().__init__(*args, **kwargs)
        self._register_plan_tool()

    def _register_plan_tool(self) -> None:
        self._custom_functions = getattr(self, "_custom_functions", {})
        self._custom_functions["update_research_plan"] = self.update_research_plan
        self._custom_tools = getattr(self, "_custom_tools", {})
        self._custom_tools["update_research_plan"] = {
            "name": "update_research_plan",
            "description": PLAN_TOOL_SCHEMA["description"],
            "module": PLAN_TOOL_SCHEMA["module"],
        }
        self.module2api = getattr(self, "module2api", {})
        tools = self.module2api.setdefault("echo_bio.context", [])
        if not any(tool.get("name") == "update_research_plan" for tool in tools):
            tools.append(PLAN_TOOL_SCHEMA.copy())
        registry = getattr(self, "tool_registry", None)
        if registry is not None:
            try:
                registry.register_tool(PLAN_TOOL_SCHEMA.copy())
            except Exception:
                # The explicit module2api and callable registrations remain authoritative.
                pass

    def _on_run_start(self, run_identity: dict[str, str], prompt: str) -> None:
        self._context_task = prompt
        super()._on_run_start(run_identity, prompt)
        self.emit_plan_update(self.research_plan.model_dump(mode="json"), source="initial_typed_plan")

    def _prepare_resources_for_retrieval(self, prompt):
        selected = super()._prepare_resources_for_retrieval(prompt)
        self._resource_slice = tuple(
            f"{category}:{self._resource_name(item)}"
            for category, values in (selected or {}).items()
            if isinstance(values, list)
            for item in values
        )
        return selected

    def _before_model_request(self, messages):
        compiled = self.context_compiler.compile(
            messages,
            task=self._context_task,
            plan=self.research_plan,
            resource_slice=self._resource_slice,
        )
        self.context_projection_records.append(compiled.record)
        if self.context_projection_ledger is not None:
            try:
                self.context_projection_ledger.append(compiled.record)
            except Exception as error:
                self.harness._failure("context_projection_persistence", error, None)
                # A treatment projection without its durable decision record is not delivered.
                return super()._before_model_request(messages)
        return super()._before_model_request(list(compiled.messages))

    def update_research_plan(
        self,
        step_id: str,
        expected_revision: int,
        target_status: str,
        completion_evidence: list[dict[str, Any]] | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Apply a schema-constrained plan transition and emit it to the immutable trace."""
        transition = PlanTransition.model_validate(
            {
                "step_id": step_id,
                "expected_revision": expected_revision,
                "target_status": target_status,
                "completion_evidence": completion_evidence or [],
                "notes": notes,
            }
        )
        self.research_plan = self._plan_machine.apply(self.research_plan, transition)
        payload = self.research_plan.model_dump(mode="json")
        self.emit_plan_update(payload, source="update_research_plan")
        return payload

    @staticmethod
    def _resource_name(item: Any) -> str:
        if isinstance(item, dict):
            return str(item.get("name") or item.get("id") or "unknown")
        return str(item)
