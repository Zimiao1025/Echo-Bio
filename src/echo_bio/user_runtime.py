"""Bounded, auditable user runtime for one Echo-Bio research task.

This module is intentionally independent from the experiment runners.  It exposes
the provider-native Biomni action ABI, the append-only trace/evidence stores and a
small result contract without importing any corpus, rubric answer or campaign
configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from echo_bio import __version__

SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class UserRunConfig:
    task: str
    workspace: Path
    data: tuple[tuple[Path, str], ...] = ()
    run_id: str | None = None
    biomni_data_root: Path | None = None
    max_requests: int = 8
    max_output_tokens: int = 4096
    max_estimated_input_tokens: int = 131_072
    timeout_seconds: int = 600
    commercial_mode: bool = False


class BiomniNativeActionProvider:
    """Project one strict provider-native function call into Biomni's XML seam."""

    def __init__(self, provider: Any) -> None:
        from echo_bio.pilot.action_abi import ProviderActionAdapter

        self._provider = provider
        self._adapter = ProviderActionAdapter(provider, "provider_native_function_v1")

    def invoke(self, messages: list[Any]) -> Any:
        response, _action = self._adapter.invoke(messages)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)


def run_user_task(config: UserRunConfig) -> dict[str, Any]:
    """Run one paid task without retry/fallback and persist an auditable result."""

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set; no provider request was sent")
    if not config.task.strip():
        raise ValueError("task must not be empty")
    if config.max_requests <= 0 or config.max_output_tokens <= 0:
        raise ValueError("request and output budgets must be positive")

    run_id = config.run_id or f"run-{uuid4().hex}"
    if not SAFE_ID.fullmatch(run_id):
        raise ValueError("run_id may contain only letters, numbers, dot, underscore and hyphen")

    data = _validate_data(config.data)
    run_root = config.workspace.expanduser().resolve() / run_id
    run_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    biomni_root = (
        config.biomni_data_root.expanduser().resolve()
        if config.biomni_data_root
        else config.workspace.expanduser().resolve() / "biomni-runtime"
    )
    biomni_root.mkdir(parents=True, exist_ok=True, mode=0o700)

    environment_digest = _sha_json(
        {
            "echo_bio": __version__,
            "python": sys.version,
            "platform": platform.platform(),
        }
    )
    data_manifest = [
        {
            "path": str(path),
            "description": description,
            "size_bytes": path.stat().st_size,
            "sha256": _sha_file(path),
        }
        for path, description in data
    ]
    data_manifest_digest = _sha_json(data_manifest)

    # Optional live dependencies are imported only after local validation.
    try:
        from echo_bio.adapters.biomni.evidence_agent import EvidenceBiomniAgent
        from echo_bio.artifacts.store import LocalArtifactStore
        from echo_bio.evidence.ledger import EvidenceLedger
        from echo_bio.providers import OpenAIResponsesChatProvider
        from echo_bio.telemetry.hooks import TelemetryConfig
        from echo_bio.telemetry.ledger import TraceLedger
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"live runtime dependency is missing ({error.name}); install Biomni and echo-bio[biomni]"
        ) from error

    provider = OpenAIResponsesChatProvider(
        api_key=api_key,
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        max_output_tokens=config.max_output_tokens,
        max_requests=config.max_requests,
        max_estimated_input_tokens=config.max_estimated_input_tokens,
        timeout_seconds=float(config.timeout_seconds),
        store=None,
        include_metadata=False,
        reasoning_effort="none",
    )
    trace = TraceLedger(run_root / "trace")
    artifacts = LocalArtifactStore(run_root / "artifacts")
    evidence = EvidenceLedger(run_root / "evidence")
    telemetry = TelemetryConfig(
        task_id=run_id,
        split="pilot",
        harness_candidate_id="echo-bio-user-runtime-v1",
        model_snapshot="deepseek-flash",
        provider="deepseek-responses",
        environment_digest=environment_digest,
        data_manifest_digest=data_manifest_digest,
        data_class="internal",
        capture_content=False,
    )
    agent = EvidenceBiomniAgent(
        path=str(biomni_root),
        llm="deepseek-flash",
        source="Custom",
        use_tool_retriever=False,
        timeout_seconds=config.timeout_seconds,
        base_url="http://provider.invalid/v1",
        api_key="configured-by-echo-bio",
        commercial_mode=config.commercial_mode,
        expected_data_lake_files=[],
        trace_ledger=trace,
        telemetry_config=telemetry,
        artifact_store=artifacts,
        evidence_ledger=evidence,
    )
    if data:
        agent.add_data({str(path): description for path, description in data})
    agent.llm = BiomniNativeActionProvider(provider)

    task = config.task.strip()
    if data:
        task += "\n\nUser-provided data files:\n" + "\n".join(
            f"- {path}: {description}" for path, description in data
        )
    summary_path = run_root / "summary.json"
    try:
        _log, answer = agent.go(
            task,
            run_id=run_id,
            attempt_id=f"attempt-{uuid4().hex}",
            session_id=f"session-{uuid4().hex}",
        )
        answer_text = str(answer or "")
        answer_path = run_root / "answer.md"
        answer_path.write_text(answer_text + ("\n" if answer_text else ""), encoding="utf-8")
        summary = {
            "schema_version": "echo.user-run-summary.v1",
            "status": "completed",
            "run_id": run_id,
            "provider": "deepseek-responses",
            "model": "deepseek-flash",
            "provider_requests": provider.request_count,
            "automatic_retry": False,
            "automatic_transport_fallback": False,
            "environment_digest": environment_digest,
            "data_manifest_digest": data_manifest_digest,
            "answer": str(answer_path),
            "trace": str(run_root / "trace" / run_id),
            "evidence": str(run_root / "evidence" / run_id / "evidence.jsonl"),
            "artifacts": str(run_root / "artifacts"),
        }
    except Exception as error:
        summary = {
            "schema_version": "echo.user-run-summary.v1",
            "status": "failed_closed",
            "run_id": run_id,
            "provider": "deepseek-responses",
            "model": "deepseek-flash",
            "provider_requests": provider.request_count,
            "automatic_retry": False,
            "automatic_transport_fallback": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def config_as_json(config: UserRunConfig) -> dict[str, Any]:
    value = asdict(config)
    value["workspace"] = str(config.workspace)
    value["biomni_data_root"] = str(config.biomni_data_root) if config.biomni_data_root else None
    value["data"] = [(str(path), description) for path, description in config.data]
    return value


def _validate_data(entries: tuple[tuple[Path, str], ...]) -> tuple[tuple[Path, str], ...]:
    validated = []
    for raw_path, description in entries:
        path = raw_path.expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"data path is not a readable file: {path}")
        if not description.strip():
            raise ValueError(f"data description must not be empty: {path}")
        validated.append((path, description.strip()))
    return tuple(validated)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
