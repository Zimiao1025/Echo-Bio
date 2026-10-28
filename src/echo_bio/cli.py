"""Command-line interface for the user-facing Echo-Bio runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

MODEL = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"


def _version(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = (completed.stdout or completed.stderr).strip().splitlines()
    return lines[0] if lines else f"exit={completed.returncode}"


def doctor(args: argparse.Namespace) -> int:
    from echo_bio import __version__

    try:
        from biomni.agent.a1 import A1

        biomni_present = True
        seam_present = all(
            hasattr(A1, name)
            for name in ("_before_model_request", "_after_execution", "_on_run_end")
        )
    except (ImportError, ModuleNotFoundError):
        biomni_present = False
        seam_present = False
    checks = {
        "python_supported": sys.version_info >= (3, 11),
        "model_profile_supported": os.environ.get("ECHO_BIO_MODEL", MODEL) == MODEL,
        "base_url_supported": os.environ.get("ECHO_BIO_BASE_URL", BASE_URL).rstrip("/") == BASE_URL,
        "api_key_present": bool(os.environ.get("DEEPSEEK_API_KEY", "").strip()),
        "biomni_installed": biomni_present,
        "biomni_echo_seam_present": seam_present,
        "r_4_4_3_available": (_version(["R", "--version"]) or "").startswith("R version 4.4.3"),
    }
    required = ["python_supported", "model_profile_supported", "base_url_supported"]
    if args.require_runtime:
        required.extend(("biomni_installed", "biomni_echo_seam_present"))
    if args.require_r:
        required.append("r_4_4_3_available")
    status = "pass" if all(checks[key] for key in required) else "fail"
    print(
        json.dumps(
            {
                "status": status,
                "echo_bio_version": __version__,
                "python": platform.python_version(),
                "provider_profile": {"model": MODEL, "base_url": BASE_URL},
                "checks": checks,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if status == "pass" else 1


def provider_check(args: argparse.Namespace) -> int:
    key_present = bool(os.environ.get("DEEPSEEK_API_KEY", "").strip())
    profile = {
        "provider": "deepseek-responses",
        "model": MODEL,
        "base_url": BASE_URL,
        "transport": "provider_native_function_v1",
        "max_requests": 1,
        "max_output_tokens": 128,
        "automatic_retry": False,
        "automatic_transport_fallback": False,
        "api_key_present": key_present,
        "api_key_value_logged": False,
    }
    if not args.live:
        print(json.dumps({"status": "dry_run", "profile": profile}, indent=2, sort_keys=True))
        return 0
    if not args.confirm_paid:
        raise SystemExit("provider-check --live requires --confirm-paid; no request was sent")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set; no request was sent")
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from echo_bio.providers import OpenAIResponsesChatProvider
    except ModuleNotFoundError as error:
        raise SystemExit(
            f"live dependency is missing ({error.name}); install with: python -m pip install -e '.[biomni]'"
        ) from error
    provider = OpenAIResponsesChatProvider(
        api_key=api_key,
        model=MODEL,
        base_url=BASE_URL,
        max_output_tokens=128,
        max_requests=1,
        max_estimated_input_tokens=1024,
        timeout_seconds=90,
        store=None,
        include_metadata=False,
        reasoning_effort="none",
    )
    message = provider.invoke_function(
        [
            SystemMessage(content="This is an ABI canary. Follow the function schema exactly."),
            HumanMessage(content="Return status ok and no scientific content."),
        ],
        function_name="emit_echo_bio_canary",
        function_description="Return the provider ABI canary status.",
        parameters={
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["ok"]}},
            "required": ["status"],
            "additionalProperties": False,
        },
        client_request_id="echo_bio_public_provider_canary",
    )
    if json.loads(str(message.content)) != {"status": "ok"}:
        raise SystemExit("provider returned an invalid canary; no retry was attempted")
    response_id = str(message.response_metadata.get("provider_response_id") or "")
    print(
        json.dumps(
            {
                "status": "pass",
                "request_count": provider.request_count,
                "usage": message.usage_metadata,
                "response_receipt_sha256": hashlib.sha256(response_id.encode()).hexdigest()
                if response_id
                else None,
                "api_key_logged": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run(args: argparse.Namespace) -> int:
    if not args.confirm_paid or not args.allow_local_execution:
        raise SystemExit(
            "run requires both --confirm-paid and --allow-local-execution; no request was sent"
        )
    if bool(args.task) == bool(args.task_file):
        raise SystemExit("provide exactly one of --task or --task-file")
    task = args.task or Path(args.task_file).expanduser().read_text(encoding="utf-8")
    data: list[tuple[Path, str]] = []
    for value in args.data:
        if "=" not in value:
            raise SystemExit("--data must use PATH=DESCRIPTION")
        path, description = value.split("=", 1)
        data.append((Path(path), description))
    from echo_bio.user_runtime import UserRunConfig, run_user_task

    summary = run_user_task(
        UserRunConfig(
            task=task,
            workspace=Path(args.workspace),
            data=tuple(data),
            run_id=args.run_id,
            biomni_data_root=Path(args.biomni_data_root) if args.biomni_data_root else None,
            max_requests=args.max_requests,
            max_output_tokens=args.max_output_tokens,
            max_estimated_input_tokens=args.max_input_tokens,
            timeout_seconds=args.timeout,
            commercial_mode=args.commercial_mode,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="echo-bio", description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    doctor_parser = commands.add_parser("doctor", help="inspect the local environment without API calls")
    doctor_parser.add_argument(
        "--require-runtime",
        action="store_true",
        help="fail unless patched Biomni is installed",
    )
    doctor_parser.add_argument(
        "--require-r",
        action="store_true",
        help="also require exactly R 4.4.3",
    )
    doctor_parser.set_defaults(handler=doctor)
    provider = commands.add_parser("provider-check", help="inspect or test the supported provider profile")
    provider.add_argument("--live", action="store_true")
    provider.add_argument("--confirm-paid", action="store_true")
    provider.set_defaults(handler=provider_check)
    research = commands.add_parser("run", help="run one bounded biomedical research task")
    task = research.add_mutually_exclusive_group(required=True)
    task.add_argument("--task")
    task.add_argument("--task-file")
    research.add_argument("--data", action="append", default=[], metavar="PATH=DESCRIPTION")
    research.add_argument("--workspace", default=".echo-bio/runs")
    research.add_argument("--biomni-data-root")
    research.add_argument("--run-id")
    research.add_argument("--max-requests", type=int, default=8)
    research.add_argument("--max-output-tokens", type=int, default=4096)
    research.add_argument("--max-input-tokens", type=int, default=131072)
    research.add_argument("--timeout", type=int, default=600)
    research.add_argument("--commercial-mode", action="store_true")
    research.add_argument("--confirm-paid", action="store_true")
    research.add_argument("--allow-local-execution", action="store_true")
    research.set_defaults(handler=run)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
