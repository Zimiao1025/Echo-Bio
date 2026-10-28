"""Small synchronous Responses API adapter with a fail-closed call budget.

The Phase 1 pilot deliberately does not retry model submissions.  The supplied
relay is synchronous and cannot reconcile a response whose initial POST was
accepted but whose response was lost; resubmission could therefore double bill.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any, Protocol
from uuid import uuid4

from langchain_core.messages import AIMessage

from echo_bio.telemetry.token_attribution import TokenEstimator


class JsonHttpTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
    ) -> tuple[int, bytes]: ...


class UrllibJsonTransport:
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
    ) -> tuple[int, bytes]:
        request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()


class ProviderRequestRejected(RuntimeError):
    """The provider returned a complete error response, so no retry is attempted."""

    def __init__(
        self,
        message: str,
        *,
        accepted_response: bool = False,
        response_document: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.accepted_response = accepted_response
        self.response_document = response_document


class ProviderSubmissionOutcomeUnknown(RuntimeError):
    """The request may have been accepted; automatic resubmission is forbidden."""


class ProviderBudgetExceeded(RuntimeError):
    """The preregistered maximum number of paid requests was reached."""


class ProviderInputBudgetExceeded(RuntimeError):
    """The local request estimate exceeds the frozen per-request input cap."""


class OpenAIResponsesChatProvider:
    """Expose a Responses API endpoint through Biomni's minimal ``invoke`` seam."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        max_output_tokens: int,
        max_requests: int,
        max_estimated_input_tokens: int | None = None,
        timeout_seconds: float = 90.0,
        allow_plain_http: bool = False,
        store: bool | None = True,
        include_metadata: bool = True,
        reasoning_effort: str | None = None,
        transport: JsonHttpTransport | None = None,
    ) -> None:
        normalized = base_url.strip().rstrip("/")
        if not api_key.strip() or not model.strip():
            raise ValueError("provider API key and model are required")
        if not normalized.startswith(("https://", "http://")):
            raise ValueError("provider base URL must be an absolute HTTP(S) URL")
        if normalized.startswith("http://") and not allow_plain_http:
            raise ValueError("plain HTTP requires explicit allow_plain_http=True")
        if max_output_tokens <= 0 or max_requests <= 0 or timeout_seconds <= 0:
            raise ValueError("provider budgets and timeout must be positive")
        if max_estimated_input_tokens is not None and max_estimated_input_tokens <= 0:
            raise ValueError("max_estimated_input_tokens must be positive when configured")
        if reasoning_effort not in {None, "none", "low", "high", "max"}:
            raise ValueError("reasoning_effort must be none, low, high, max, or omitted")
        self.api_key = api_key
        self.model_name = model
        self.base_url = normalized
        self.max_output_tokens = max_output_tokens
        self.max_requests = max_requests
        self.max_estimated_input_tokens = max_estimated_input_tokens
        self.timeout_seconds = timeout_seconds
        self.store = store
        self.include_metadata = include_metadata
        self.reasoning_effort = reasoning_effort
        self.transport = transport or UrllibJsonTransport()
        self.request_count = 0
        self.response_ids: list[str] = []
        self.token_estimator = TokenEstimator(model)

    def invoke(
        self,
        messages: Sequence[Any],
        *,
        response_schema: Mapping[str, Any] | None = None,
        response_schema_name: str = "echo_structured_response",
        client_request_id: str | None = None,
    ) -> AIMessage:
        if response_schema is not None and (
            not response_schema_name or not response_schema_name.replace("_", "").isalnum()
        ):
            raise ValueError("response_schema_name must contain only letters, numbers, and underscores")
        extra_payload: dict[str, Any] = {}
        if response_schema is not None:
            extra_payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": response_schema_name,
                    "schema": dict(response_schema),
                    "strict": True,
                }
            }
        document, request_id = self._submit(
            messages,
            extra_payload=extra_payload,
            client_request_id=client_request_id,
        )
        return self._with_client_request_id(self._parse(document), request_id)

    def invoke_function(
        self,
        messages: Sequence[Any],
        *,
        function_name: str,
        function_description: str,
        parameters: Mapping[str, Any],
        client_request_id: str | None = None,
    ) -> AIMessage:
        """Force one provider-native Responses function-call output."""

        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", function_name):
            raise ValueError("function_name must match [A-Za-z0-9_-]{1,128}")
        document, request_id = self._submit(
            messages,
            extra_payload={
                "tools": [
                    {
                        "type": "function",
                        "name": function_name,
                        "description": function_description,
                        "parameters": dict(parameters),
                        "strict": True,
                    }
                ],
                "tool_choice": {"type": "function", "name": function_name},
            },
            client_request_id=client_request_id,
        )
        return self._with_client_request_id(self._parse_function(document), request_id)

    def _submit(
        self,
        messages: Sequence[Any],
        *,
        extra_payload: Mapping[str, Any],
        client_request_id: str | None,
    ) -> tuple[dict[str, Any], str]:
        serialized_messages = [self._message(message) for message in messages]
        estimated_input_tokens = sum(self.token_estimator.count(message["content"]) for message in serialized_messages)
        if self.max_estimated_input_tokens is not None and estimated_input_tokens > self.max_estimated_input_tokens:
            raise ProviderInputBudgetExceeded(
                "estimated provider input exceeds frozen cap "
                f"({estimated_input_tokens}/{self.max_estimated_input_tokens})"
            )
        if self.request_count >= self.max_requests:
            raise ProviderBudgetExceeded(
                f"provider request budget exhausted ({self.request_count}/{self.max_requests})"
            )
        request_id = client_request_id or f"echo_{uuid4().hex}"
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", request_id):
            raise ValueError("client_request_id contains unsupported characters")
        self.request_count += 1
        payload = {
            "model": self.model_name,
            "input": serialized_messages,
            "max_output_tokens": self.max_output_tokens,
        }
        payload.update(dict(extra_payload))
        if self.store is not None:
            payload["store"] = self.store
        if self.include_metadata:
            payload["metadata"] = {
                "echo_bio_purpose": "phase1-observe-only-pilot",
                "echo_bio_client_request_id": request_id,
            }
        if self.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        try:
            status, raw = self.transport.request(
                "POST",
                f"{self.base_url}/responses",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "X-Client-Request-Id": request_id,
                },
                body=json.dumps(payload, separators=(",", ":"), sort_keys=True).encode(),
                timeout_seconds=self.timeout_seconds,
            )
        except (TimeoutError, urllib.error.URLError, ConnectionError, OSError) as error:
            raise ProviderSubmissionOutcomeUnknown(
                f"provider submission outcome is unknown ({type(error).__name__}); do not retry automatically"
            ) from error
        document = self._json_document(raw, status)
        if status < 200 or status >= 300:
            error_value = document.get("error")
            message = error_value.get("message") if isinstance(error_value, dict) else None
            raise ProviderRequestRejected(f"provider HTTP {status}: {message or 'request rejected'}")
        return document, request_id

    @staticmethod
    def _message(message: Any) -> dict[str, str]:
        role = str(getattr(message, "type", getattr(message, "role", "user"))).lower()
        role = {"human": "user", "ai": "assistant"}.get(role, role)
        if role not in {"system", "user", "assistant", "developer"}:
            role = "user"
        content = getattr(message, "content", message)
        if isinstance(content, list):
            pieces = []
            for block in content:
                if isinstance(block, dict):
                    value = block.get("text") or block.get("content")
                    if isinstance(value, str):
                        pieces.append(value)
            content = "".join(pieces)
        return {"role": role, "content": str(content)}

    @staticmethod
    def _json_document(raw: bytes, status: int) -> dict[str, Any]:
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            if 200 <= status < 300:
                raise ProviderSubmissionOutcomeUnknown(
                    "provider returned an unreadable success response; do not retry automatically"
                ) from error
            raise ProviderRequestRejected(f"provider returned non-JSON HTTP {status}") from error
        if not isinstance(value, dict):
            raise ProviderRequestRejected("provider returned a non-object response")
        return value

    def _parse(self, document: dict[str, Any]) -> AIMessage:
        try:
            content = self._output_text(document)
        except ProviderRequestRejected as error:
            raise ProviderRequestRejected(
                str(error), accepted_response=True, response_document=document
            ) from error
        usage_metadata, response_metadata = self._response_metadata(document)
        return AIMessage(
            content=content,
            usage_metadata=usage_metadata,
            response_metadata=response_metadata,
        )

    def _parse_function(self, document: dict[str, Any]) -> AIMessage:
        usage_metadata, response_metadata = self._response_metadata(document)
        calls: list[dict[str, str]] = []
        message_pieces: list[str] = []
        output = document.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "function_call":
                    calls.append(
                        {
                            "call_id": str(item.get("call_id") or ""),
                            "name": str(item.get("name") or ""),
                            "arguments": str(item.get("arguments") or ""),
                            "status": str(item.get("status") or ""),
                        }
                    )
                elif item.get("type") == "message":
                    content = item.get("content")
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") in {
                                "output_text",
                                "text",
                            }:
                                value = block.get("text")
                                if isinstance(value, str):
                                    message_pieces.append(value)
        message_text = "".join(message_pieces)
        response_metadata.update(
            {
                "provider_output_kind": (
                    "function_call" if calls and not message_text else "mixed" if calls and message_text else "message"
                ),
                "provider_function_calls": calls,
                "provider_message_output_text": message_text,
            }
        )
        return AIMessage(
            content=calls[0]["arguments"] if len(calls) == 1 else message_text,
            usage_metadata=usage_metadata,
            response_metadata=response_metadata,
        )

    def _response_metadata(self, document: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        response_id = str(document.get("id") or "")
        if not response_id:
            raise ProviderSubmissionOutcomeUnknown(
                "provider response has no durable response id; do not retry automatically"
            )
        status = str(document.get("status") or "")
        if status not in {"completed", "incomplete"}:
            error_value = document.get("error")
            message = error_value.get("message") if isinstance(error_value, dict) else None
            raise ProviderRequestRejected(
                f"provider response ended as {status or 'unknown'}: {message or ''}".strip(),
                accepted_response=True,
                response_document=document,
            )
        usage = document.get("usage") if isinstance(document.get("usage"), dict) else {}
        input_tokens = self._non_negative_int(usage.get("input_tokens"))
        output_tokens = self._non_negative_int(usage.get("output_tokens"))
        reasoning_tokens = None
        cached_tokens = None
        input_details = usage.get("input_tokens_details")
        if isinstance(input_details, dict):
            cached_tokens = self._optional_non_negative_int(input_details.get("cached_tokens"))
        details = usage.get("output_tokens_details")
        if isinstance(details, dict):
            reasoning_tokens = self._optional_non_negative_int(details.get("reasoning_tokens"))
        self.response_ids.append(response_id)
        incomplete = document.get("incomplete_details")
        incomplete_reason = incomplete.get("reason") if isinstance(incomplete, dict) else None
        usage_metadata: dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        if reasoning_tokens is not None:
            usage_metadata["output_token_details"] = {"reasoning": reasoning_tokens}
        if cached_tokens is not None:
            usage_metadata["input_token_details"] = {"cache_read": cached_tokens}
        return (
            usage_metadata,
            {
                "provider_response_id": response_id,
                "model_name": str(document.get("model") or self.model_name),
                "finish_reason": str(incomplete_reason or "completed"),
                "provider_status": status,
            },
        )

    @staticmethod
    def _with_client_request_id(response: AIMessage, request_id: str) -> AIMessage:
        metadata = dict(response.response_metadata)
        metadata["client_request_id"] = request_id
        return response.model_copy(update={"response_metadata": metadata})

    @staticmethod
    def _non_negative_int(value: Any) -> int:
        return int(value) if isinstance(value, int) and value >= 0 else 0

    @staticmethod
    def _optional_non_negative_int(value: Any) -> int | None:
        return int(value) if isinstance(value, int) and value >= 0 else None

    @staticmethod
    def _output_text(document: dict[str, Any]) -> str:
        direct = document.get("output_text")
        if isinstance(direct, str) and direct:
            return direct
        pieces: list[str] = []
        output = document.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if isinstance(block, dict) and block.get("type") in {"output_text", "text"}:
                        value = block.get("text")
                        if isinstance(value, str):
                            pieces.append(value)
                    elif isinstance(block, dict) and block.get("type") == "refusal":
                        raise ProviderRequestRejected("provider refused the request")
        if not pieces:
            raise ProviderRequestRejected("completed provider response contains no output text")
        return "".join(pieces)
