from __future__ import annotations

import json

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from echo_bio.providers import (
    OpenAIResponsesChatProvider,
    ProviderBudgetExceeded,
    ProviderInputBudgetExceeded,
    ProviderSubmissionOutcomeUnknown,
)


class RecordingTransport:
    def __init__(self, response: dict | None = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.requests = []

    def request(self, method, url, *, headers, body, timeout_seconds):
        self.requests.append((method, url, headers, json.loads(body), timeout_seconds))
        if self.error:
            raise self.error
        return 200, json.dumps(self.response).encode()


def completed_response():
    return {
        "id": "resp_test_123",
        "status": "completed",
        "model": "gpt-5.6-terra-2026-09-01",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "<solution>bounded answer</solution>"}],
            }
        ],
        "usage": {
            "input_tokens": 120,
            "output_tokens": 30,
            "output_tokens_details": {"reasoning_tokens": 12},
        },
    }


def completed_function_response(arguments: str):
    document = completed_response()
    document["output"] = [
        {
            "type": "function_call",
            "call_id": "call_test_123",
            "name": "emit_biomni_action",
            "arguments": arguments,
            "status": "completed",
        }
    ]
    return document


def test_provider_maps_messages_usage_and_receipt_without_leaking_key():
    transport = RecordingTransport(completed_response())
    provider = OpenAIResponsesChatProvider(
        api_key="test-secret-value",
        model="gpt-5.6-terra",
        base_url="http://relay.example/v1",
        max_output_tokens=768,
        max_requests=1,
        allow_plain_http=True,
        transport=transport,
    )

    response = provider.invoke([SystemMessage(content="system"), HumanMessage(content="question")])

    assert response.content == "<solution>bounded answer</solution>"
    assert response.usage_metadata["input_tokens"] == 120
    assert response.usage_metadata["output_token_details"]["reasoning"] == 12
    assert response.response_metadata["provider_response_id"] == "resp_test_123"
    _, url, headers, payload, _ = transport.requests[0]
    assert url == "http://relay.example/v1/responses"
    assert headers["Authorization"] == "Bearer test-secret-value"
    assert "test-secret-value" not in json.dumps(payload)
    assert payload["max_output_tokens"] == 768
    assert "background" not in payload
    with pytest.raises(ProviderBudgetExceeded):
        provider.invoke([HumanMessage(content="second call")])


def test_plain_http_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="explicit"):
        OpenAIResponsesChatProvider(
            api_key="key",
            model="model",
            base_url="http://relay.example/v1",
            max_output_tokens=10,
            max_requests=1,
        )


def test_transport_timeout_is_unknown_and_never_retried():
    transport = RecordingTransport(error=TimeoutError("lost"))
    provider = OpenAIResponsesChatProvider(
        api_key="key",
        model="model",
        base_url="https://provider.example/v1",
        max_output_tokens=10,
        max_requests=2,
        transport=transport,
    )
    with pytest.raises(ProviderSubmissionOutcomeUnknown, match="do not retry"):
        provider.invoke([HumanMessage(content="question")])
    assert provider.request_count == 1
    assert len(transport.requests) == 1


def test_input_budget_rejects_locally_before_counting_or_transport():
    transport = RecordingTransport(completed_response())
    provider = OpenAIResponsesChatProvider(
        api_key="key",
        model="model",
        base_url="https://provider.example/v1",
        max_output_tokens=10,
        max_requests=2,
        max_estimated_input_tokens=1,
        transport=transport,
    )
    with pytest.raises(ProviderInputBudgetExceeded, match="frozen cap"):
        provider.invoke([HumanMessage(content="this request is longer than one token")])
    assert provider.request_count == 0
    assert transport.requests == []


def test_deepseek_profile_omits_unsupported_fields_and_records_cache_usage():
    document = completed_response()
    document["usage"]["input_tokens_details"] = {"cached_tokens": 100}
    transport = RecordingTransport(document)
    provider = OpenAIResponsesChatProvider(
        api_key="key",
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        max_output_tokens=2048,
        max_requests=1,
        store=None,
        include_metadata=False,
        reasoning_effort="low",
        transport=transport,
    )

    response = provider.invoke([HumanMessage(content="question")])

    payload = transport.requests[0][3]
    assert "store" not in payload
    assert "metadata" not in payload
    assert payload["reasoning"] == {"effort": "low"}
    assert response.usage_metadata["input_token_details"]["cache_read"] == 100


def test_structured_response_schema_is_sent_through_responses_text_format():
    transport = RecordingTransport(completed_response())
    provider = OpenAIResponsesChatProvider(
        api_key="key",
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        max_output_tokens=2048,
        max_requests=1,
        store=None,
        include_metadata=False,
        transport=transport,
    )
    schema = {
        "type": "object",
        "properties": {"tools": {"type": "array", "items": {"type": "integer"}}},
        "required": ["tools"],
        "additionalProperties": False,
    }

    provider.invoke(
        [HumanMessage(content="select tools")],
        response_schema=schema,
        response_schema_name="biomni_resource_selection",
    )

    assert transport.requests[0][3]["text"] == {
        "format": {
            "type": "json_schema",
            "name": "biomni_resource_selection",
            "schema": schema,
            "strict": True,
        }
    }


def test_native_function_payload_omits_reasoning_when_profile_uses_provider_default():
    arguments = '{"action":"solution","language":"none","content":"done"}'
    transport = RecordingTransport(completed_function_response(arguments))
    provider = OpenAIResponsesChatProvider(
        api_key="key",
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        max_output_tokens=4096,
        max_requests=1,
        store=None,
        include_metadata=False,
        reasoning_effort=None,
        transport=transport,
    )

    response = provider.invoke_function(
        [HumanMessage(content="answer")],
        function_name="emit_biomni_action",
        function_description="Emit one action.",
        parameters={"type": "object"},
    )

    payload = transport.requests[0][3]
    assert "reasoning" not in payload
    assert payload["tool_choice"] == {
        "type": "function",
        "name": "emit_biomni_action",
    }
    assert len(payload["tools"]) == 1
    assert response.content == arguments
    assert response.response_metadata["provider_output_kind"] == "function_call"


def test_native_function_payload_explicitly_disables_deepseek_thinking():
    arguments = '{"action":"solution","language":"none","content":"done"}'
    transport = RecordingTransport(completed_function_response(arguments))
    provider = OpenAIResponsesChatProvider(
        api_key="key",
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        max_output_tokens=4096,
        max_requests=1,
        store=None,
        include_metadata=False,
        reasoning_effort="none",
        transport=transport,
    )

    provider.invoke_function(
        [HumanMessage(content="answer")],
        function_name="emit_biomni_action",
        function_description="Emit one action.",
        parameters={"type": "object"},
    )

    payload = transport.requests[0][3]
    assert payload["reasoning"] == {"effort": "none"}
    assert payload["tool_choice"] == {
        "type": "function",
        "name": "emit_biomni_action",
    }


def test_invalid_structured_response_schema_name_fails_before_transport():
    transport = RecordingTransport(completed_response())
    provider = OpenAIResponsesChatProvider(
        api_key="key",
        model="model",
        base_url="https://provider.example/v1",
        max_output_tokens=10,
        max_requests=1,
        transport=transport,
    )

    with pytest.raises(ValueError, match="response_schema_name"):
        provider.invoke(
            [HumanMessage(content="question")],
            response_schema={"type": "object"},
            response_schema_name="bad-name",
        )
    assert transport.requests == []


def test_incomplete_max_output_response_preserves_partial_content_usage_and_receipt():
    document = completed_response()
    document["status"] = "incomplete"
    document["incomplete_details"] = {"reason": "max_output_tokens"}
    transport = RecordingTransport(document)
    provider = OpenAIResponsesChatProvider(
        api_key="key",
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        max_output_tokens=10,
        max_requests=1,
        store=None,
        include_metadata=False,
        transport=transport,
    )

    response = provider.invoke([HumanMessage(content="question")])

    assert response.content == "<solution>bounded answer</solution>"
    assert response.response_metadata["provider_status"] == "incomplete"
    assert response.response_metadata["finish_reason"] == "max_output_tokens"
    assert provider.response_ids == ["resp_test_123"]
