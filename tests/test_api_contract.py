from __future__ import annotations

import json


def _parse_sse_payloads(body: str) -> list[object]:
    payloads: list[object] = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        raw_payload = line.removeprefix("data: ")
        if raw_payload == "[DONE]":
            payloads.append(raw_payload)
            continue
        payloads.append(json.loads(raw_payload))
    return payloads


def test_health_contract(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models_contract_preserves_envelope_and_model_fields(client):
    response = client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "fidel-chat-v1-125M"
    assert body["data"][0]["name"] == "Fidel Chat 125M"
    assert body["data"][0]["description"] == "Model-free contract test payload"


def test_chat_completion_non_stream_contract(client, fake_runtime):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fidel-chat-v1-125M",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "fidel-chat-v1-125M"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "Hello world"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert set(body["usage"]) == {"prompt_tokens", "completion_tokens", "total_tokens"}
    assert fake_runtime.last_request is not None
    assert fake_runtime.last_request.prompt.endswith("[BOT]")


def test_chat_completion_stream_contract(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fidel-chat-v1-125M",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    )

    assert response.status_code == 206
    payloads = _parse_sse_payloads(response.text)
    assert payloads[-1] == "[DONE]"

    first_event = payloads[0]
    assert first_event["object"] == "chat.completion.chunk"
    assert first_event["choices"][0]["delta"] == {"role": "assistant"}

    text_events = payloads[1:-2]
    streamed_text = "".join(
        event["choices"][0]["delta"]["content"] for event in text_events
    )
    assert streamed_text == "Hello world"

    final_event = payloads[-2]
    assert final_event["choices"][0]["delta"] == {}
    assert final_event["choices"][0]["finish_reason"] == "stop"


def test_stop_sequences_are_applied_in_stream_and_non_stream_modes(client):
    payload = {
        "model": "fidel-chat-v1-125M",
        "messages": [{"role": "user", "content": "Hello"}],
        "stop": [" world"],
    }

    non_stream_response = client.post("/v1/chat/completions", json=payload)
    assert non_stream_response.status_code == 200
    assert non_stream_response.json()["choices"][0]["message"]["content"] == "Hello"

    stream_response = client.post(
        "/v1/chat/completions",
        json={**payload, "stream": True},
    )
    assert stream_response.status_code == 206
    payloads = _parse_sse_payloads(stream_response.text)
    text_events = payloads[1:-2]
    assert "".join(event["choices"][0]["delta"]["content"] for event in text_events) == "Hello"
