from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import InferenceConfig, Settings
from app.main import create_app


@dataclass
class RecordedRequest:
    prompt: str
    config: InferenceConfig


class FakeRuntime:
    def __init__(self) -> None:
        self.available_models = [
            {
                "id": "fidel-chat-v1-125M",
                "object": "model",
                "owned_by": "fidel",
                "name": "Fidel Chat 125M",
                "description": "Model-free contract test payload",
            }
        ]
        self.generated_text = "Hello world"
        self.last_request: RecordedRequest | None = None

    def build_inference_config(self, **overrides: Any) -> InferenceConfig:
        max_tokens = overrides.get("max_tokens")
        temperature = overrides.get("temperature")
        top_p = overrides.get("top_p")
        return InferenceConfig(
            max_tokens=256 if max_tokens is None else max_tokens,
            top_k=overrides.get("top_k"),
            top_p=0.9 if top_p is None else top_p,
            temperature=0.7 if temperature is None else temperature,
            stop=tuple(overrides.get("stop") or ()),
        )

    def get_available_models(self) -> list[dict[str, Any]]:
        return self.available_models

    def generate_text(self, prompt: str, config: InferenceConfig) -> Iterator[str]:
        self.last_request = RecordedRequest(prompt=prompt, config=config)
        text = self.generated_text
        stop_positions = [text.find(stop) for stop in config.stop if stop and stop in text]
        if stop_positions:
            text = text[: min(stop_positions)]

        if not text:
            return

        midpoint = max(1, len(text) // 2)
        yield text[:midpoint]
        if midpoint < len(text):
            yield text[midpoint:]

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(text.split())


@pytest.fixture
def fake_runtime() -> FakeRuntime:
    return FakeRuntime()


@pytest.fixture
def app(fake_runtime: FakeRuntime):
    settings = Settings(
        model_id="fidel-chat-v1-125M",
        tokenizer_path="tokenizers/amharic-bpe-tokenizer-25k.model",
    )
    return create_app(settings=settings, runtime_provider=lambda: fake_runtime)


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
