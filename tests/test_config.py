from __future__ import annotations

import pytest

from app.config import APP_ROOT, Settings
from app.runtime import InferenceRuntime


def test_settings_parse_false_empty_top_k_and_origins():
    settings = Settings.from_env(
        {
            "MODEL_ID": "fidel-chat-v1-125M",
            "LORA": "false",
            "TOKENIZER_PATH": "tokenizers/amharic-bpe-tokenizer-25k.model",
            "TOP_K": "",
            "ALLOW_ORIGINS": "https://one.example, https://two.example",
        }
    )

    assert settings.lora is False
    assert settings.top_k is None
    assert settings.allow_origins == ("https://one.example", "https://two.example")


def test_settings_resolve_paths_from_repo_root():
    settings = Settings(
        model_id="fidel-chat-v1-125M",
        tokenizer_path="tokenizers/amharic-bpe-tokenizer-25k.model",
    )

    assert settings.tokenizer_file == (
        APP_ROOT / "tokenizers" / "amharic-bpe-tokenizer-25k.model"
    ).resolve()
    assert settings.model_dir == (APP_ROOT / "models" / "fidel-chat-v1-125M").resolve()


def test_runtime_requires_configured_tokenizer_path():
    runtime = InferenceRuntime(Settings(model_id="fidel-chat-v1-125M"))

    with pytest.raises(FileNotFoundError, match="Tokenizer path is not configured"):
        runtime.ensure_ready()


def test_request_scoped_inference_configs_do_not_leak():
    runtime = InferenceRuntime(
        Settings(
            model_id="fidel-chat-v1-125M",
            tokenizer_path="tokenizers/amharic-bpe-tokenizer-25k.model",
            temperature=0.7,
            top_p=0.9,
        )
    )

    first = runtime.build_inference_config(top_k=5, stop=["END"])
    second = runtime.build_inference_config(top_k=10)

    assert first.top_k == 5
    assert first.stop == ("END",)
    assert second.top_k == 10
    assert second.stop == ()
