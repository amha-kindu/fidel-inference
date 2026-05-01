from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy
import torch

random.seed(4321)
torch.manual_seed(4321)
numpy.random.seed(4321)

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_ROOT = REPO_ROOT / "app"


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _parse_int(value: str | None, *, default: int | None = None) -> int | None:
    if value is None or value.strip() == "":
        return default
    return int(value)


def _parse_float(value: str | None, *, default: float) -> float:
    if value is None or value.strip() == "":
        return default
    return float(value)


def _parse_csv(value: str | None, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None or value.strip() == "":
        return default
    return tuple(part.strip() for part in value.split(",") if part.strip())


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(3000)
        device = torch.device("cuda")
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def is_amp_enabled(settings: "Settings", device: torch.device) -> bool:
    return (
        settings.amp
        and device.type != "cpu"
        and torch.amp.autocast_mode.is_autocast_available(device.type)
    )


DEVICE = detect_device()


@dataclass(frozen=True, slots=True)
class Settings:
    model_id: str = ""
    lora: bool = False
    tokenizer_path: str = ""
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int | None = None
    allow_origins: tuple[str, ...] = ("*",)
    amp: bool = True
    system_prompt: str = ""
    inference_timeout: int | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        source = os.environ if env is None else env
        max_tokens = _parse_int(source.get("MAX_TOKENS"), default=256)
        if max_tokens is None:
            max_tokens = 256
        return cls(
            model_id=source.get("MODEL_ID", ""),
            lora=_parse_bool(source.get("LORA"), default=False),
            tokenizer_path=source.get("TOKENIZER_PATH", ""),
            max_tokens=max_tokens,
            temperature=_parse_float(source.get("TEMPERATURE"), default=0.7),
            top_p=_parse_float(source.get("TOP_P"), default=0.9),
            top_k=_parse_int(source.get("TOP_K")),
            allow_origins=_parse_csv(source.get("ALLOW_ORIGINS"), default=("*",)),
            amp=_parse_bool(source.get("AMP"), default=True),
            system_prompt=source.get("SYSTEM_PROMPT", ""),
            inference_timeout=_parse_int(source.get("INFERENCE_TIMEOUT")),
        )

    @property
    def model_dir(self) -> Path:
        return (APP_ROOT / "models" / self.model_id).resolve()

    @property
    def model_list_path(self) -> Path:
        return (APP_ROOT / "models" / "list.json").resolve()

    @property
    def tokenizer_file(self) -> Path:
        return self.resolve_app_path(self.tokenizer_path)

    def resolve_app_path(self, relative_path: str | Path) -> Path:
        return (APP_ROOT / Path(relative_path)).resolve()

    def require_path(self, path: Path, label: str) -> Path:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found at {path}")
        return path

    def cors_origins(self) -> list[str]:
        return list(self.allow_origins)


@dataclass(slots=True)
class ModelConfig:
    embed_dim: int = 512
    n_blocks: int = 6
    vocab_size: int = 25000
    ff_dim: int = 2048
    heads: int = 8
    dropout: float = 0.1
    seq_len: int = 50
    metric_dim: int = 128
    epsilon: float = 1e-2

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelConfig":
        data = dict(payload)
        if "metric_epsilon" in data and "epsilon" not in data:
            data["epsilon"] = data.pop("metric_epsilon")
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in data.items() if key in allowed})


@dataclass(slots=True)
class ModelWithLoRAConfig(ModelConfig):
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_targets: dict[str, Any] = field(default_factory=dict)
    lora_dropout: float = 0.05

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelWithLoRAConfig":
        data = dict(payload)
        if "metric_epsilon" in data and "epsilon" not in data:
            data["epsilon"] = data.pop("metric_epsilon")
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in data.items() if key in allowed})


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    max_tokens: int = 256
    top_k: int | None = 0
    top_p: float = 1.0
    temperature: float = 1.0
    max_temp: float = 2.0
    repetition_penalty: float = 1.15
    presence_penalty: float = 0.0
    freq_penalty: float = 0.3
    no_repeat_ngram_size: int = 3
    rep_window: int = 200
    kv_cache_size: int = 0
    stop: tuple[str, ...] = ()

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        **overrides: Any,
    ) -> "InferenceConfig":
        payload: dict[str, Any] = {
            "max_tokens": settings.max_tokens,
            "top_k": settings.top_k,
            "top_p": settings.top_p,
            "temperature": settings.temperature,
        }
        for key, value in overrides.items():
            if value is not None:
                payload[key] = value

        stop_value = payload.get("stop", ())
        if isinstance(stop_value, str):
            payload["stop"] = (stop_value,)
        else:
            payload["stop"] = tuple(stop_value or ())

        return cls(**payload)


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
