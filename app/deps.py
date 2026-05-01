from __future__ import annotations

from functools import lru_cache
from typing import Callable

from fastapi import Request

from .config import get_settings
from .runtime import InferenceRuntime

RuntimeProvider = Callable[[], InferenceRuntime]


@lru_cache
def create_runtime() -> InferenceRuntime:
    return InferenceRuntime(get_settings())


def get_runtime() -> InferenceRuntime:
    return create_runtime()


def resolve_runtime(request: Request) -> InferenceRuntime:
    provider: RuntimeProvider | None = getattr(request.app.state, "runtime_provider", None)
    if provider is None:
        return get_runtime()
    return provider()


def reset_runtime_state() -> None:
    create_runtime.cache_clear()
    get_settings.cache_clear()
