from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router as api_router
from .config import Settings, get_settings
from .deps import get_runtime
from .health import router as health_router


def create_app(
    *,
    settings: Settings | None = None,
    runtime_provider=None,
) -> FastAPI:
    app_settings = settings or get_settings()
    app = FastAPI(
        title="Fidel-AI Inference API",
        version="1.0.0",
        description="OpenAI-compatible chat completion API backed by a local model.",
    )

    app.state.settings = app_settings
    app.state.runtime_provider = runtime_provider or get_runtime

    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, tags=["inference"])
    app.include_router(health_router, tags=["health"])
    return app


app = create_app()
