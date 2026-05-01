<div align="center">
  <h1>Fidel Inference API</h1>
  <p><strong>OpenAI-compatible FastAPI backend for real-time custom language model inference</strong></p>
  <p>
    SSE chat streaming, centralized runtime loading, model serving for Fidel checkpoints,
    CPU and CUDA deployment paths, and a production-shaped inference layer in a single service.
  </p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+">
    <img src="https://img.shields.io/badge/FastAPI-0.116.2-009688?logo=fastapi&logoColor=white" alt="FastAPI 0.116.2">
    <img src="https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white" alt="Pydantic v2">
    <img src="https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.4.1">
    <img src="https://img.shields.io/badge/Poetry-Managed-60A5FA?logo=poetry&logoColor=white" alt="Poetry Managed">
    <img src="https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker&logoColor=white" alt="Docker Supported">
    <img src="https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia&logoColor=white" alt="CUDA 12.1">
    <img src="https://img.shields.io/badge/Streaming-SSE-C2410C" alt="Streaming SSE">
    <img src="https://img.shields.io/badge/API-OpenAI%20Compatible-111111" alt="OpenAI Compatible API">
  </p>
</div>

---

## Overview

Fidel Inference is the backend serving layer for Fidel chat checkpoints. It exposes a compact,
OpenAI-compatible HTTP surface while keeping model loading, tokenization, and generation logic
inside a controlled runtime boundary.

It is designed for teams that want:

- a stable `/v1/models` and `/v1/chat/completions` interface
- stateless request handling where clients provide conversation context per call
- streaming chat completions over Server-Sent Events
- containerized CPU and CUDA deployment options
- a codebase that is straightforward to test, lint, and operate

## What Makes It Useful

- **OpenAI-style API**: easy integration with existing SDKs, tools, and WebUI clients.
- **Stateless request handling**: each chat completion request is self-contained, and the server
  does not persist conversation state between calls.
- **Streaming-first support**: sends incremental `chat.completion.chunk` events over SSE.
- **Runtime safety**: model/tokenizer bootstrap is centralized behind a lazy runtime layer.
- **Operational simplicity**: Gunicorn + Uvicorn worker setup, health checks, and Docker flows.
- **Hardware flexibility**: supports both CPU-only and CUDA-backed deployments.

## API Surface

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness check for local/dev/container health probes |
| `GET` | `/v1/models` | Returns the models exposed by this instance |
| `POST` | `/v1/chat/completions` | OpenAI-style chat completion endpoint with optional streaming |

### Streaming Behavior

When `stream=true`, the server returns `text/event-stream` and emits:

1. an initial assistant role chunk
2. incremental content chunks
3. a final stop chunk
4. the `[DONE]` sentinel

## Quick Start

### 1. Configure the environment

```bash
cp .env.example .env
```

Set the model and tokenizer values in `.env` to match the assets you want to serve.

### 2. Install project dependencies

```bash
poetry install
```

Install the Torch build that matches your environment.

CPU:

```bash
poetry run python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1
```

CUDA 12.1:

```bash
poetry run python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1
```

### 3. Run the API

```bash
poetry run gunicorn app.main:app -k uvicorn.workers.UvicornWorker -c gunicorn_conf.py
```

The service binds to `http://localhost:7890`.

### 4. Verify the service

```bash
curl http://localhost:7890/health
curl http://localhost:7890/v1/models
```

## Example Request

Non-streaming chat completion:

```bash
curl -X POST http://localhost:7890/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fidel-chat-v1-125M",
    "messages": [
      {"role": "user", "content": "Selam"}
    ]
  }'
```

Streaming chat completion:

```bash
curl -N -X POST http://localhost:7890/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fidel-chat-v1-125M",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Tell me about Addis Ababa."}
    ]
  }'
```

## Configuration

The service is environment-driven. Key variables:

| Variable | Description |
| --- | --- |
| `MODEL_ID` | Directory name loaded from `app/models/<MODEL_ID>` |
| `TOKENIZER_PATH` | Tokenizer path relative to `app/` |
| `LORA` | Enables LoRA checkpoint loading when set to `true` |
| `SYSTEM_PROMPT` | Optional prompt prepended internally during inference |
| `AMP` | Enables automatic mixed precision when supported |
| `ALLOW_ORIGINS` | Comma-separated CORS origin list |
| `MAX_TOKENS` | Default completion token limit |
| `TOP_P` | Default nucleus sampling value |
| `TOP_K` | Default top-k sampling value |
| `TEMPERATURE` | Default sampling temperature |

### Expected Model Layout

The runtime expects model assets under:

```text
app/models/<MODEL_ID>/
```

Typical files include:

- `checkpoint.pt`
- `metadata.json`
- `checkpoint-lora.pt`
- `metadata-lora.json`

## Docker

### CUDA stack with Open WebUI

```bash
docker compose -f compose-gpu.yml up --build
```

This brings up:

- the inference API on `http://localhost:7890`
- Open WebUI on `http://localhost:2345`

### CPU-only image

```bash
docker build -f Dockerfile.cpu -t fidel-inference-cpu .
docker run --env-file .env -p 7890:7890 fidel-inference-cpu
```

## Development Quality Checks

Run linting:

```bash
poetry run ruff check .
```

Run type checking:

```bash
poetry run mypy
```

Run tests:

```bash
poetry run pytest
```

## Project Notes

- The API contract is intentionally narrow and stable.
- Runtime bootstrap is centralized so model loading is not scattered across imports.
- The repository is optimized for inference serving concerns, not model training workflows.

---

Built to serve Fidel checkpoints with a clean developer experience and a production-shaped API.
