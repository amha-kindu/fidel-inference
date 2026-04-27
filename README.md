# Fidel: High-Performance Model Inference Server

A production-ready **FastAPI** backend designed for serving Large Language Models (LLMs) via an OpenAI-compatible interface. This project focuses strictly on the engineering challenges of model serving: asynchronous concurrency, GPU memory management, and high-throughput streaming.

## 🚀 System Overview

`fidel-inference` provides a robust API layer for model inference, abstracting complex PyTorch/Cuda logic behind a standardized RESTful interface.

- **OpenAI-Compatible API**: Implements `/v1/models` and `/v1/chat/completions` for drop-in compatibility with existing AI ecosystems.
- **Asynchronous Streaming**: Full support for Server-Sent Events (SSE) to provide real-time token streaming.
- **Resource Management**: Implements an inference lock mechanism to serialize GPU access, preventing memory fragmentation and race conditions during generation.
- **Production-Grade Orchestration**: Optimized Docker & Gunicorn configurations tailored for compute-intensive Python workloads.

---

## 🏗 Backend Architecture

### Inference Engine (FastAPI + PyTorch)
The core logic handles the transformation of HTTP requests into model tensors:
- **Streaming & Non-Streaming Modes**: Unified logic for handling both standard JSON responses and generator-based streams.
- **Sampling Logic**: Configurable temperature, top-p, top-k, and stop-token handling.
- **Concurrency**: Native FastAPI `async` support to handle I/O-bound tasks while the GPU handles the compute-bound generation.
- **Health Monitoring**: Dedicated `/health` endpoint for integration with load balancers and container orchestrators (K8s/GKE).



### Deployment & Scalability
- **Containerization**: Optimized Dockerfiles for both **CPU** and **CUDA** environments.
- **Process Management**: Uses Gunicorn with custom worker tuning (`gunicorn_conf.py`) to manage keepalive timeouts and worker recycling.
- **CORS & Security**: Configurable middleware to secure the API for specific consumer origins.

---

## ⚙ Configuration

The server is entirely environment-driven, allowing for seamless transitions between development and production environments.

| Variable | Description |
|---|---|
| `MODEL_ID` | The identifier returned in the OpenAI-compatible response. |
| `AMP` | Toggle Automatic Mixed Precision for optimized GPU throughput. |
| `ALLOW_ORIGINS` | Strict CORS configuration for API security. |
| `MAX_TOKENS` | Hard limit on generation length to prevent resource exhaustion. |
| `INFERENCE_TIMEOUT` | Global timeout for long-running generation tasks. |

---

## 💡 Quick Start

1. **Environment Setup**:
   ```bash
   cp .env.example .env
   # Set your MODEL_ID and hardware preferences
   ```

2. **Launch with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

3. **Verify Service**:
   ```bash
   curl http://localhost:8000/v1/models
   ```

---

## 🛠 Engineering Highlights
* **Serialization**: Efficient handling of concurrent requests using an async-aware locking strategy.
* **OpenAPI Specs**: Automatically generates interactive documentation (Swagger/ReDoc) for easy integration testing.
* **Logging**: Structured logging setup to track inference latency and hardware utilization.
