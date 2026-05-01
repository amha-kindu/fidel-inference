import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Body, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from .chat_format import render_chat
from .deps import resolve_runtime
from .runtime import InferenceRuntime
from .schemas import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatRequest,
    ModelListResponse,
)
from .sse import sse_stream

router = APIRouter()
RuntimeDependency = Annotated[InferenceRuntime, Depends(resolve_runtime)]
ChatRequestBody = Annotated[ChatRequest, Body(...)]


@router.get(
    "/v1/models",
    response_model=ModelListResponse,
    summary="List available models",
    description="Returns all models currently served by this instance.",
    responses={
        200: {
            "description": "List of models currently loaded and available.",
        }
    },
)
def list_models(runtime: RuntimeDependency):
    return {"object": "list", "data": runtime.get_available_models()}


@router.post(
    "/v1/chat/completions",
    summary="Create chat completion",
    description="""
Chat-style completion endpoint with optional streaming.

- If `stream=false` (default): returns one `chat.completion` JSON object.
- If `stream=true`: returns `text/event-stream` (SSE). Each event is a
  `chat.completion.chunk` with incremental deltas, similar to OpenAI.

**Streaming event shape (`stream=true`)**:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion.chunk",
  "created": 1730025600,
  "model": "my-model-name",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "partial text here"
      },
      "finish_reason": null
    }
  ]
}
````

The last event has `finish_reason: "stop"` and `delta: {}`.
""",
    responses={
        200: {
            "description": "Non-streaming chat completion response",
            "model": ChatCompletionResponse,
        },
        206: {
            "description": "Streaming SSE chunks (only if `stream=true`). "
            "Each chunk is `chat.completion.chunk`.",
            "model": ChatCompletionChunk,
            "content": {
                "text/event-stream": {
                    "schema": ChatCompletionChunk.model_json_schema(),
                }
            },
        },
    },
    openapi_extra={
        "x-streaming-response": {
            "content-type": "text/event-stream",
            "chunk-schema-ref": "#/components/schemas/ChatCompletionChunk",
        }
    },
)
def chat_completions(
    req: ChatRequestBody,
    runtime: RuntimeDependency,
):
    created = int(time.time())
    chat_id = f"chatcmpl-{uuid.uuid4()}"
    prompt = render_chat([message.model_dump() for message in req.messages])
    inference_config = runtime.build_inference_config(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        stop=req.stop or (),
    )

    if req.stream:

        def iterator():
            yield {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }

            for piece in runtime.generate_text(prompt, inference_config):
                yield {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": piece, "role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }

            yield {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }

        return StreamingResponse(
            sse_stream(iterator()),
            media_type="text/event-stream",
            status_code=206,
        )

    text = "".join(runtime.generate_text(prompt, inference_config))
    prompt_tokens = runtime.count_tokens(prompt)
    completion_tokens = runtime.count_tokens(text)

    body = ChatCompletionResponse(
        id=chat_id,
        created=created,
        model=req.model,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )
    return JSONResponse(body.model_dump())
