import json
from typing import Iterable


def sse_pack(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def sse_stream(chunks: Iterable[dict]):
    for c in chunks:
        yield sse_pack(c)
    yield "data: [DONE]\n\n"
