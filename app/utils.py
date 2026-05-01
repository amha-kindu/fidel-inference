from __future__ import annotations

from typing import Sequence


def first_stop_index(text: str, stop_sequences: Sequence[str]) -> int | None:
    positions = [text.find(stop) for stop in stop_sequences if stop and text.find(stop) >= 0]
    if not positions:
        return None
    return min(positions)


def safe_emittable_length(text: str, stop_sequences: Sequence[str]) -> int:
    normalized = [stop for stop in stop_sequences if stop]
    if not normalized:
        return len(text)

    longest = max(len(stop) for stop in normalized)
    if longest <= 1:
        return len(text)
    return max(0, len(text) - longest + 1)
