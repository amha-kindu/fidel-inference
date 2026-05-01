from __future__ import annotations

from types import SimpleNamespace

import torch

from app.config import InferenceConfig
from app.inference import InferenceEngine


class DummyTokenizer:
    def __init__(self) -> None:
        pieces = [
            "[PAD]",
            "[UNK]",
            "[BOT]",
            "[STOP]",
            "[USER]",
            "[SYSTEM]",
            "\u2581",
            "P",
            "H",
            "e",
            "l",
            "o",
            "!",
            "A",
            "B",
            "C",
        ]
        self._piece_to_id = {piece: index for index, piece in enumerate(pieces)}
        self._id_to_piece = {index: piece for piece, index in self._piece_to_id.items()}

    def PieceToId(self, piece: str) -> int:
        return self._piece_to_id[piece]

    def IdToPiece(self, token_id: int) -> str:
        return self._id_to_piece[token_id]

    def Encode(self, text: str, out_type=int) -> list[int]:
        return [
            self._piece_to_id[character]
            for character in text
            if character in self._piece_to_id
        ]


class DummyModel(torch.nn.Module):
    def __init__(self, output_tokens: list[int], vocab_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(seq_len=64, n_blocks=1)
        self._output_tokens = output_tokens
        self._vocab_size = vocab_size

    def forward(self, x, mask, use_cache=False, kv_caches=None):
        step = max(0, x.shape[1] - 1)
        next_token = self._output_tokens[min(step, len(self._output_tokens) - 1)]
        logits = torch.full((1, x.shape[1], self._vocab_size), -1e9)
        logits[0, -1, next_token] = 1e9
        return logits


def _build_engine(output_tokens: list[int]) -> InferenceEngine:
    tokenizer = DummyTokenizer()
    model = DummyModel(output_tokens, vocab_size=len(tokenizer._piece_to_id))
    return InferenceEngine(
        model,
        tokenizer,
        device=torch.device("cpu"),
        amp_enabled=False,
    )


def test_inference_engine_applies_stop_sequences_across_token_boundaries():
    tokenizer = DummyTokenizer()
    engine = _build_engine(
        [
            tokenizer.PieceToId("H"),
            tokenizer.PieceToId("e"),
            tokenizer.PieceToId("l"),
            tokenizer.PieceToId("l"),
            tokenizer.PieceToId("o"),
            tokenizer.PieceToId("[STOP]"),
        ]
    )

    result = "".join(
        engine.generate(
            "P",
            InferenceConfig(max_tokens=8, stop=("lo",)),
        )
    )

    assert result == "Hel"


def test_inference_engine_does_not_retain_prior_request_state():
    tokenizer = DummyTokenizer()
    engine = _build_engine(
        [
            tokenizer.PieceToId("A"),
            tokenizer.PieceToId("B"),
            tokenizer.PieceToId("C"),
            tokenizer.PieceToId("[STOP]"),
        ]
    )

    first = "".join(engine.generate("P", InferenceConfig(max_tokens=8, stop=("BC",))))
    second = "".join(engine.generate("P", InferenceConfig(max_tokens=8)))

    assert first == "A"
    assert second == "ABC"
