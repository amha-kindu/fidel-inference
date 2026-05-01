from __future__ import annotations

from collections import Counter
from typing import Iterator

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import InferenceConfig
from .kv_cache import SlidingKVCache
from .preprocessor import AmharicPreprocessor
from .utils import first_stop_index, safe_emittable_length


class InferenceEngine:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: spm.SentencePieceProcessor,
        *,
        device: torch.device,
        amp_enabled: bool,
        preprocessor: AmharicPreprocessor | None = None,
        system_prompt: str = "",
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.amp_enabled = amp_enabled
        self.preprocessor = preprocessor or AmharicPreprocessor()
        self.max_len = self.model.config.seq_len

        self.space_marker = "\u2581"
        self.space_marker_id = self.tokenizer.PieceToId(self.space_marker)
        self.pad_token_id = self.tokenizer.PieceToId("[PAD]")
        self.unk_token_id = self.tokenizer.PieceToId("[UNK]")
        self.bot_token_id = self.tokenizer.PieceToId("[BOT]")
        self.stop_token_id = self.tokenizer.PieceToId("[STOP]")
        self.user_token_id = self.tokenizer.PieceToId("[USER]")
        self.system_token_id = self.tokenizer.PieceToId("[SYSTEM]")
        self.system_tokens = self._encode_system_prompt(system_prompt)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.Encode(text, out_type=int))

    def generate(self, prompt: str, config: InferenceConfig) -> Iterator[str]:
        prompt_token_ids = self.encode_prompt(prompt)
        yield from self._stream_text(prompt_token_ids, config)

    def encode_prompt(self, prompt: str) -> list[int]:
        normalized_prompt = self.preprocessor.execute(prompt)
        token_ids = self.tokenizer.Encode(normalized_prompt, out_type=int)
        if self.space_marker_id < 0:
            return token_ids
        return [token_id for token_id in token_ids if token_id != self.space_marker_id]

    def decode_token(self, token_id: int) -> str:
        token = self.tokenizer.IdToPiece(token_id)
        if not token:
            return ""
        token = token.replace(self.space_marker, " ")
        token = token.replace(self.preprocessor.newline_placeholder, "\n")
        token = token.replace(self.preprocessor.tab_placeholder, "\t")
        return token

    def _encode_system_prompt(self, system_prompt: str) -> list[int]:
        if not system_prompt:
            return []
        normalized = self.preprocessor.execute(system_prompt)
        return [self.system_token_id, *self.tokenizer.Encode(normalized, out_type=int)]

    def _stream_text(self, prompt_token_ids: list[int], config: InferenceConfig) -> Iterator[str]:
        stop_sequences = tuple(stop for stop in config.stop if stop)
        if not stop_sequences:
            for token_id in self._generate_token_ids(prompt_token_ids, config):
                piece = self.decode_token(token_id)
                if piece:
                    yield piece
            return

        pending = ""
        for token_id in self._generate_token_ids(prompt_token_ids, config):
            piece = self.decode_token(token_id)
            if not piece:
                continue

            pending += piece
            stop_index = first_stop_index(pending, stop_sequences)
            if stop_index is not None:
                if stop_index > 0:
                    yield pending[:stop_index]
                return

            emit_length = safe_emittable_length(pending, stop_sequences)
            if emit_length > 0:
                yield pending[:emit_length]
                pending = pending[emit_length:]

        if not pending:
            return

        stop_index = first_stop_index(pending, stop_sequences)
        if stop_index is not None:
            if stop_index > 0:
                yield pending[:stop_index]
            return
        yield pending

    def _build_kv_caches(self, config: InferenceConfig) -> list[SlidingKVCache]:
        if config.kv_cache_size <= 0:
            return []
        return [SlidingKVCache(config.kv_cache_size) for _ in range(self.model.config.n_blocks)]

    def _ban_tokens(self, logits: torch.Tensor, banned_token_ids: list[int]) -> None:
        for token_id in banned_token_ids:
            if token_id >= 0:
                logits[token_id] = -float("inf")

    def _no_repeat_ngrams_ids(self, history: list[int], n: int) -> list[int]:
        if n <= 1 or len(history) < n - 1:
            return []

        prefix = tuple(history[-(n - 1) :])
        bans: set[int] = set()
        for index in range(len(history) - n + 1):
            if tuple(history[index : index + n - 1]) == prefix:
                bans.add(history[index + n - 1])
        return list(bans)

    def _apply_penalties(
        self,
        logits: torch.Tensor,
        history: list[int],
        config: InferenceConfig,
    ) -> None:
        if not history:
            return

        counts = Counter(history[-config.rep_window :])
        if (
            config.repetition_penalty <= 1.0
            and config.freq_penalty <= 0.0
            and config.presence_penalty <= 0.0
        ):
            return

        for token_id, count in counts.items():
            if token_id < 0:
                continue

            value = logits[token_id]
            logits[token_id] = torch.where(
                value > 0,
                value / config.repetition_penalty,
                value * config.repetition_penalty,
            )
            logits[token_id] = logits[token_id] - (
                config.freq_penalty * float(count) + config.presence_penalty
            )

    @torch.no_grad()
    def _generate_token_ids(
        self,
        prompt_token_ids: list[int],
        config: InferenceConfig,
    ) -> Iterator[int]:
        token_ids = [*self.system_tokens, *prompt_token_ids]
        if not token_ids:
            return

        temperature = min(config.temperature, config.max_temp) + 1e-5
        kv_caches = self._build_kv_caches(config)
        use_kv_cache = bool(kv_caches)
        generated_tokens = 0

        while token_ids and len(token_ids) < self.max_len and generated_tokens < config.max_tokens:
            decoder_input = torch.tensor([token_ids], dtype=torch.int64, device=self.device)

            with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                logits: torch.Tensor = self.model(decoder_input, None, use_kv_cache, kv_caches)

            logits = logits[0, -1, :].float() / temperature
            self._ban_tokens(
                logits,
                [
                    self.unk_token_id,
                    self.user_token_id,
                    self.bot_token_id,
                    self.system_token_id,
                    self.pad_token_id,
                    *self._no_repeat_ngrams_ids(token_ids, config.no_repeat_ngram_size),
                ],
            )
            self._apply_penalties(logits, token_ids, config)

            if config.top_k and config.top_k > 0:
                top_k_logits, top_k_indices = torch.topk(
                    logits,
                    k=min(config.top_k, logits.size(0)),
                    dim=0,
                )
                logits = torch.full_like(logits, float("-inf")).scatter(
                    dim=0,
                    index=top_k_indices,
                    src=top_k_logits,
                )

            if config.top_p and config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=0)
                sorted_probs = F.softmax(sorted_logits, dim=0)
                cumulative_probabilities = torch.cumsum(sorted_probs, dim=0)
                mask = cumulative_probabilities <= config.top_p
                mask[0] = True
                logits = torch.full_like(logits, float("-inf")).scatter(
                    dim=0,
                    index=sorted_indices[mask],
                    src=sorted_logits[mask],
                )

            probabilities = F.softmax(logits, dim=0)
            mass = probabilities.sum(dim=0, keepdim=True)

            if (mass <= 1e-12).any():
                predicted_token = int(logits.argmax(dim=0).item())
            else:
                predicted_token = int(torch.multinomial(probabilities, num_samples=1).item())

            if predicted_token == self.stop_token_id:
                return

            token_ids.append(predicted_token)
            generated_tokens += 1
            yield predicted_token
