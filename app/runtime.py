from __future__ import annotations

import importlib
import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch

from .config import (
    DEVICE,
    InferenceConfig,
    ModelConfig,
    ModelWithLoRAConfig,
    Settings,
    is_amp_enabled,
)
from .inference import InferenceEngine
from .logging import LOGGER
from .models.lora import LoRAdapter
from .preprocessor import AmharicPreprocessor


@dataclass(slots=True)
class RuntimeAssets:
    engine: InferenceEngine
    tokenizer: spm.SentencePieceProcessor
    preprocessor: AmharicPreprocessor


class InferenceRuntime:
    def __init__(self, settings: Settings, *, device: torch.device = DEVICE) -> None:
        self.settings = settings
        self.device = device
        self.amp_enabled = is_amp_enabled(settings, device)
        self._bootstrap_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._assets: RuntimeAssets | None = None
        self._available_models: list[dict[str, Any]] | None = None

    def build_inference_config(self, **overrides: Any) -> InferenceConfig:
        return InferenceConfig.from_settings(self.settings, **overrides)

    def get_available_models(self) -> list[dict[str, Any]]:
        if self._available_models is None:
            model_list_path = self.settings.require_path(
                self.settings.model_list_path,
                "Model list",
            )
            with model_list_path.open("r", encoding="utf-8") as handle:
                self._available_models = json.load(handle)
        return self._available_models

    def count_tokens(self, text: str) -> int:
        return self._ensure_assets().engine.count_tokens(text)

    def generate_text(self, prompt: str, config: InferenceConfig) -> Iterator[str]:
        assets = self._ensure_assets()
        with self._inference_lock:
            yield from assets.engine.generate(prompt, config)

    def ensure_ready(self) -> None:
        self._ensure_assets()

    def _ensure_assets(self) -> RuntimeAssets:
        if self._assets is None:
            with self._bootstrap_lock:
                if self._assets is None:
                    self._assets = self._load_assets()
        return self._assets

    def _load_assets(self) -> RuntimeAssets:
        if not self.settings.tokenizer_path:
            raise FileNotFoundError("Tokenizer path is not configured.")

        tokenizer = spm.SentencePieceProcessor()
        tokenizer_path = self.settings.require_path(self.settings.tokenizer_file, "Tokenizer")
        if not tokenizer.LoadFromFile(str(tokenizer_path)):
            raise RuntimeError(f"Failed to load tokenizer from {tokenizer_path}")

        preprocessor = AmharicPreprocessor()
        model = self._load_model().to(self.device).eval()
        engine = InferenceEngine(
            model,
            tokenizer,
            device=self.device,
            amp_enabled=self.amp_enabled,
            preprocessor=preprocessor,
            system_prompt=self.settings.system_prompt,
        )
        return RuntimeAssets(engine=engine, tokenizer=tokenizer, preprocessor=preprocessor)

    def _load_model(self) -> torch.nn.Module:
        if not self.settings.model_id:
            raise FileNotFoundError("MODEL_ID is not configured.")

        model_dir = self.settings.require_path(self.settings.model_dir, "Model directory")
        checkpoint_path = self.settings.require_path(model_dir / "checkpoint.pt", "Base checkpoint")
        metadata_path = self.settings.require_path(model_dir / "metadata.json", "Model metadata")

        LOGGER.info("Loading checkpoint from %s...", checkpoint_path)
        weights = self._load_checkpoint(checkpoint_path)

        metadata = self._load_json(metadata_path)
        model_config: ModelConfig | ModelWithLoRAConfig = ModelConfig.from_dict(metadata)

        model_module = importlib.import_module(f"app.models.{self.settings.model_id}.model")
        model_class = model_module.GPTmodel

        if self.settings.lora:
            lora_checkpoint_path = self.settings.require_path(
                model_dir / "checkpoint-lora.pt",
                "LoRA checkpoint",
            )
            lora_metadata_path = self.settings.require_path(
                model_dir / "metadata-lora.json",
                "LoRA metadata",
            )
            LOGGER.info("Loading checkpoint from %s...", lora_checkpoint_path)
            weights.update(self._load_checkpoint(lora_checkpoint_path))
            model_config = ModelWithLoRAConfig.from_dict(self._load_json(lora_metadata_path))
        else:
            update_path = model_dir / "checkpoint-update.pt"
            if update_path.exists():
                LOGGER.info("Loading update from %s...", update_path)
                weights.update(self._load_checkpoint(update_path))
            else:
                LOGGER.info(
                    "No checkpoint update found at %s. Continuing with base checkpoint.",
                    update_path,
                )

        model: torch.nn.Module = model_class.build(model_config, weights=weights)

        if self.settings.lora:
            merged = False
            for module in model.modules():
                if isinstance(module, LoRAdapter):
                    module.merge()
                    merged = True
            if merged:
                LOGGER.info("Merged LoRA adapters in model...")

        total_params = sum(parameter.numel() for parameter in model.parameters())
        LOGGER.info("Device: %s", self.device)
        LOGGER.info("Total Parameters: %s", total_params)
        LOGGER.info("Model Size(MB): %.2fMB", total_params * 4 / (1024**2))
        LOGGER.info(
            "Initiating inference with %s...",
            "mixed-precision" if self.amp_enabled else "single-precision",
        )
        return model

    def _load_checkpoint(self, path: Path) -> dict[str, Any]:
        return torch.load(path, map_location=self.device, weights_only=False)

    def _load_json(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
