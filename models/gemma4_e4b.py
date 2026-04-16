from __future__ import annotations

import time
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from models.base import AudioModel, InferenceResult

_MODEL_ID = "google/gemma-4-E4B-it"
_SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


class Gemma4E4BModel(AudioModel):
    def __init__(self, device: str = "cuda") -> None:
        self._requested_device = device
        self._device: str | None = None
        self._model = None
        self._processor = None

    @property
    def display_name(self) -> str:
        return "Gemma-4-E4B"

    @property
    def model_id(self) -> str:
        return _MODEL_ID

    def load(self) -> None:
        if self._model is not None:
            return

        device = self._requested_device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self._device = device

        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        _processor = AutoProcessor.from_pretrained(_MODEL_ID)
        _model = AutoModelForMultimodalLM.from_pretrained(
            _MODEL_ID,
            dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )

        if device != "cuda":
            _model = _model.to(device)

        _model.eval()
        self._processor = _processor
        self._model = _model

    def run_inference(
        self,
        audio_path: Path,
        question: str,
        max_new_tokens: int = 512,
    ) -> InferenceResult:
        if audio_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format: {audio_path.suffix}. "
                f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
            )

        if self._model is None or self._processor is None:
            raise RuntimeError("Model is not loaded. Call load() before run_inference().")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(audio_path)},
                    {"type": "text", "text": question},
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._device)

        input_len = inputs["input_ids"].shape[-1]

        t0 = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        answer = self._processor.decode(
            generated_ids[0][input_len:],
            skip_special_tokens=True,
        )

        return InferenceResult(
            answer=answer,
            latency_ms=latency_ms,
            model_id=_MODEL_ID,
        )
