from __future__ import annotations

import base64
import logging
import os
import re
import time
from pathlib import Path

import httpx

from models.base import AudioModel, InferenceResult

logger = logging.getLogger(__name__)

_MODEL_ID = "stepfun-ai/Step-Audio-R1.1"
_SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
_AUDIO_MIME: dict[str, str] = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
}
_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)
_VLLM_READY_TIMEOUT = 600  # seconds


def _split_thinking(text: str) -> tuple[str | None, str]:
    m = _THINK_RE.match(text)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    return None, text


class StepAudioR1Model(AudioModel):
    def __init__(self, device: str = "cuda") -> None:
        self._vllm_url = os.environ.get(
            "STEP_AUDIO_VLLM_URL", "http://step-audio-vllm:9999"
        )

    @property
    def display_name(self) -> str:
        return "Step-Audio-R1.1"

    @property
    def model_id(self) -> str:
        return _MODEL_ID

    def load(self) -> None:
        logger.info("Waiting for vLLM server at %s ...", self._vllm_url)
        deadline = time.time() + _VLLM_READY_TIMEOUT
        while time.time() < deadline:
            try:
                r = httpx.get(f"{self._vllm_url}/health", timeout=5.0)
                if r.status_code == 200:
                    logger.info("vLLM server ready.")
                    return
            except Exception:
                pass
            time.sleep(5)
        raise RuntimeError(
            f"vLLM server at {self._vllm_url} did not become ready within "
            f"{_VLLM_READY_TIMEOUT}s"
        )

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

        mime = _AUDIO_MIME.get(audio_path.suffix.lower(), "audio/wav")
        audio_b64 = base64.b64encode(audio_path.read_bytes()).decode()
        audio_data_url = f"data:{mime};base64,{audio_b64}"

        payload = {
            "model": _MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": audio_data_url},
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }

        t0 = time.perf_counter()
        with httpx.Client(timeout=300.0) as client:
            r = client.post(
                f"{self._vllm_url}/v1/chat/completions", json=payload
            )
        r.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000

        content = r.json()["choices"][0]["message"]["content"]
        thinking, answer = _split_thinking(content)

        return InferenceResult(
            answer=answer,
            latency_ms=latency_ms,
            model_id=_MODEL_ID,
            thinking=thinking,
        )
