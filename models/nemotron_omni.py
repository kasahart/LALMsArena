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

_MODEL_ID = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
_SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
_AUDIO_MIME: dict[str, str] = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
}
_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)
_VLLM_READY_TIMEOUT = 1200


_THINK_CLOSE_RE = re.compile(r"^(.*?)</think>\s*", re.DOTALL)


def _split_thinking(text: str) -> tuple[str | None, str]:
    # Pattern 1: full <think>...</think> wrapper
    m = _THINK_RE.match(text)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    # Pattern 2: vLLM strips the opening <think> tag; only </think> remains
    m2 = _THINK_CLOSE_RE.match(text)
    if m2:
        thinking = m2.group(1).strip()
        answer = text[m2.end():].strip()
        return (thinking or None), answer
    return None, text


class NemotronOmniReasoningModel(AudioModel):
    def __init__(self, device: str = "cuda") -> None:
        self._vllm_url = os.environ.get(
            "NEMOTRON_VLLM_URL", "http://nemotron-vllm:9999"
        )

    @property
    def display_name(self) -> str:
        return "Nemotron-Nano-Omni-Reasoning"

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
            "temperature": 1.0,
            "extra_body": {
                "top_k": 1,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        }

        t0 = time.perf_counter()
        with httpx.Client(timeout=300.0) as client:
            r = client.post(
                f"{self._vllm_url}/v1/chat/completions", json=payload
            )
        r.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000

        message = r.json()["choices"][0]["message"]
        content = message.get("content") or ""

        # vLLM may surface thinking in reasoning_content (separate field)
        # or inline with <think>...</think> tags in content
        reasoning_content = message.get("reasoning_content")
        if reasoning_content:
            thinking = reasoning_content.strip() or None
            answer = content.strip()
        else:
            thinking, answer = _split_thinking(content)

        return InferenceResult(
            answer=answer,
            latency_ms=latency_ms,
            model_id=_MODEL_ID,
            thinking=thinking,
        )
