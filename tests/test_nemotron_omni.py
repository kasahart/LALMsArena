from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from models.base import InferenceResult
from models.nemotron_omni import (
    NemotronOmniReasoningModel,
    _MODEL_ID,
    _split_thinking,
)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_display_name():
    assert NemotronOmniReasoningModel().display_name == "Nemotron-Nano-Omni-Reasoning"


def test_model_id():
    assert NemotronOmniReasoningModel().model_id == _MODEL_ID


def test_vllm_url_default():
    m = NemotronOmniReasoningModel()
    assert m._vllm_url == "http://nemotron-vllm:9999"


def test_vllm_url_from_env(monkeypatch):
    monkeypatch.setenv("NEMOTRON_VLLM_URL", "http://custom-host:1234")
    m = NemotronOmniReasoningModel()
    assert m._vllm_url == "http://custom-host:1234"


# ---------------------------------------------------------------------------
# _split_thinking
# ---------------------------------------------------------------------------

def test_split_thinking_full_tags():
    text = "<think>some reasoning</think> Final answer."
    thinking, answer = _split_thinking(text)
    assert thinking == "some reasoning"
    assert answer == "Final answer."


def test_split_thinking_no_block():
    text = "Direct answer."
    thinking, answer = _split_thinking(text)
    assert thinking is None
    assert answer == "Direct answer."


def test_split_thinking_multiline():
    text = "<think>\nline1\nline2\n</think>\nAnswer"
    thinking, answer = _split_thinking(text)
    assert "line1" in thinking
    assert answer == "Answer"


def test_split_thinking_close_tag_only():
    """vLLM strips the opening <think> tag; only </think> remains."""
    text = "reasoning content here\n</think>\nFinal answer."
    thinking, answer = _split_thinking(text)
    assert thinking == "reasoning content here"
    assert answer == "Final answer."


def test_split_thinking_empty_thinking_returns_none():
    text = "\n</think>\nAnswer only."
    thinking, answer = _split_thinking(text)
    assert thinking is None
    assert answer == "Answer only."


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------

def _mock_httpx_get(status_code: int):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    return mock_resp


def test_load_succeeds_when_vllm_ready():
    m = NemotronOmniReasoningModel()
    with patch("models.nemotron_omni.httpx.get", return_value=_mock_httpx_get(200)), \
         patch("models.nemotron_omni.time.sleep"):
        m.load()  # should not raise


def test_load_raises_when_vllm_never_ready():
    m = NemotronOmniReasoningModel()
    with patch("models.nemotron_omni.httpx.get", side_effect=Exception("refused")), \
         patch("models.nemotron_omni.time.sleep"), \
         patch("models.nemotron_omni.time.time", side_effect=[0, 1201]):
        with pytest.raises(RuntimeError, match="did not become ready"):
            m.load()


# ---------------------------------------------------------------------------
# run_inference guards
# ---------------------------------------------------------------------------

def test_run_inference_raises_on_unsupported_extension(tmp_path):
    audio_file = tmp_path / "audio.xyz"
    audio_file.write_bytes(b"fake")

    m = NemotronOmniReasoningModel()
    with pytest.raises(ValueError, match="Unsupported audio format"):
        m.run_inference(audio_file, "What do you hear?")


# ---------------------------------------------------------------------------
# run_inference happy path (mocked httpx)
# ---------------------------------------------------------------------------

def _make_httpx_mock(content: str, reasoning_content: str | None = None):
    message: dict = {"content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content

    mock_response = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": message}]}
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_response
    return mock_client


def test_run_inference_returns_result(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"RIFF fake wav")

    mock_client = _make_httpx_mock("The answer.")

    with patch("models.nemotron_omni.httpx.Client", return_value=mock_client):
        m = NemotronOmniReasoningModel()
        result = m.run_inference(audio_file, "What do you hear?")

    assert isinstance(result, InferenceResult)
    assert result.answer == "The answer."
    assert result.thinking is None
    assert result.model_id == _MODEL_ID
    assert result.latency_ms >= 0


def test_run_inference_sends_base64_audio(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"RIFF fake wav")

    mock_client = _make_httpx_mock("ok")

    with patch("models.nemotron_omni.httpx.Client", return_value=mock_client):
        m = NemotronOmniReasoningModel()
        m.run_inference(audio_file, "question")

    call_kwargs = mock_client.post.call_args
    payload = call_kwargs[1]["json"] if call_kwargs[1] else call_kwargs[0][1]
    audio_content = payload["messages"][0]["content"][0]
    assert audio_content["type"] == "audio_url"
    assert audio_content["audio_url"]["url"].startswith("data:audio/wav;base64,")


def test_run_inference_splits_think_tags(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"RIFF fake wav")

    mock_client = _make_httpx_mock("<think>my reasoning</think> The answer.")

    with patch("models.nemotron_omni.httpx.Client", return_value=mock_client):
        m = NemotronOmniReasoningModel()
        result = m.run_inference(audio_file, "What do you hear?")

    assert result.answer == "The answer."
    assert result.thinking == "my reasoning"


def test_run_inference_splits_close_tag_only(tmp_path):
    """Pattern 2: vLLM strips opening <think> tag."""
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"RIFF fake wav")

    mock_client = _make_httpx_mock("reasoning here\n</think>\nFinal answer.")

    with patch("models.nemotron_omni.httpx.Client", return_value=mock_client):
        m = NemotronOmniReasoningModel()
        result = m.run_inference(audio_file, "What do you hear?")

    assert result.answer == "Final answer."
    assert result.thinking == "reasoning here"


def test_run_inference_uses_reasoning_content_field(tmp_path):
    """vLLM reasoning_content field takes precedence over tag parsing."""
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"RIFF fake wav")

    mock_client = _make_httpx_mock(
        content="Clean answer.",
        reasoning_content="Separate thinking.",
    )

    with patch("models.nemotron_omni.httpx.Client", return_value=mock_client):
        m = NemotronOmniReasoningModel()
        result = m.run_inference(audio_file, "What do you hear?")

    assert result.answer == "Clean answer."
    assert result.thinking == "Separate thinking."


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_contains_nemotron():
    from models import list_models
    assert "Nemotron-Nano-Omni-Reasoning" in list_models()


def test_get_model_returns_nemotron_instance():
    from models import get_model
    m = get_model("Nemotron-Nano-Omni-Reasoning", device="cpu")
    assert isinstance(m, NemotronOmniReasoningModel)


# ---------------------------------------------------------------------------
# GPU end-to-end (requires running container on port 8615)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_api_inference(sample_wav: Path) -> None:
    import httpx as _httpx

    url = "http://localhost:8615"
    try:
        r = _httpx.get(f"{url}/health", timeout=5.0)
        assert r.status_code == 200, "Container not healthy"
    except Exception as exc:
        pytest.skip(f"Container not reachable: {exc}")

    with sample_wav.open("rb") as f:
        r = _httpx.post(
            f"{url}/infer",
            files={"audio": ("sample.wav", f, "audio/wav")},
            data={"question": "What sound do you hear?", "max_new_tokens": "512"},
            timeout=300.0,
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["answer"], str)
    assert len(body["answer"]) > 0
    assert body["model_id"] == _MODEL_ID
    assert body["latency_ms"] > 0
