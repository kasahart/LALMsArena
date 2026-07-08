from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from models.base import InferenceResult
from models.nemotron_audex import (
    NemotronLabsAudexModel,
    _AUDIO_PLACEHOLDER_TOKEN_IDS,
    _MODEL_ID,
    _SERVED_MODEL_NAME,
    _split_thinking,
)


def test_display_name():
    assert NemotronLabsAudexModel().display_name == "Nemotron-Labs-Audex-30B-A3B"


def test_model_id():
    assert NemotronLabsAudexModel().model_id == _MODEL_ID


def test_vllm_url_default():
    m = NemotronLabsAudexModel()
    assert m._vllm_url == "http://audex-vllm:9999"


def test_vllm_url_from_env(monkeypatch):
    monkeypatch.setenv("AUDEX_VLLM_URL", "http://custom-host:1234")
    m = NemotronLabsAudexModel()
    assert m._vllm_url == "http://custom-host:1234"


def test_split_thinking_full_tags():
    thinking, answer = _split_thinking("<think>reasoning</think> Answer.")
    assert thinking == "reasoning"
    assert answer == "Answer."


def test_split_thinking_close_tag_only():
    thinking, answer = _split_thinking("reasoning\n</think>\nAnswer.")
    assert thinking == "reasoning"
    assert answer == "Answer."


def test_split_thinking_no_block():
    thinking, answer = _split_thinking("Answer only.")
    assert thinking is None
    assert answer == "Answer only."


def test_load_succeeds_when_vllm_ready():
    m = NemotronLabsAudexModel()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("models.nemotron_audex.httpx.get", return_value=mock_resp), \
         patch("models.nemotron_audex.time.sleep"):
        m.load()


def test_load_raises_when_vllm_never_ready():
    m = NemotronLabsAudexModel()
    with patch("models.nemotron_audex.httpx.get", side_effect=Exception("refused")), \
         patch("models.nemotron_audex.time.sleep"), \
         patch("models.nemotron_audex.time.time", side_effect=[0, 1201]):
        with pytest.raises(RuntimeError, match="did not become ready"):
            m.load()


def test_run_inference_raises_on_unsupported_extension(tmp_path):
    audio_file = tmp_path / "audio.xyz"
    audio_file.write_bytes(b"fake")

    m = NemotronLabsAudexModel()
    with pytest.raises(ValueError, match="Unsupported audio format"):
        m.run_inference(audio_file, "What do you hear?")


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

    with patch("models.nemotron_audex.httpx.Client", return_value=mock_client):
        result = NemotronLabsAudexModel().run_inference(
            audio_file,
            "What do you hear?",
        )

    assert isinstance(result, InferenceResult)
    assert result.answer == "The answer."
    assert result.thinking is None
    assert result.model_id == _MODEL_ID
    assert result.latency_ms >= 0


def test_run_inference_sends_audio_and_recommended_sampling(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"RIFF fake wav")
    mock_client = _make_httpx_mock("ok")

    with patch("models.nemotron_audex.httpx.Client", return_value=mock_client):
        NemotronLabsAudexModel().run_inference(audio_file, "question")

    payload = mock_client.post.call_args.kwargs["json"]
    assert payload["model"] == _SERVED_MODEL_NAME
    audio_content = payload["messages"][0]["content"][0]
    assert audio_content["type"] == "audio_url"
    assert audio_content["audio_url"]["url"].startswith("data:audio/wav;base64,")
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.9
    assert payload["extra_body"]["top_k"] == 0
    assert payload["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    allowed_token_ids = payload["extra_body"]["allowed_token_ids"]
    for token_id in _AUDIO_PLACEHOLDER_TOKEN_IDS:
        assert token_id not in allowed_token_ids


def test_run_inference_uses_reasoning_content_field(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"RIFF fake wav")
    mock_client = _make_httpx_mock(
        content="Clean answer.",
        reasoning_content="Separate thinking.",
    )

    with patch("models.nemotron_audex.httpx.Client", return_value=mock_client):
        result = NemotronLabsAudexModel().run_inference(audio_file, "question")

    assert result.answer == "Clean answer."
    assert result.thinking == "Separate thinking."


def test_registry_contains_audex():
    from models import list_models

    assert "Nemotron-Labs-Audex-30B-A3B" in list_models()


def test_get_model_returns_audex_instance():
    from models import get_model

    m = get_model("Nemotron-Labs-Audex-30B-A3B", device="cpu")
    assert isinstance(m, NemotronLabsAudexModel)


@pytest.mark.gpu
def test_api_inference(sample_wav: Path) -> None:
    import httpx as _httpx

    url = "http://localhost:8618"
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
