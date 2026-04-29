from __future__ import annotations

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


# ---------------------------------------------------------------------------
# _split_thinking
# ---------------------------------------------------------------------------

def test_split_thinking_extracts_think_block():
    text = "<think>some reasoning</think> Final answer."
    thinking, answer = _split_thinking(text)
    assert thinking == "some reasoning"
    assert answer == "Final answer."


def test_split_thinking_returns_none_when_no_block():
    text = "Direct answer."
    thinking, answer = _split_thinking(text)
    assert thinking is None
    assert answer == "Direct answer."


def test_split_thinking_multiline():
    text = "<think>\nline1\nline2\n</think>\nAnswer"
    thinking, answer = _split_thinking(text)
    assert "line1" in thinking
    assert answer == "Answer"


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------

def test_load_is_idempotent():
    m = NemotronOmniReasoningModel(device="cpu")

    with patch("models.nemotron_omni.Qwen3OmniMoeForConditionalGeneration") as MockModel, \
         patch("models.nemotron_omni.Qwen3OmniMoeProcessor") as MockProcessor, \
         patch("models.nemotron_omni.torch") as mock_torch:

        mock_torch.cuda.is_available.return_value = False
        mock_torch.float32 = "float32"
        mock_torch.bfloat16 = "bfloat16"
        MockModel.from_pretrained.return_value = MagicMock()
        MockProcessor.from_pretrained.return_value = MagicMock()

        m.load()
        m.load()

        assert MockModel.from_pretrained.call_count == 1


def test_load_falls_back_to_cpu_when_cuda_unavailable():
    m = NemotronOmniReasoningModel(device="cuda")

    with patch("models.nemotron_omni.Qwen3OmniMoeForConditionalGeneration") as MockModel, \
         patch("models.nemotron_omni.Qwen3OmniMoeProcessor") as MockProcessor, \
         patch("models.nemotron_omni.torch") as mock_torch:

        mock_torch.cuda.is_available.return_value = False
        mock_torch.float32 = "float32"
        mock_torch.bfloat16 = "bfloat16"
        MockModel.from_pretrained.return_value = MagicMock()
        MockProcessor.from_pretrained.return_value = MagicMock()

        m.load()

        assert m._device == "cpu"


# ---------------------------------------------------------------------------
# run_inference guards
# ---------------------------------------------------------------------------

def test_run_inference_raises_on_unsupported_extension(tmp_path):
    audio_file = tmp_path / "audio.xyz"
    audio_file.write_bytes(b"fake")

    m = NemotronOmniReasoningModel()
    m._model = MagicMock()
    m._processor = MagicMock()

    with pytest.raises(ValueError, match="Unsupported audio format"):
        m.run_inference(audio_file, "What do you hear?")


def test_run_inference_raises_if_not_loaded(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake")

    m = NemotronOmniReasoningModel()
    with pytest.raises(RuntimeError, match="not loaded"):
        m.run_inference(audio_file, "What do you hear?")


# ---------------------------------------------------------------------------
# run_inference happy path (mocked)
# ---------------------------------------------------------------------------

def _make_loaded_model(raw_output: str) -> NemotronOmniReasoningModel:
    mock_token_ids = MagicMock()
    mock_token_ids.__getitem__ = lambda self, key: MagicMock()

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.dtype = "float32"
    mock_model.generate.return_value = mock_token_ids

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "prompt text"

    mock_inputs = MagicMock()
    mock_input_ids = MagicMock()
    mock_input_ids.shape = [1, 10]
    mock_inputs.__getitem__ = lambda self, key: mock_input_ids
    mock_inputs.to.return_value = mock_inputs

    mock_processor.return_value = mock_inputs
    mock_processor.batch_decode.return_value = [raw_output]

    m = NemotronOmniReasoningModel(device="cpu")
    m._model = mock_model
    m._processor = mock_processor
    m._device = "cpu"
    return m


def test_run_inference_returns_inference_result(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake wav")

    m = _make_loaded_model("Direct answer.")

    with patch("models.nemotron_omni.process_mm_info", return_value=([], None, None)), \
         patch("models.nemotron_omni.torch") as mock_torch:
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=None)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_torch.inference_mode.return_value = mock_ctx

        result = m.run_inference(audio_file, "What do you hear?")

    assert isinstance(result, InferenceResult)
    assert result.answer == "Direct answer."
    assert result.thinking is None
    assert result.model_id == _MODEL_ID
    assert result.latency_ms >= 0


def test_run_inference_splits_think_block(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake wav")

    m = _make_loaded_model("<think>my reasoning</think> The answer.")

    with patch("models.nemotron_omni.process_mm_info", return_value=([], None, None)), \
         patch("models.nemotron_omni.torch") as mock_torch:
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=None)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_torch.inference_mode.return_value = mock_ctx

        result = m.run_inference(audio_file, "What do you hear?")

    assert result.answer == "The answer."
    assert result.thinking == "my reasoning"


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
# GPU end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_api_inference(sample_wav: Path) -> None:
    """Integration test: POST to running arena-nemotron-omni-reasoning container (port 8615)."""
    import httpx

    url = "http://localhost:8615"
    try:
        r = httpx.get(f"{url}/health", timeout=5.0)
        assert r.status_code == 200, "Container not healthy"
    except Exception as exc:
        pytest.skip(f"Container not reachable: {exc}")

    with sample_wav.open("rb") as f:
        r = httpx.post(
            f"{url}/infer",
            files={"audio": ("sample.wav", f, "audio/wav")},
            data={"question": "What sound do you hear?", "max_new_tokens": 512},
            timeout=300.0,
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["answer"], str)
    assert len(body["answer"]) > 0
    assert body["model_id"] == _MODEL_ID
