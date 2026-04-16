from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from models.gemma4_e4b import Gemma4E4BModel
from models.base import InferenceResult


def test_display_name():
    m = Gemma4E4BModel()
    assert m.display_name == "Gemma-4-E4B"


def test_model_id():
    m = Gemma4E4BModel()
    assert m.model_id == "google/gemma-4-E4B-it"


def test_load_is_idempotent():
    m = Gemma4E4BModel(device="cpu")

    with patch("models.gemma4_e4b.AutoModelForMultimodalLM") as MockModel, \
         patch("models.gemma4_e4b.AutoProcessor") as MockProcessor, \
         patch("models.gemma4_e4b.torch") as mock_torch:

        mock_torch.cuda.is_available.return_value = False
        mock_torch.float32 = "float32"
        MockModel.from_pretrained.return_value = MagicMock()
        MockProcessor.from_pretrained.return_value = MagicMock()

        m.load()
        m.load()

        assert MockModel.from_pretrained.call_count == 1


def test_load_falls_back_to_cpu_when_cuda_unavailable():
    m = Gemma4E4BModel(device="cuda")

    with patch("models.gemma4_e4b.AutoModelForMultimodalLM") as MockModel, \
         patch("models.gemma4_e4b.AutoProcessor") as MockProcessor, \
         patch("models.gemma4_e4b.torch") as mock_torch:

        mock_torch.cuda.is_available.return_value = False
        mock_torch.float32 = "float32"
        MockModel.from_pretrained.return_value = MagicMock()
        MockProcessor.from_pretrained.return_value = MagicMock()

        m.load()

        assert m._device == "cpu"


def test_run_inference_raises_on_unsupported_extension(tmp_path):
    audio_file = tmp_path / "audio.xyz"
    audio_file.write_bytes(b"fake")

    m = Gemma4E4BModel()
    m._model = MagicMock()
    m._processor = MagicMock()
    m._device = "cpu"

    with pytest.raises(ValueError, match="Unsupported audio format"):
        m.run_inference(audio_file, "What do you hear?")


def test_run_inference_raises_if_not_loaded(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake")

    m = Gemma4E4BModel()

    with pytest.raises(RuntimeError, match="not loaded"):
        m.run_inference(audio_file, "What do you hear?")


def test_run_inference_returns_inference_result(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake wav")

    mock_input_ids = MagicMock()
    mock_input_ids.shape = [1, 10]
    mock_inputs = {"input_ids": mock_input_ids}

    mock_apply_result = MagicMock()
    mock_apply_result.to.return_value = mock_inputs

    mock_generated = MagicMock()
    mock_token_seq = MagicMock()
    mock_generated.__getitem__ = lambda self, key: mock_token_seq

    mock_model = MagicMock()
    mock_model.generate.return_value = mock_generated

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = mock_apply_result
    mock_processor.decode.return_value = "This is the answer"

    m = Gemma4E4BModel()
    m._model = mock_model
    m._processor = mock_processor
    m._device = "cpu"

    with patch("models.gemma4_e4b.torch") as mock_torch:
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=None)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_torch.inference_mode.return_value = mock_ctx

        result = m.run_inference(audio_file, "What do you hear?")

    assert isinstance(result, InferenceResult)
    assert result.answer == "This is the answer"
    assert result.model_id == "google/gemma-4-E4B-it"
    assert result.latency_ms >= 0


def test_run_inference_uses_audio_key_not_path(tmp_path):
    """音声入力は {"type": "audio", "audio": path} 形式（"path" キーではない）"""
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake wav")

    mock_input_ids = MagicMock()
    mock_input_ids.shape = [1, 10]
    mock_inputs = {"input_ids": mock_input_ids}
    mock_apply_result = MagicMock()
    mock_apply_result.to.return_value = mock_inputs

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = mock_apply_result
    mock_processor.decode.return_value = "answer"

    m = Gemma4E4BModel()
    m._model = MagicMock()
    m._model.generate.return_value = MagicMock()
    m._processor = mock_processor
    m._device = "cpu"

    with patch("models.gemma4_e4b.torch") as mock_torch:
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=None)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_torch.inference_mode.return_value = mock_ctx
        m.run_inference(audio_file, "What?")

    call_args = mock_processor.apply_chat_template.call_args
    messages = call_args[0][0]
    audio_content = messages[0]["content"][0]
    assert audio_content["type"] == "audio"
    assert "audio" in audio_content
    assert "path" not in audio_content
