from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from models.audio_flamingo import AudioFlamingoModel
from models.base import InferenceResult


def test_display_name():
    m = AudioFlamingoModel()
    assert m.display_name == "Audio Flamingo"


def test_model_id():
    m = AudioFlamingoModel()
    assert m.model_id == "nvidia/audio-flamingo-next-hf"


def test_load_is_idempotent():
    m = AudioFlamingoModel(device="cpu")

    with patch("models.audio_flamingo.AutoModel") as MockModel, \
         patch("models.audio_flamingo.AutoProcessor") as MockProcessor, \
         patch("models.audio_flamingo.torch") as mock_torch:

        mock_torch.cuda.is_available.return_value = False
        mock_torch.float32 = "float32"
        MockModel.from_pretrained.return_value = MagicMock()
        MockProcessor.from_pretrained.return_value = MagicMock()

        m.load()
        m.load()  # 2回目は早期リターンするため from_pretrained は1回だけ

        assert MockModel.from_pretrained.call_count == 1


def test_load_falls_back_to_cpu_when_cuda_unavailable():
    m = AudioFlamingoModel(device="cuda")

    with patch("models.audio_flamingo.AutoModel") as MockModel, \
         patch("models.audio_flamingo.AutoProcessor") as MockProcessor, \
         patch("models.audio_flamingo.torch") as mock_torch:

        mock_torch.cuda.is_available.return_value = False
        mock_torch.float32 = "float32"
        MockModel.from_pretrained.return_value = MagicMock()
        MockProcessor.from_pretrained.return_value = MagicMock()

        m.load()

        assert m._device == "cpu"


def test_load_raises_runtime_error_on_unsupported_transformers():
    m = AudioFlamingoModel(device="cpu")

    with patch("models.audio_flamingo.AutoModel") as MockModel, \
         patch("models.audio_flamingo.AutoProcessor") as MockProcessor, \
         patch("models.audio_flamingo.torch") as mock_torch:

        mock_torch.cuda.is_available.return_value = False
        mock_torch.float32 = "float32"
        MockProcessor.from_pretrained.return_value = MagicMock()
        MockModel.from_pretrained.side_effect = ValueError("audioflamingonext not supported")

        with pytest.raises(RuntimeError, match="transformers build"):
            m.load()


def test_run_inference_raises_on_unsupported_extension(tmp_path):
    audio_file = tmp_path / "audio.xyz"
    audio_file.write_bytes(b"fake")

    m = AudioFlamingoModel()
    m._model = MagicMock()
    m._processor = MagicMock()
    m._device = "cpu"

    with pytest.raises(ValueError, match="Unsupported audio format"):
        m.run_inference(audio_file, "What do you hear?")


def test_run_inference_returns_inference_result(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake wav")

    mock_input_ids = MagicMock()
    mock_input_ids.shape = (1, 10)
    mock_batch = {"input_ids": mock_input_ids}

    mock_apply_result = MagicMock()
    mock_apply_result.to.return_value = mock_batch

    mock_generated = MagicMock()

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.dtype = MagicMock()
    mock_model.generate.return_value = mock_generated

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = mock_apply_result
    mock_processor.batch_decode.return_value = ["This is the answer"]

    m = AudioFlamingoModel()
    m._model = mock_model
    m._processor = mock_processor
    m._device = "cpu"

    with patch("models.audio_flamingo.torch") as mock_torch:
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=None)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_torch.inference_mode.return_value = mock_ctx

        result = m.run_inference(audio_file, "What do you hear?")

    assert isinstance(result, InferenceResult)
    assert result.answer == "This is the answer"
    assert result.model_id == "nvidia/audio-flamingo-next-hf"
    assert result.latency_ms >= 0


@pytest.mark.gpu
def test_inference_end_to_end(sample_wav: Path) -> None:
    """Full load + run_inference with a real model on GPU."""
    m = AudioFlamingoModel(device="cuda")
    m.load()
    result = m.run_inference(sample_wav, "What sound do you hear?")
    assert isinstance(result, InferenceResult)
    assert isinstance(result.answer, str)
    assert len(result.answer) > 0
    assert result.latency_ms > 0
    assert result.model_id == "nvidia/audio-flamingo-next-hf"
