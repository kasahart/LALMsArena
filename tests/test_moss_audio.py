from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from models.moss_audio import MossAudio4BModel, MossAudio8BModel, _MossAudioBase
from models.base import InferenceResult


# --- identity ---

def test_4b_display_name():
    assert MossAudio4BModel().display_name == "MOSS-Audio-4B"


def test_4b_model_id():
    assert MossAudio4BModel().model_id == "OpenMOSS-Team/MOSS-Audio-4B-Instruct"


def test_8b_display_name():
    assert MossAudio8BModel().display_name == "MOSS-Audio-8B"


def test_8b_model_id():
    assert MossAudio8BModel().model_id == "OpenMOSS-Team/MOSS-Audio-8B-Instruct"


# --- load ---

def _make_mock_model_cls():
    mock_instance = MagicMock()
    mock_instance.dtype = "bfloat16"
    mock_cls = MagicMock()
    mock_cls.from_pretrained.return_value = mock_instance
    return mock_cls, mock_instance


def test_load_is_idempotent():
    m = MossAudio4BModel(device="cpu")
    mock_cls, _ = _make_mock_model_cls()

    mock_proc_cls = MagicMock()
    mock_proc_cls.from_pretrained.return_value = MagicMock()

    with patch("models.moss_audio.torch") as mock_torch, \
         patch("models.moss_audio._load_moss_classes", return_value=(mock_cls, mock_proc_cls)):

        mock_torch.cuda.is_available.return_value = False
        mock_torch.float32 = "float32"

        m.load()
        m.load()

        assert mock_cls.from_pretrained.call_count == 1


def test_load_falls_back_to_cpu():
    m = MossAudio4BModel(device="cuda")
    mock_cls, _ = _make_mock_model_cls()

    mock_proc_cls = MagicMock()
    mock_proc_cls.from_pretrained.return_value = MagicMock()

    with patch("models.moss_audio.torch") as mock_torch, \
         patch("models.moss_audio._load_moss_classes", return_value=(mock_cls, mock_proc_cls)):

        mock_torch.cuda.is_available.return_value = False
        mock_torch.float32 = "float32"

        m.load()
        assert m._device == "cpu"


# --- run_inference guards ---

def test_run_inference_raises_on_unsupported_extension(tmp_path):
    audio_file = tmp_path / "audio.xyz"
    audio_file.write_bytes(b"fake")

    m = MossAudio4BModel()
    m._model = MagicMock()
    m._processor = MagicMock()
    m._device = "cpu"

    with pytest.raises(ValueError, match="Unsupported audio format"):
        m.run_inference(audio_file, "What do you hear?")


def test_run_inference_raises_if_not_loaded(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake")

    m = MossAudio4BModel()
    with pytest.raises(RuntimeError, match="not loaded"):
        m.run_inference(audio_file, "What do you hear?")


# --- run_inference happy path ---

def _run_inference_with_mocks(tmp_path, model_variant):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake wav")

    mock_model = MagicMock()
    mock_model.dtype = "bfloat16"
    mock_model.generate.return_value = MagicMock()

    mock_input_ids = MagicMock()
    mock_input_ids.shape = [1, 8]
    mock_inputs = {"input_ids": mock_input_ids, "audio_data": MagicMock()}
    captured = {}

    mock_inputs_obj = MagicMock()
    mock_inputs_obj.get = mock_inputs.get
    mock_inputs_obj.__getitem__ = lambda s, k: mock_inputs[k]
    mock_inputs_obj.__setitem__ = lambda s, k, v: captured.update({k: v})
    mock_inputs_obj.to.return_value = mock_inputs_obj

    mock_processor = MagicMock()
    mock_processor.return_value = mock_inputs_obj
    mock_processor.audio_token_id = 999
    mock_processor.decode.return_value = "The answer"

    m = model_variant(device="cpu")
    m._model = mock_model
    m._processor = mock_processor
    m._device = "cpu"

    with patch("models.moss_audio._load_audio", return_value=MagicMock()), \
         patch("models.moss_audio.torch") as mock_torch:

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=None)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_torch.inference_mode.return_value = mock_ctx

        result = m.run_inference(audio_file, "What do you hear?")

    return result


def test_run_inference_returns_result_4b(tmp_path):
    result = _run_inference_with_mocks(tmp_path, MossAudio4BModel)
    assert isinstance(result, InferenceResult)
    assert result.answer == "The answer"
    assert result.model_id == "OpenMOSS-Team/MOSS-Audio-4B-Instruct"
    assert result.latency_ms >= 0


def test_run_inference_returns_result_8b(tmp_path):
    result = _run_inference_with_mocks(tmp_path, MossAudio8BModel)
    assert isinstance(result, InferenceResult)
    assert result.model_id == "OpenMOSS-Team/MOSS-Audio-8B-Instruct"


def test_run_inference_builds_audio_input_mask(tmp_path):
    """audio_input_mask は input_ids == audio_token_id で構築される。"""
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake wav")

    mock_model = MagicMock()
    mock_model.dtype = "bfloat16"

    mock_input_ids = MagicMock()
    mock_input_ids.shape = [1, 8]
    captured = {}

    mock_inputs_obj = MagicMock()
    mock_inputs_obj.get.return_value = None
    mock_inputs_obj.__getitem__ = lambda s, k: mock_input_ids if k == "input_ids" else MagicMock()
    mock_inputs_obj.__setitem__ = lambda s, k, v: captured.update({k: v})
    mock_inputs_obj.to.return_value = mock_inputs_obj

    mock_processor = MagicMock()
    mock_processor.return_value = mock_inputs_obj
    mock_processor.audio_token_id = 42
    mock_processor.decode.return_value = "answer"

    m = MossAudio4BModel(device="cpu")
    m._model = mock_model
    m._processor = mock_processor
    m._device = "cpu"

    with patch("models.moss_audio._load_audio", return_value=MagicMock()), \
         patch("models.moss_audio.torch") as mock_torch:

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=None)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_torch.inference_mode.return_value = mock_ctx
        mock_model.generate.return_value = MagicMock()

        m.run_inference(audio_file, "What?")

    assert "audio_input_mask" in captured


@pytest.mark.gpu
def test_inference_end_to_end_4b(sample_wav: Path) -> None:
    """Full load + run_inference with MOSS-Audio-4B on GPU."""
    m = MossAudio4BModel(device="cuda")
    m.load()
    result = m.run_inference(sample_wav, "What sound do you hear?")
    assert isinstance(result, InferenceResult)
    assert isinstance(result.answer, str)
    assert len(result.answer) > 0
    assert result.latency_ms > 0
    assert result.model_id == "OpenMOSS-Team/MOSS-Audio-4B-Instruct"


@pytest.mark.gpu
def test_inference_end_to_end_8b(sample_wav: Path) -> None:
    """Full load + run_inference with MOSS-Audio-8B on GPU."""
    m = MossAudio8BModel(device="cuda")
    m.load()
    result = m.run_inference(sample_wav, "What sound do you hear?")
    assert isinstance(result, InferenceResult)
    assert isinstance(result.answer, str)
    assert len(result.answer) > 0
    assert result.latency_ms > 0
    assert result.model_id == "OpenMOSS-Team/MOSS-Audio-8B-Instruct"
