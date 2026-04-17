from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from models.salmonn_13b import SALMONNModel
from models.base import InferenceResult


def test_display_name():
    m = SALMONNModel()
    assert m.display_name == "SALMONN-13B"


def test_model_id():
    m = SALMONNModel()
    assert m.model_id == "tsinghua-ee/SALMONN"


def test_load_is_idempotent(monkeypatch):
    monkeypatch.setenv("SALMONN_BEATS_PATH", "/fake/beats.pt")

    mock_salmonn_cls = MagicMock()
    mock_instance = MagicMock()
    mock_salmonn_cls.from_config.return_value = mock_instance

    m = SALMONNModel(device="cpu")

    with patch("models.salmonn_13b.torch") as mock_torch, \
         patch("models.salmonn_13b._load_salmonn_class", return_value=mock_salmonn_cls), \
         patch("models.salmonn_13b.hf_hub_download", return_value="/fake/salmonn_v1.pth"), \
         patch("models.salmonn_13b.WhisperFeatureExtractor") as MockWP:

        mock_torch.cuda.is_available.return_value = False
        MockWP.from_pretrained.return_value = MagicMock()

        m.load()
        m.load()

        assert mock_salmonn_cls.from_config.call_count == 1


def test_load_falls_back_to_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setenv("SALMONN_BEATS_PATH", "/fake/beats.pt")

    mock_salmonn_cls = MagicMock()
    mock_salmonn_cls.from_config.return_value = MagicMock()

    m = SALMONNModel(device="cuda")

    with patch("models.salmonn_13b.torch") as mock_torch, \
         patch("models.salmonn_13b._load_salmonn_class", return_value=mock_salmonn_cls), \
         patch("models.salmonn_13b.hf_hub_download", return_value="/fake/salmonn_v1.pth"), \
         patch("models.salmonn_13b.WhisperFeatureExtractor") as MockWP:

        mock_torch.cuda.is_available.return_value = False
        MockWP.from_pretrained.return_value = MagicMock()

        m.load()

        assert m._device == "cpu"


def test_load_raises_if_beats_path_not_set(monkeypatch):
    monkeypatch.delenv("SALMONN_BEATS_PATH", raising=False)

    m = SALMONNModel(device="cpu")

    with patch("models.salmonn_13b.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        with pytest.raises(RuntimeError, match="SALMONN_BEATS_PATH"):
            m.load()


def test_run_inference_raises_on_unsupported_extension(tmp_path):
    audio_file = tmp_path / "audio.xyz"
    audio_file.write_bytes(b"fake")

    m = SALMONNModel()
    m._model = MagicMock()
    m._wav_processor = MagicMock()
    m._device = "cpu"

    with pytest.raises(ValueError, match="Unsupported audio format"):
        m.run_inference(audio_file, "What do you hear?")


def test_run_inference_raises_if_not_loaded(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake")

    m = SALMONNModel()

    with pytest.raises(RuntimeError, match="not loaded"):
        m.run_inference(audio_file, "What do you hear?")


def test_run_inference_returns_inference_result(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake wav")

    mock_model = MagicMock()
    mock_model.generate.return_value = ["This is the answer"]

    mock_wav_proc = MagicMock()
    mock_wav_proc.return_value = {"input_features": MagicMock()}

    m = SALMONNModel()
    m._model = mock_model
    m._wav_processor = mock_wav_proc
    m._device = "cpu"

    fake_audio = MagicMock()
    fake_audio.ndim = 1
    fake_audio.__len__ = lambda self: 16000

    with patch("models.salmonn_13b.sf") as mock_sf, \
         patch("models.salmonn_13b.torch") as mock_torch, \
         patch("models.salmonn_13b.np") as mock_np:

        mock_sf.read.return_value = (fake_audio, 16000)
        mock_np.concatenate = MagicMock(return_value=fake_audio)
        mock_np.zeros = MagicMock(return_value=MagicMock())

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=None)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_torch.inference_mode.return_value = mock_ctx
        mock_torch.from_numpy.return_value = MagicMock()
        mock_torch.zeros.return_value = MagicMock()

        result = m.run_inference(audio_file, "What do you hear?")

    assert isinstance(result, InferenceResult)
    assert result.answer == "This is the answer"
    assert result.model_id == "tsinghua-ee/SALMONN"
    assert result.latency_ms >= 0


def test_run_inference_passes_question_as_prompt(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake wav")

    mock_model = MagicMock()
    mock_model.generate.return_value = ["answer"]

    mock_wav_proc = MagicMock()
    mock_wav_proc.return_value = {"input_features": MagicMock()}

    m = SALMONNModel()
    m._model = mock_model
    m._wav_processor = mock_wav_proc
    m._device = "cpu"

    fake_audio = MagicMock()
    fake_audio.ndim = 1
    fake_audio.__len__ = lambda self: 16000

    with patch("models.salmonn_13b.sf") as mock_sf, \
         patch("models.salmonn_13b.torch") as mock_torch, \
         patch("models.salmonn_13b.np") as mock_np:

        mock_sf.read.return_value = (fake_audio, 16000)
        mock_np.concatenate = MagicMock(return_value=fake_audio)
        mock_np.zeros = MagicMock(return_value=MagicMock())

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=None)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_torch.inference_mode.return_value = mock_ctx
        mock_torch.from_numpy.return_value = MagicMock()
        mock_torch.zeros.return_value = MagicMock()

        m.run_inference(audio_file, "Describe the sound")

    call_kwargs = mock_model.generate.call_args
    assert call_kwargs.kwargs.get("prompts") == ["Describe the sound"]


@pytest.mark.gpu
def test_inference_end_to_end(sample_wav: Path) -> None:
    """Full load + run_inference with SALMONN-13B on GPU.

    Requires SALMONN_BEATS_PATH env var pointing to
    BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt.
    """
    beats_path = os.environ.get("SALMONN_BEATS_PATH", "")
    if not beats_path:
        pytest.skip("SALMONN_BEATS_PATH not set")

    m = SALMONNModel(device="cuda")
    m.load()
    result = m.run_inference(sample_wav, "What sound do you hear?")
    assert isinstance(result, InferenceResult)
    assert isinstance(result.answer, str)
    assert len(result.answer) > 0
    assert result.latency_ms > 0
    assert result.model_id == "tsinghua-ee/SALMONN"
