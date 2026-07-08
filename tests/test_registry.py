from __future__ import annotations

import pytest

from models import get_model, list_models
from models.audio_flamingo import AudioFlamingoModel
from models.gemma4_12b import Gemma412BModel


def test_list_models_contains_audio_flamingo():
    assert "Audio Flamingo" in list_models()


def test_get_model_returns_audio_flamingo_instance():
    m = get_model("Audio Flamingo", device="cpu")
    assert isinstance(m, AudioFlamingoModel)


def test_get_model_passes_device_to_instance():
    m = get_model("Audio Flamingo", device="cpu")
    assert m._requested_device == "cpu"


def test_get_model_returns_gemma_12b_instance():
    m = get_model("Gemma-4-12B", device="cpu")
    assert isinstance(m, Gemma412BModel)


def test_get_model_raises_on_unknown_model():
    with pytest.raises(KeyError, match="not found"):
        get_model("Unknown Model")
