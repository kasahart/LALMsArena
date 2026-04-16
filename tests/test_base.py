from __future__ import annotations

import pytest
from pathlib import Path

from models.base import AudioModel, InferenceResult


def test_inference_result_stores_fields():
    result = InferenceResult(answer="hello", latency_ms=123.4, model_id="test/model")
    assert result.answer == "hello"
    assert result.latency_ms == 123.4
    assert result.model_id == "test/model"


def test_audio_model_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        AudioModel()  # type: ignore[abstract]


class _ConcreteModel(AudioModel):
    def load(self) -> None:
        pass

    def run_inference(self, audio_path: Path, question: str, max_new_tokens: int = 512) -> InferenceResult:
        return InferenceResult(answer="ok", latency_ms=1.0, model_id="test/model")

    @property
    def display_name(self) -> str:
        return "Test Model"

    @property
    def model_id(self) -> str:
        return "test/model"


def test_concrete_subclass_satisfies_interface():
    m = _ConcreteModel()
    assert m.display_name == "Test Model"
    assert m.model_id == "test/model"


def test_concrete_subclass_run_inference_returns_result(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    m = _ConcreteModel()
    result = m.run_inference(audio, "What is this?")
    assert isinstance(result, InferenceResult)
    assert result.answer == "ok"
