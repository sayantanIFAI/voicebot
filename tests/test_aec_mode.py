import numpy as np

from agent.full_duplex import FullDuplexProcessor


def test_default_mode_runs_the_canceller_on_a_phone_line(monkeypatch):
    monkeypatch.delenv("AEC_MODE", raising=False)
    assert FullDuplexProcessor.for_phone_line().skip_cancel is False


def test_phone_light_skips_the_maths_on_a_phone_line_only(monkeypatch):
    monkeypatch.setenv("AEC_MODE", "phone_light")
    assert FullDuplexProcessor.for_phone_line().skip_cancel is True
    assert FullDuplexProcessor().skip_cancel is False  # a browser call keeps the full canceller


def test_phone_light_passes_the_microphone_through_while_the_agent_speaks(monkeypatch):
    monkeypatch.setenv("AEC_MODE", "phone_light")
    fd = FullDuplexProcessor.for_phone_line()
    clip = (np.random.default_rng(0).standard_normal(16000) * 0.1).astype(np.float32)
    fd.place_reference(clip)
    assert fd.agent_speaking()
    mic = (np.random.default_rng(1).standard_normal(512) * 0.05).astype(np.float32)
    res = fd.process(mic)
    assert res.cleaned.shape == mic.shape
    assert np.allclose(res.cleaned, mic)
