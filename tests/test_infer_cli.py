import numpy as np
import pytest
import soundfile as sf

from duplex import infer_cli
from duplex.io import read_json
from duplex.model import PIN


def test_input_is_mono_24k_with_response_time(tmp_path):
    source = tmp_path / "stereo.wav"
    sf.write(source, np.full((16000, 2), 0.02, dtype=np.float32), 16000)
    report = infer_cli.prepare_input(source, tmp_path / "out", 2)
    audio, sr = sf.read(tmp_path / "out/input/user.wav")
    assert sr == 24000 and audio.ndim == 1
    assert len(audio) % 1920 == 0 and len(audio) >= 72000
    assert np.all(audio[24000:] == 0)
    assert report["original_input_s"] == 1


def test_inference_rejects_existing_output_before_loading(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    (out / "generated.wav").touch()
    monkeypatch.setattr(
        infer_cli, "load_real", lambda _: pytest.fail("Should reject before GPU loading")
    )
    with pytest.raises(ValueError, match="nonempty"):
        infer_cli.infer_file(tmp_path, None, tmp_path / "input.wav", out)


def test_adapter_inference_uses_audio_only_and_exact_prompt(tmp_path, monkeypatch):
    source = tmp_path / "input.wav"
    sf.write(source, np.zeros(24000, dtype=np.float32), 24000)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter.safetensors").write_bytes(b"mock-only-not-real-weights")
    metadata = {
        "real_checkpoint": True,
        "identity": {"base_revision": PIN, "system_prompt": "Recorded training prompt"},
        "train_config": {},
    }
    monkeypatch.setattr(infer_cli, "adapter_metadata", lambda _: metadata)
    monkeypatch.setattr(infer_cli, "load_real", lambda _: (object(), object(), object()))
    monkeypatch.setattr(infer_cli, "attach", lambda *args: None)
    monkeypatch.setattr(infer_cli, "load_adapter", lambda *args: None)

    def fake_generate(lm, mimi, tok, visible, root, output, **kwargs):
        assert visible.user_audio == "input/user.wav" and visible.events == []
        assert kwargs["system_prompt"] == "Recorded training prompt"
        assert kwargs["adapter_sha256"]
        assert not hasattr(visible, "supervision") and not hasattr(visible, "truth")
        return {"internal_text_proxy": "mock output", "generation_s": 0.1}

    monkeypatch.setattr(infer_cli, "generate", fake_generate)
    monkeypatch.delenv("DUPLEX_MODAL", raising=False)
    result = infer_cli.infer_file(tmp_path, adapter, source, tmp_path / "output", response_s=2)
    assert result["adapter_sha256"]
    saved = read_json(tmp_path / "output/inference.json")
    assert saved["teacher_used"] is False and saved["response_tail_s"] == 2


def test_raw_adapter_requires_config_directory(tmp_path):
    path = tmp_path / "adapter.safetensors"
    path.touch()
    with pytest.raises(ValueError, match="directory"):
        infer_cli.resolve_adapter(path)
