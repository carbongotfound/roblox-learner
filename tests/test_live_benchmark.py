from argparse import Namespace
import json
import pytest
from PIL import Image
from roblox_learner.benchmark import benchmark_live_capture
from roblox_learner.desktop import DesktopError


def test_live_capture_benchmark_never_sends_predicted_controls(monkeypatch, tmp_path):
    import roblox_learner.desktop as desktop
    import roblox_learner.play as play
    class Backend:
        def escape_pressed(self): return False
        def capture(self): return Image.new("RGB", (320, 180))
        def diagnostics(self): return {"capture_backend": "test"}
        def key(self, *args): pytest.fail("Shadow benchmark injected a key")
        def mouse_button(self, *args): pytest.fail("Shadow benchmark injected a click")
        def mouse_move(self, *args): pytest.fail("Shadow benchmark moved the mouse")
    class Policy:
        metadata = {"epoch": 0}
        def __init__(self, *args): pass
        def predict(self, image): return play.Prediction(1, 1, 0)
        def accounted_memory_mb(self): return 200
    monkeypatch.setattr(desktop, "MacDesktop", Backend)
    monkeypatch.setattr(play, "TorchPolicy", Policy)
    monkeypatch.setattr(desktop, "countdown", lambda _: None)
    report = benchmark_live_capture(Namespace(checkpoint="unused.pt", device="cpu", threads=1,
        iterations=2, warmup=1, countdown=0, output=tmp_path / "report.json"))
    assert report["live_input"] is False
    assert report["gameplay_success"] == "NOT EVALUATED"
    assert json.loads((tmp_path / "report.json").read_text())["iterations"] == 2


def test_live_capture_benchmark_rejects_unbounded_run_before_loading_model():
    with pytest.raises(ValueError):
        benchmark_live_capture(Namespace(iterations=100000, warmup=0))
