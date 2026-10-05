"""Portable safety tests: no live Roblox, GUI calls, or permissions required."""

import json
import threading
import time

import pytest
from PIL import Image

from roblox_learner.desktop import (
    Action, DesktopError, EmergencyStop, FocusLost, InputController, RawInput,
    MacDesktop, Watchdog, Window, parse_actions,
)
from roblox_learner.play import Prediction, run_episode
from roblox_learner.record import match_action, predominant_inputs, record_episode


class FakeDesktop:
    def __init__(self):
        self.focused = True
        self.state = RawInput()
        self.events = []
        self.frames = 0

    def is_focused(self):
        return self.focused

    def window(self):
        return Window(100, 101, 0, 0, 320, 180)

    def capture(self):
        if not self.focused:
            raise FocusLost("Fake Roblox lost focus")
        self.frames += 1
        return Image.new("RGB", (320, 180), (self.frames % 255, 20, 30))

    def input_state(self):
        return self.state

    def key(self, name, down):
        if down and not self.focused:
            raise FocusLost("Refusing key down outside Roblox")
        self.events.append(("key", name, down))

    def mouse_button(self, name, down):
        if down and not self.focused:
            raise FocusLost("Refusing mouse down outside Roblox")
        self.events.append(("button", name, down))

    def mouse_move(self, dx, dy):
        if not self.focused:
            raise FocusLost("Refusing mouse move outside Roblox")
        self.events.append(("mouse", dx, dy))


class FixedPolicy:
    def __init__(self, action=0, callback=None, confidence=0.99):
        self.action, self.callback, self.confidence = action, callback, confidence

    def predict(self, frame):
        if self.callback:
            self.callback()
        return Prediction(self.action, self.confidence, 0.1)


def test_action_validation_and_round_trip():
    raw = [{"name": "jump_forward", "keys": ["w", "space"], "mouse": {"dx": .05, "button": "right"}, "duration": .2}]
    actions = parse_actions(raw)
    assert parse_actions([a.to_dict() for a in actions]) == actions
    assert parse_actions({"actions": raw}) == actions


@pytest.mark.parametrize("raw", [
    [], [{"name": "bad", "keys": ["escape"]}], [{"name": "bad", "keys": ["f8"]}],
    [{"name": "bad", "keys": ["command"]}], [{"name": "bad", "keys": ["w", "w"]}],
    [{"name": "bad", "mouse": {"dx": float("nan")}}], [{"name": "bad", "mouse": {"dy": 2}}],
    [{"name": "bad", "mouse": {"x": .4}}], [{"name": "bad", "duration": -1}],
    [{"name": "same"}, {"name": "same"}], [{"name": "bad", "mouse": {"button": "back"}}],
    [{"name": "bad", "mouse": {"button": []}}], [{"name": "bad", "keyz": ["w"]}],
])
def test_action_validation_rejects_invalid_or_reserved_controls(raw):
    with pytest.raises(ValueError):
        parse_actions(raw)


def test_input_transition_keeps_held_keys_without_repeat_and_releases_on_exit():
    backend = FakeDesktop()
    with InputController(backend) as controller:
        controller.apply(Action("move", ("w",), button="left"))
        controller.apply(Action("jump", ("w", "space"), button="left"))
        controller.apply(Action("idle"))
        assert not controller.held_keys and controller.held_button is None
    assert backend.events.count(("key", "w", True)) == 1
    assert backend.events.count(("key", "w", False)) == 1
    assert backend.events.count(("button", "left", True)) == 1
    assert backend.events.count(("button", "left", False)) == 1


def test_focus_loss_releases_previous_controls_and_sends_no_new_down():
    backend = FakeDesktop()
    controller = InputController(backend)
    controller.apply(Action("move", ("w",), button="right"))
    backend.events.clear()
    backend.focused = False
    with pytest.raises(FocusLost):
        controller.apply(Action("jump", ("space",), button="left"))
    assert backend.events == [("key", "w", False), ("button", "right", False)]


def test_escape_releases_inputs():
    backend = FakeDesktop()
    controller = InputController(backend)
    controller.apply(Action("move", ("w",)))
    backend.state = RawInput(escape=True)
    with pytest.raises(EmergencyStop):
        controller.check()
    assert not controller.held_keys
    assert backend.events[-1] == ("key", "w", False)


def test_watchdog_timeout_independently_releases_held_key():
    backend = FakeDesktop()
    controller = InputController(backend)
    controller.apply(Action("move", ("w",)))
    clock = [100.0]
    guard = Watchdog(controller, timeout=.5, rss=lambda: 50, clock=lambda: clock[0])
    clock[0] += .6
    guard.check_once()
    assert guard.stop_event.is_set()
    assert "watchdog" in guard.reason
    assert not controller.held_keys


def test_watchdog_releases_on_memory_limit():
    backend = FakeDesktop()
    controller = InputController(backend)
    controller.apply(Action("move", ("w",)))
    guard = Watchdog(controller, rss=lambda: 6000)
    guard.check_once()
    assert guard.stop_event.is_set()
    assert "memory" in guard.reason
    assert not controller.held_keys


def test_recording_never_invents_labels_for_unknown_controls():
    actions = [Action("idle"), Action("forward", ("w",)), Action("look", dx=.1)]
    assert match_action(actions, ("w",), (), 0, 0)[0] == 1
    assert match_action(actions, ("w", "space"), (), 0, 0)[0] is None
    assert match_action(actions, (), (), .11, 0)[0] == 2
    assert match_action(actions, (), ("left",), 0, 0)[0] is None
    assert match_action(actions, (), (), .9, 0)[0] is None
    assert predominant_inputs([RawInput(keys=("w",)), RawInput(), RawInput(keys=("w",))]) == (("w",), ())


def test_episode_saves_real_frames_and_cleans_up(tmp_path):
    backend = FakeDesktop()
    summary = run_episode(backend, FixedPolicy(), [Action("forward", ("w",))], tmp_path / "episode", seconds=.2, fps=60)
    assert summary["steps"] >= 1
    assert summary["outcome"] == "unverified"
    assert ("key", "w", False) in backend.events
    rows = [json.loads(line) for line in (tmp_path / "episode" / "episode.jsonl").read_text().splitlines()]
    assert rows[0]["type"] == "start" and rows[-1]["type"] == "stop"
    assert (tmp_path / "episode" / rows[0]["frame_path"]).is_file()
    assert (tmp_path / "episode" / rows[-1]["frame_path"]).is_file()
    step = next(row for row in rows if row["type"] == "step")
    assert step["end_to_end_ms"] >= 0
    assert "rss_mb" in step


def test_episode_abstains_without_sending_key_down(tmp_path):
    backend = FakeDesktop()
    run_episode(backend, FixedPolicy(confidence=.1), [Action("forward", ("w",))], tmp_path / "episode", seconds=.04, fps=60, min_confidence=.9)
    assert not [event for event in backend.events if event[0] == "key" and event[-1] is True]


def test_episode_focus_loss_during_inference_prevents_action(tmp_path):
    backend = FakeDesktop()
    policy = FixedPolicy(callback=lambda: setattr(backend, "focused", False))
    summary = run_episode(backend, policy, [Action("forward", ("w",))], tmp_path / "episode", seconds=.1)
    assert "error" in summary["reason"]
    assert not backend.events
    assert summary["terminal_frame_current"] is False


def test_episode_invalid_prediction_cannot_touch_keyboard(tmp_path):
    backend = FakeDesktop()
    summary = run_episode(backend, FixedPolicy(action=999), [Action("idle")], tmp_path / "episode", seconds=.1)
    assert "invalid action" in summary["reason"]
    assert not backend.events


def test_episode_watchdog_releases_keys_during_slow_inference(tmp_path):
    released = threading.Event()
    inference_entered = threading.Event()
    class ObservedDesktop(FakeDesktop):
        def key(self, name, down):
            super().key(name, down)
            if name == "w" and not down:
                released.set()
    backend = ObservedDesktop()
    calls = [0]
    def slow_second_call():
        calls[0] += 1
        if calls[0] == 2:
            inference_entered.set()
            # The model cannot return until the independent watchdog releases
            # W. A generous outer episode budget avoids conflating a loaded
            # host's startup time with the inference timeout being tested.
            assert released.wait(timeout=5), "Watchdog did not release input while inference was blocked"
    summary = run_episode(backend, FixedPolicy(callback=slow_second_call), [Action("forward", ("w",))], tmp_path / "episode", seconds=10, fps=60, watchdog_seconds=.25)
    assert inference_entered.is_set()
    assert "watchdog" in summary["reason"]
    assert released.is_set()
    assert ("key", "w", False) in backend.events


def test_recorder_writes_aligned_labels_without_any_injection(tmp_path):
    backend = FakeDesktop()
    backend.state = RawInput(keys=("w",))
    output = tmp_path / "demonstration"
    summary = record_episode(backend, [Action("idle"), Action("forward", ("w",))], output, seconds=.04, fps=60, start_immediately=True)
    assert summary["samples"] >= 1
    assert not backend.events
    rows = [json.loads(line) for line in (output / "actions.jsonl").read_text().splitlines()]
    assert all(row["action"] == 1 for row in rows)
    assert all((output / row["frame"]).is_file() for row in rows)
    assert all(row["capture_to_label"] == "frame_start_then_predominant_inputs" for row in rows)
    assert rows[0]["history_reset"] is True


def test_recorder_drops_unmapped_samples(tmp_path):
    backend = FakeDesktop()
    backend.state = RawInput(keys=("q",))
    summary = record_episode(backend, [Action("idle")], tmp_path / "episode", seconds=.035, fps=60, start_immediately=True)
    assert summary["samples"] == 0 and summary["dropped_unmapped"] >= 1
    assert not backend.events


def test_recorder_labels_post_capture_inputs_even_when_capture_is_slow(tmp_path):
    class SlowDesktop(FakeDesktop):
        def capture(self):
            time.sleep(.025)
            self.state = RawInput(keys=("w",))
            return super().capture()
    backend = SlowDesktop()
    output = tmp_path / "episode"
    summary = record_episode(backend, [Action("idle"), Action("forward", ("w",))], output,
        seconds=.08, fps=60, start_immediately=True)
    assert summary["samples"] >= 1
    rows = [json.loads(line) for line in (output / "actions.jsonl").read_text().splitlines()]
    assert all(row["action"] == 1 for row in rows)
    assert all(row["capture_ms"] >= 20 for row in rows)


def test_recorder_raw_deltas_handle_locked_cursor(tmp_path):
    class LockedCursorDesktop(FakeDesktop):
        def __init__(self):
            super().__init__()
            self.total = 0
            self.reads_after_capture = 0
            self.stopped = False
        def start_mouse_monitor(self):
            return True
        def stop_mouse_monitor(self):
            self.stopped = True
        def capture(self):
            self.reads_after_capture = 0
            return super().capture()
        def input_state(self):
            self.reads_after_capture += 1
            if self.reads_after_capture == 2:
                self.total += 32
            return RawInput(buttons=("right",), x=160, y=90,
                mouse_total_dx=self.total, mouse_total_dy=0)
    backend = LockedCursorDesktop()
    output = tmp_path / "episode"
    record_episode(backend, [Action("hold", button="right"), Action("turn", dx=.1, button="right")], output,
        seconds=.045, fps=60, start_immediately=True, require_raw_mouse=True)
    rows = [json.loads(line) for line in (output / "actions.jsonl").read_text().splitlines()]
    assert rows and all(row["action"] == 1 for row in rows)
    assert backend.stopped


def test_recorder_refuses_mouse_lock_labels_without_raw_access(tmp_path):
    with pytest.raises(DesktopError, match="Raw mouse capture"):
        record_episode(FakeDesktop(), [Action("turn", dx=.1)], tmp_path / "episode", require_raw_mouse=True)


def test_posted_mouse_down_selects_drag_before_os_state_updates():
    from types import SimpleNamespace
    class FakeQuartz:
        kCGEventLeftMouseUp, kCGEventLeftMouseDown = 1, 2
        kCGEventRightMouseUp, kCGEventRightMouseDown = 3, 4
        kCGEventOtherMouseUp, kCGEventOtherMouseDown = 5, 6
        kCGEventMouseMoved, kCGEventLeftMouseDragged = 7, 8
        kCGEventRightMouseDragged, kCGEventOtherMouseDragged = 9, 10
        kCGHIDEventTap, kCGMouseEventDeltaX, kCGMouseEventDeltaY = 0, 100, 101
        def __init__(self):
            self.posted = []
        def CGEventCreate(self, _): return None
        def CGEventGetLocation(self, _): return SimpleNamespace(x=160, y=90)
        def CGEventCreateMouseEvent(self, _, event_type, point, button): return {"type": event_type}
        def CGEventSetIntegerValueField(self, event, key, value): event[key] = value
        def CGEventPost(self, _, event): self.posted.append(event)
    backend = object.__new__(MacDesktop)
    backend.q = FakeQuartz()
    backend._held_mouse_button = None
    backend.require_focus = lambda: None
    backend.window = lambda: Window(100, 101, 0, 0, 320, 180)
    backend.input_state = lambda: RawInput(x=160, y=90)  # OS button state still empty.
    backend.mouse_button("right", True)
    backend.mouse_move(.1, 0)
    assert backend.q.posted[-1]["type"] == backend.q.kCGEventRightMouseDragged


def test_real_policy_checkpoint_round_trip_infers_without_desktop_calls(tmp_path):
    from roblox_learner.model import CompactPolicy, PolicyConfig, save_checkpoint
    from roblox_learner.play import TorchPolicy
    model = CompactPolicy(PolicyConfig(action_count=2))
    checkpoint = tmp_path / "untrained-inference-fixture.pt"
    save_checkpoint(checkpoint, model, metadata={"actions": [{"name": "idle"}, {"name": "forward", "keys": ["w"]}],
        "fixture": "Untrained weights for API testing only"})
    policy = TorchPolicy(checkpoint)
    prediction = policy.predict(Image.new("RGB", (320, 180), (20, 30, 40)))
    assert 0 <= prediction.action < 2 and 0 <= prediction.confidence <= 1
    assert prediction.inference_ms > 0
    assert policy.accounted_memory_mb() > 0
