"""Collection fixtures exercise OS boundaries; they are not gameplay evidence."""

import json
from pathlib import Path
import time

from PIL import Image
import pytest

from roblox_learner.collect import _CommandLease, collect_session, parse_command
from roblox_learner.desktop import Action, InputController, RawInput, Window


class FakeClock:
    def __init__(self):
        self.value = 0.0
        self.on_sleep = None

    def __call__(self):
        return self.value

    def sleep(self, duration):
        self.value += max(duration, 0.000001)
        if self.on_sleep:
            self.on_sleep(self.value)


class FakeBackend:
    def __init__(self):
        self.focused = True
        self.escape = False
        self.held = set()
        self.events = []
        self.capture_count = 0
        self.lose_focus_on_capture = None

    def is_focused(self):
        return self.focused

    def window(self):
        return Window(1, 1, 0, 0, 200, 150)

    def capture(self):
        self.capture_count += 1
        self.events.append(("capture", tuple(sorted(self.held))))
        if self.capture_count == self.lose_focus_on_capture:
            self.focused = False
        return Image.new("RGB", (100, 100), (200 if self.held else 0, 0, 0))

    def input_state(self):
        return RawInput(escape=self.escape)

    def key(self, name, down):
        self.events.append(("key", name, down))
        if down:
            self.held.add(name)
        else:
            self.held.discard(name)

    def mouse_button(self, name, down):
        self.events.append(("button", name, down))

    def mouse_move(self, dx, dy):
        self.events.append(("move", dx, dy))


class FakeWatchdog:
    def __init__(self, controller, **_):
        self.controller = controller

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.controller.release_all()

    def heartbeat(self):
        pass

    def raise_if_stopped(self):
        pass


ACTIONS = [Action("idle"), Action("forward", keys=("w",)), Action("jump", keys=("space",))]


def _run(tmp_path, *, command=None, backend=None, clock=None, **kwargs):
    backend = backend or FakeBackend()
    clock = clock or FakeClock()
    inbox = tmp_path / "command.json"
    if command is not None:
        inbox.write_text(json.dumps(command))
    result = collect_session(backend, ACTIONS, tmp_path / "episode", inbox,
                             tmp_path / "preview.png", tmp_path / "status.json",
                             seconds=0.6, fps=10, clock=clock, sleep=clock.sleep,
                             watchdog_factory=FakeWatchdog, **kwargs)
    return result, backend


@pytest.mark.parametrize("value", [
    {"id": "x", "action": "w; arbitrary command", "seconds": 1},
    {"id": "x", "action": "forward", "seconds": 5.1},
    {"id": "x", "action": "forward", "seconds": float("nan")},
    {"id": "x", "action": "forward", "seconds": True},
    {"id": "x", "action": "forward", "seconds": 1, "keys": ["return"]},
    {"id": "", "action": "forward", "seconds": 1},
    {"id": "x", "action": ["forward"], "seconds": 1},
])
def test_teacher_commands_are_finite_and_vocabulary_constrained(value):
    with pytest.raises(ValueError):
        parse_command(value, ACTIONS)


def test_collector_labels_causal_frames_and_never_replays_seen_command(tmp_path):
    result, backend = _run(tmp_path, command={"id": "forward-1", "action": "forward", "seconds": 0.2})
    rows = [json.loads(line) for line in (tmp_path / "episode/actions.jsonl").read_text().splitlines()]
    assert result["commands_completed"] == 1
    assert len(rows) == 2
    assert [row["action"] for row in rows] == [1, 1]
    assert [row["history_reset"] for row in rows] == [True, False]
    assert all(row["source"] == "tool_directed_teacher" for row in rows)
    assert all(row["interval_seconds"] > 0 for row in rows)
    with Image.open(tmp_path / "episode" / rows[0]["frame"]) as first:
        assert first.getpixel((0, 0)) == (0, 0, 0)
    assert backend.events.index(("key", "w", True)) > backend.events.index(("capture", ()))
    assert backend.events.count(("key", "w", True)) == 1
    assert ("key", "w", False) in backend.events
    assert not backend.held
    assert (tmp_path / "preview.png").is_file()
    assert json.loads((tmp_path / "status.json").read_text())["last_completed_command"] == "forward-1"


def test_idle_preview_produces_no_demonstration_labels_or_input(tmp_path):
    result, backend = _run(tmp_path)
    assert result["samples"] == 0
    assert (tmp_path / "episode/actions.jsonl").read_text() == ""
    assert backend.capture_count == 1
    assert not [event for event in backend.events if event[0] in ("key", "button", "move")]


def test_capture_only_rejects_inbox_actions_without_input(tmp_path):
    result, backend = _run(tmp_path, command={"id": "x", "action": "forward", "seconds": 0.2}, capture_only=True)
    assert result["commands_completed"] == result["samples"] == 0
    assert not [event for event in backend.events if event[0] in ("key", "button", "move")]
    assert "capture_only" in (tmp_path / "episode/events.jsonl").read_text()


def test_focus_loss_during_capture_stops_without_saving_foreign_frame(tmp_path):
    backend = FakeBackend()
    backend.lose_focus_on_capture = 3  # Initial preview, causal frame, then focus lost.
    result, backend = _run(tmp_path, command={"id": "x", "action": "forward", "seconds": 0.3}, backend=backend)
    assert result["stop_reason"] == "stopped"
    assert "focus" in result["error"].lower()
    assert not backend.held
    assert len(list((tmp_path / "episode/frames").glob("*.jpg"))) == 1
    assert backend.events.count(("key", "w", False)) >= 1


def test_escape_releases_injected_input_and_stops(tmp_path):
    clock, backend = FakeClock(), FakeBackend()
    clock.on_sleep = lambda now: setattr(backend, "escape", now >= 0.05)
    result, backend = _run(tmp_path, command={"id": "x", "action": "forward", "seconds": 0.3}, backend=backend, clock=clock)
    assert "Escape" in result["error"]
    assert not backend.held


def test_new_command_after_idle_starts_a_new_history_segment(tmp_path):
    clock = FakeClock()
    def command_two(now):
        if now >= 0.25:
            (tmp_path / "command.json").write_text(json.dumps({"id": "two", "action": "jump", "seconds": 0.1}))
    clock.on_sleep = command_two
    result, _ = _run(tmp_path, command={"id": "one", "action": "forward", "seconds": 0.1}, clock=clock)
    rows = [json.loads(line) for line in (tmp_path / "episode/actions.jsonl").read_text().splitlines()]
    assert result["commands_completed"] == 2
    assert [row["action"] for row in rows] == [1, 2]
    assert all(row["history_reset"] for row in rows)


def test_independent_deadline_releases_input_during_blocked_main_thread():
    backend = FakeBackend()
    controller = InputController(backend)
    with _CommandLease(controller, time.monotonic() + 0.03, time.monotonic) as lease:
        assert lease.apply(ACTIONS[1])
        time.sleep(0.1)  # Simulate a blocked screenshot call, without polling.
        assert not backend.held
        assert not lease.apply(ACTIONS[1])

