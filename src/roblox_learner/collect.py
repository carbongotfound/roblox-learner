"""Collect real Roblox demonstrations from bounded, local teacher commands.

The teacher is external to the trained policy: it observes the preview and sends
one finite, vocabulary-constrained action at a time. These recordings are not
evidence that a learned policy can play the game.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any, Callable, Sequence
import uuid

from .desktop import Action, DesktopBackend, DesktopError, InputController, MacDesktop, Watchdog, countdown, parse_actions


@dataclass(frozen=True)
class TeacherCommand:
    id: str
    action_index: int
    seconds: float


def parse_command(value: Any, actions: Sequence[Action]) -> TeacherCommand:
    """Accept a finite action name only; never interpret strings as executable code."""
    if not isinstance(value, dict) or set(value) != {"id", "action", "seconds"}:
        raise ValueError("Command must contain exactly id, action and seconds")
    identifier, name, seconds = value["id"], value["action"], value["seconds"]
    if not isinstance(identifier, str) or not 1 <= len(identifier) <= 128 or any(ord(c) < 32 for c in identifier):
        raise ValueError("Command id must be a nonempty string of at most 128 printable characters")
    if not isinstance(name, str):
        raise ValueError("Command action must be a vocabulary action name")
    by_name = {action.name: index for index, action in enumerate(actions)}
    if name not in by_name:
        raise ValueError(f"Unknown action name: {name!r}")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0.01 <= seconds <= 5:
        raise ValueError("Command seconds must be finite and between 0.01 and 5")
    return TeacherCommand(identifier, by_name[name], float(seconds))


def _json_file(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _preview_file(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.save(temporary, format="PNG")
    temporary.replace(path)


class _CommandLease:
    """Release held inputs at the command deadline, even during blocked capture."""

    def __init__(self, controller: InputController, deadline: float, clock: Callable[[], float]):
        self.controller, self.deadline, self.clock = controller, deadline, clock
        self.expired = threading.Event()
        self._lock = threading.RLock()
        self.release_error: Exception | None = None
        self.timer = threading.Timer(max(0.0, deadline - clock()), self._expire)
        self.timer.daemon = True

    def _expire(self) -> None:
        self.expired.set()
        with self._lock:
            try:
                self.controller.release_all()
            except Exception as exc:
                self.release_error = exc

    def active(self) -> bool:
        if self.release_error:
            raise DesktopError(f"Command deadline cleanup failed: {self.release_error}")
        return not self.expired.is_set() and self.clock() < self.deadline

    def apply(self, action: Action) -> bool:
        with self._lock:
            if not self.active():
                return False
            self.controller.apply(action)
            return True

    def __enter__(self) -> "_CommandLease":
        self.timer.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.expired.set()
        self.timer.cancel()
        with self._lock:
            self.controller.release_all()
        self.timer.join(timeout=1)
        if self.release_error:
            raise DesktopError(f"Command deadline cleanup failed: {self.release_error}")


def collect_session(
    backend: DesktopBackend,
    actions: Sequence[Action],
    output: Path,
    inbox: Path,
    preview: Path,
    status: Path,
    *,
    seconds: float = 900,
    fps: float = 10,
    capture_only: bool = False,
    max_side: int = 384,
    max_disk_mb: float = 1024,
    watchdog_seconds: float = 2,
    config: dict[str, Any] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    watchdog_factory: Callable[..., Any] = Watchdog,
) -> dict[str, Any]:
    """Run a finite teacher collection session using only the visible game window.

    An inbox command repeats its discrete action at up to ``fps``. Keys/buttons
    stay held between observations; mouse deltas execute once per action tick.
    No frames are labeled while waiting for commands or in capture-only mode.
    """
    if not math.isfinite(seconds) or not 0 < seconds <= 86400:
        raise ValueError("seconds must be finite, positive and at most 86400")
    if not math.isfinite(fps) or not 1 <= fps <= 60:
        raise ValueError("fps must be between 1 and 60")
    if not 96 <= max_side <= 4096 or not math.isfinite(max_disk_mb) or max_disk_mb <= 0:
        raise ValueError("max-side must be 96..4096 and max-disk-mb finite and positive")
    if not math.isfinite(watchdog_seconds) or not 0.1 <= watchdog_seconds <= 5:
        raise ValueError("watchdog-seconds must be between 0.1 and 5")
    if not actions or len({action.name for action in actions}) != len(actions):
        raise ValueError("Collection requires a nonempty vocabulary with unique action names")
    output, inbox, preview, status = map(Path, (output, inbox, preview, status))
    if len({p.resolve() for p in (inbox, preview, status)}) != 3:
        raise ValueError("Inbox, preview and status paths must be distinct")
    reserved = {output.resolve() / name for name in ("actions.json", "actions.jsonl", "events.jsonl", "episode.json")}
    if any(p.resolve() in reserved for p in (inbox, preview, status)):
        raise ValueError("Control files cannot overwrite dataset manifests")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to mix or overwrite an existing episode: {output}")
    (output / "frames").mkdir(parents=True, exist_ok=True)
    _json_file(output / "actions.json", [action.to_dict() for action in actions])
    episode_id = str(uuid.uuid4())
    started, started_at = clock(), time.time()
    deadline = started + seconds
    samples = commands_completed = size_bytes = 0
    last_content: bytes | None = None
    seen_commands: set[str] = set()
    last_preview = -float("inf")
    state: dict[str, Any] = {
        "schema_version": 1, "source": "tool_directed_teacher", "episode_id": episode_id,
        "pid": os.getpid(),
        "phase": "starting", "capture_only": capture_only, "samples": 0,
        "commands_completed": 0, "last_completed_command": None,
        "preview": str(preview.resolve()), "output": str(output.resolve()),
        "gameplay_success": "NOT EVALUATED; commands are teacher-directed",
    }
    reason, error = "duration_limit", None
    _json_file(status, state)
    with (output / "actions.jsonl").open("w", buffering=1) as records, (output / "events.jsonl").open("w", buffering=1) as events:
        def event(kind: str, **fields: Any) -> None:
            events.write(json.dumps({"type": kind, "timestamp": time.time(), "episode_id": episode_id, **fields}, allow_nan=False) + "\n")

        def publish(phase: str | None = None, **fields: Any) -> None:
            state.update(fields)
            state.update({"samples": samples, "commands_completed": commands_completed, "updated_at": time.time()})
            if phase is not None:
                state["phase"] = phase
            _json_file(status, state)

        try:
            with InputController(backend) as controller, watchdog_factory(
                controller, timeout=watchdog_seconds, memory_limit_mb=5500, clock=clock
            ) as watchdog:
                def check() -> None:
                    watchdog.raise_if_stopped()
                    controller.check()
                    watchdog.heartbeat()

                def capture() -> Any:
                    check()
                    frame = backend.capture()
                    check()  # Never save an observation after a detected focus change.
                    return frame

                def refresh_preview(frame: Any | None = None) -> None:
                    nonlocal last_preview
                    check()
                    if frame is None:
                        frame = capture()
                    check()
                    _preview_file(preview, frame)
                    last_preview = clock()
                    state["preview_updated_at"] = time.time()

                def save_pending(pending: dict[str, Any] | None, interval_end: float) -> None:
                    nonlocal samples
                    if pending is None:
                        return
                    observation_time = pending.pop("_monotonic")
                    pending["interval_seconds"] = max(0.000001, interval_end - observation_time)
                    records.write(json.dumps(pending, allow_nan=False) + "\n")
                    samples += 1

                event("session_start", requested_fps=fps, capture_only=capture_only)
                refresh_preview()
                publish("ready")
                while clock() < deadline:
                    check()
                    if clock() - last_preview >= 1:
                        refresh_preview()
                        publish()
                    content = None
                    if inbox.is_file():
                        if inbox.stat().st_size > 4096:
                            content = b"oversized-command"
                        else:
                            content = inbox.read_bytes()
                    if content is None or content == last_content:
                        sleep(min(0.02, max(0.0, deadline - clock())))
                        continue
                    last_content = content
                    try:
                        command = parse_command(json.loads(content), actions)
                    except (ValueError, UnicodeDecodeError) as exc:
                        event("command_rejected", error=str(exc))
                        publish("rejected", error=str(exc), command_id=None)
                        continue
                    if command.id in seen_commands:
                        continue
                    seen_commands.add(command.id)
                    if len(seen_commands) > 10000:
                        raise DesktopError("Session command limit reached")
                    if capture_only:
                        event("command_rejected", command_id=command.id, error="capture_only")
                        publish("rejected", error="capture_only: no commands execute", command_id=command.id)
                        continue
                    action = actions[command.action_index]
                    before_samples = samples
                    publish("running", command_id=command.id, action=action.name,
                            requested_seconds=command.seconds, error=None)
                    event("command_start", command_id=command.id, action=action.name, seconds=command.seconds)
                    # Obtain the first causal observation before starting the action timer.
                    frame = capture()
                    command_started = clock()
                    command_deadline = min(deadline, command_started + command.seconds)
                    pending = None
                    first = True
                    with _CommandLease(controller, command_deadline, clock) as lease:
                        while lease.active() and clock() < deadline:
                            check()
                            if not first:
                                frame = capture()
                            if not lease.active():
                                break
                            observed = clock()
                            captured_at = time.time()
                            save_pending(pending, observed)
                            pending = None
                            if not lease.apply(action):
                                break
                            # The immutable screenshot precedes this injected action.
                            filename = f"frames/{samples:07d}.jpg"
                            stored = frame.copy()
                            stored.thumbnail((max_side, max_side))
                            stored.save(output / filename, format="JPEG", quality=90)
                            size_bytes += (output / filename).stat().st_size
                            pending = {"frame": filename, "action": command.action_index,
                                       "timestamp": captured_at, "episode_id": episode_id,
                                       "history_reset": first, "teacher_command_id": command.id,
                                       "source": "tool_directed_teacher", "_monotonic": observed,
                                       "capture_to_label": "observation_before_injected_action"}
                            first = False
                            if size_bytes > max_disk_mb * 1_000_000:
                                raise DesktopError("Episode disk limit reached")
                            interval_end = min(command_deadline, clock() + 1 / fps)
                            while clock() < interval_end and lease.active():
                                check()
                                sleep(min(0.01, max(0.0, interval_end - clock())))
                        save_pending(pending, min(clock(), command_deadline))
                    commands_completed += 1
                    event("command_complete", command_id=command.id, samples=samples - before_samples,
                          elapsed_seconds=clock() - command_started)
                    refresh_preview()
                    publish("complete", last_completed_command=command.id,
                            command_samples=samples - before_samples, command_id=command.id)
        except KeyboardInterrupt:
            reason = "keyboard_interrupt"
        except Exception as exc:
            reason, error = "stopped", str(exc)
            event("stopped", error=error)
        finally:
            summary = {**state, "phase": "stopped", "stop_reason": reason, "error": error,
                       "samples": samples, "commands_completed": commands_completed,
                       "started_at": started_at, "ended_at": time.time(),
                       "elapsed_seconds": clock() - started, "requested_fps": fps,
                       "image_max_side": max_side, "image_bytes": size_bytes,
                       "config": config or {}}
            _json_file(output / "episode.json", summary)
            _json_file(status, summary)
            event("session_end", stop_reason=reason, samples=samples, commands_completed=commands_completed)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="New independent episode directory")
    parser.add_argument("--inbox", required=True, help="Local JSON teacher command file; write atomically")
    parser.add_argument("--preview", required=True, help="Latest Roblox-window PNG")
    parser.add_argument("--status", required=True, help="Atomic command/session status JSON")
    parser.add_argument("--seconds", type=float, default=900)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--capture-only", action="store_true", help="Preview only: never inject inputs or label demonstrations")
    parser.add_argument("--max-side", type=int, default=384)
    parser.add_argument("--max-disk-mb", type=float, default=1024)
    parser.add_argument("--watchdog-seconds", type=float, default=2)
    parser.add_argument("--countdown", type=float, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    effective = list(sys.argv[1:] if argv is None else argv)
    if effective and effective[0] == 'potato':
        from .potato import main as potato_main
        return potato_main(effective[1:])
    args = build_parser().parse_args(argv)
    def terminate_gracefully(_signal: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    # App launchers commonly use SIGTERM when stopping a child. Convert that
    # into normal context-manager cleanup, including release of held keys.
    signal.signal(signal.SIGTERM, terminate_gracefully)
    try:
        config = json.loads(Path(args.config).read_text())
        actions = parse_actions(config)
        print("Teacher collection only; not trained-policy evaluation. Focus Roblox. Escape stops.", flush=True)
        countdown(args.countdown)
        result = collect_session(MacDesktop(), actions, Path(args.output), Path(args.inbox),
                                 Path(args.preview), Path(args.status), seconds=args.seconds, fps=args.fps,
                                 capture_only=args.capture_only, max_side=args.max_side,
                                 max_disk_mb=args.max_disk_mb, watchdog_seconds=args.watchdog_seconds, config=config)
        print(json.dumps(result, indent=2), flush=True)
        return 2 if result.get("error") else 0
    except (ValueError, OSError, DesktopError) as exc:
        print(f"Collection error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
