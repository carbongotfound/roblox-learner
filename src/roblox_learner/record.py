"""Record human demonstrations from visible Roblox pixels and OS inputs.

F8 starts/pauses recording; Escape ends it. Frames are captured at the start of
each interval and paired with the predominant human action during that interval.
Unknown key combinations are dropped, never silently relabeled as idle.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

from .desktop import (
    Action, DesktopBackend, DesktopError, FocusLost, MacDesktop, RawInput,
    countdown, parse_actions, portable_diagnostics, process_rss_mb,
)


def match_action(actions: Sequence[Action], keys: Sequence[str], buttons: Sequence[str],
                 dx: float, dy: float, max_mouse_error: float = 0.08) -> tuple[int | None, float]:
    """Exact discrete inputs, nearest mouse bin; reject distant/unmapped input."""
    candidates = []
    for index, action in enumerate(actions):
        expected_buttons = {action.button} if action.button else set()
        if set(keys) != set(action.keys) or set(buttons) != expected_buttons:
            continue
        distance = math.hypot(dx - action.dx, dy - action.dy)
        candidates.append((distance, index))
    if not candidates:
        return None, math.inf
    distance, index = min(candidates)
    return (index if distance <= max_mouse_error else None), distance


def predominant_inputs(states: Sequence[RawInput]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not states:
        return (), ()
    return collections.Counter((tuple(sorted(s.keys)), tuple(sorted(s.buttons))) for s in states).most_common(1)[0][0]


def record_episode(backend: DesktopBackend, actions: Sequence[Action], output: Path,
                   *, seconds: float = 120, fps: float = 10, max_side: int = 384,
                   start_immediately: bool = False, max_mouse_error: float = 0.08,
                   max_disk_mb: float = 1024, config: dict[str, Any] | None = None,
                   require_raw_mouse: bool = False) -> dict[str, Any]:
    if not math.isfinite(seconds) or not 0 < seconds <= 86400:
        raise ValueError("seconds must be finite, positive, and at most 86400")
    if not math.isfinite(fps) or not 1 <= fps <= 60:
        raise ValueError("fps must be between 1 and 60")
    if max_side < 96 or max_side > 4096:
        raise ValueError("max_side must be between 96 and 4096")
    if max_disk_mb <= 0 or not math.isfinite(max_disk_mb):
        raise ValueError("max_disk_mb must be finite and positive")
    if max_mouse_error < 0 or not math.isfinite(max_mouse_error):
        raise ValueError("max_mouse_error must be finite and nonnegative")
    if not actions:
        raise ValueError("A nonempty action vocabulary is required")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to mix/overwrite an existing episode: {output}")
    mouse_monitor = getattr(backend, "start_mouse_monitor", None)
    has_raw_mouse = bool(mouse_monitor()) if mouse_monitor is not None else False
    if require_raw_mouse and not has_raw_mouse and any(a.dx or a.dy for a in actions):
        stop_monitor = getattr(backend, "stop_mouse_monitor", None)
        if stop_monitor:
            stop_monitor()
        raise DesktopError("Raw mouse capture requires existing Input Monitoring access. Absolute cursor deltas cannot reliably label Roblox camera turns. --allow-cursor-deltas is only for games where the cursor never locks or recenters.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "frames").mkdir()
    (output / "actions.json").write_text(json.dumps([a.to_dict() for a in actions], indent=2) + "\n")
    episode_id = str(uuid.uuid4())
    start = time.monotonic()
    deadline = start + seconds
    active = start_immediately
    previous_toggle = False
    samples = dropped = size_bytes = 0
    reason = "duration_limit"
    action_counts: collections.Counter[str] = collections.Counter()
    history_reset = True
    print("Recording human controls only. F8 starts/pauses, Escape stops.", flush=True)
    with (output / "actions.jsonl").open("w", buffering=1) as records, (output / "events.jsonl").open("w", buffering=1) as events:
        def event(kind: str, **fields: Any) -> None:
            events.write(json.dumps({"type": kind, "timestamp": time.time(), "episode_id": episode_id, **fields}) + "\n")

        event("start", fps=fps, seconds=seconds, initially_active=active, config=config or {},
            mouse_capture_mode="raw_event_deltas" if has_raw_mouse else "absolute_cursor_fallback")
        try:
            while time.monotonic() < deadline:
                if process_rss_mb() > 5500:
                    reason = "memory_limit"
                    break
                if not backend.is_focused():
                    raise FocusLost("Roblox lost foreground focus; recording stopped")
                initial = backend.input_state()
                if initial.escape:
                    reason = "escape"
                    break
                if initial.toggle and not previous_toggle:
                    active = not active
                    history_reset = True
                    event("resume" if active else "pause")
                    print("Recording" if active else "Paused", flush=True)
                previous_toggle = initial.toggle
                if not active:
                    time.sleep(0.01)
                    continue
                window = backend.window()
                capture_started_at = time.time()
                frame = backend.capture()
                captured_at = time.time()
                interval_start = time.monotonic()
                # Label inputs after the observation actually exists. Slow
                # capture must never pair new pixels with only pre-capture keys.
                initial = backend.input_state()
                if initial.escape:
                    reason = "escape"
                    break
                if not backend.is_focused():
                    raise FocusLost("Roblox lost foreground focus; recording stopped")
                states = [initial]
                interval_end = min(interval_start + 1 / fps, deadline)
                last = initial
                while time.monotonic() < interval_end:
                    time.sleep(min(0.01, max(0, interval_end - time.monotonic())))
                    last = backend.input_state()
                    if not backend.is_focused():
                        raise FocusLost("Roblox lost foreground focus; recording stopped")
                    if last.escape:
                        reason = "escape"
                        break
                    states.append(last)
                if reason == "escape":
                    break
                # Do not miss a short F8 press between frame boundaries.
                if any(s.toggle for s in states) and not previous_toggle:
                    active = False
                    event("pause")
                previous_toggle = last.toggle
                keys, buttons = predominant_inputs(states)
                if initial.mouse_total_dx is not None and last.mouse_total_dx is not None:
                    dx = (last.mouse_total_dx - initial.mouse_total_dx) / window.width
                    dy = (last.mouse_total_dy - initial.mouse_total_dy) / window.height
                else:
                    if require_raw_mouse and has_raw_mouse:
                        raise DesktopError("Raw mouse event monitor stopped during recording")
                    dx, dy = (last.x - initial.x) / window.width, (last.y - initial.y) / window.height
                index, distance = match_action(actions, keys, buttons, dx, dy, max_mouse_error)
                raw = {"keys": list(keys), "buttons": list(buttons), "dx": dx, "dy": dy}
                if index is None:
                    dropped += 1
                    history_reset = True
                    event("unmapped", raw_input=raw, match_distance=distance if math.isfinite(distance) else None)
                    continue
                frame.thumbnail((max_side, max_side))
                frame_path = f"frames/{samples:07d}.jpg"
                target = output / frame_path
                frame.save(target, format="JPEG", quality=90)
                size_bytes += target.stat().st_size
                record = {"frame": frame_path, "action": index, "timestamp": captured_at,
                    "episode_id": episode_id, "raw_input": raw, "match_distance": distance,
                    "history_reset": history_reset,
                    "interval_seconds": time.monotonic() - interval_start,
                    "capture_started_at": capture_started_at,
                    "capture_ms": (captured_at - capture_started_at) * 1000,
                    "capture_to_label": "frame_start_then_predominant_inputs"}
                records.write(json.dumps(record) + "\n")
                history_reset = not active
                samples += 1
                action_counts[actions[index].name] += 1
                if size_bytes > max_disk_mb * 1_000_000:
                    reason = "disk_limit"
                    break
        except KeyboardInterrupt:
            reason = "keyboard_interrupt"
        except Exception as exc:
            reason = f"error: {exc}"
            event("error", message=str(exc))
        finally:
            stop_monitor = getattr(backend, "stop_mouse_monitor", None)
            if stop_monitor:
                stop_monitor()
        summary = {"episode_id": episode_id, "samples": samples, "dropped_unmapped": dropped,
            "action_counts": dict(action_counts), "duration_seconds": time.monotonic() - start,
            "requested_fps": fps, "image_max_side": max_side,
            "mouse_capture_mode": "raw_event_deltas" if has_raw_mouse else "absolute_cursor_fallback",
            "frame_bytes": size_bytes, "reason": reason, "source": "human_demonstration",
            "has_success_label": False}
        event("stop", **summary)
    (output / "episode.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Game config containing actions, or an action array")
    parser.add_argument("--output", type=Path, default=Path("demos") / time.strftime("episode-%Y%m%d-%H%M%S"))
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--max-side", type=int, default=384)
    parser.add_argument("--countdown", type=float, default=5)
    parser.add_argument("--start-immediately", action="store_true", help="Start after countdown without waiting for F8")
    parser.add_argument("--max-mouse-error", type=float, default=0.08)
    parser.add_argument("--max-disk-mb", type=float, default=1024)
    parser.add_argument("--allow-cursor-deltas", action="store_true", help="Allow absolute cursor fallback only for games without mouse locking/recentering")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and dependencies without capturing or sending input")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text()) if args.config else None
        actions = parse_actions(config) if config is not None else []
        if args.dry_run:
            print(json.dumps({**portable_diagnostics(), "action_count": len(actions), "mode": "record_dry_run"}, indent=2))
            return 0
        if not actions:
            parser.error("--config is required for recording")
        backend = MacDesktop()
        print(json.dumps(backend.diagnostics(), indent=2), flush=True)
        countdown(args.countdown)
        summary = record_episode(backend, actions, args.output, seconds=args.seconds, fps=args.fps,
            max_side=args.max_side, start_immediately=args.start_immediately,
            max_mouse_error=args.max_mouse_error, max_disk_mb=args.max_disk_mb,
            config=config if isinstance(config, dict) else None, require_raw_mouse=not args.allow_cursor_deltas)
        print(json.dumps(summary, indent=2))
        return 1 if summary["reason"].startswith("error") else 0
    except (DesktopError, ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
