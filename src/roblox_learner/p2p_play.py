"""Finite, recorded external Roblox trials using pretrained Pixel2Play."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch

from .desktop import InputController, MacDesktop, Watchdog, DesktopError, countdown, process_rss_mb
from .p2p import P2PPolicy, decode_actions, CHECKPOINT_SHA256


class P2PController(InputController):
    """Preserve simultaneous aiming/firing buttons as well as held movement."""
    def __init__(self, backend):
        super().__init__(backend)
        self.buttons = set()

    def apply_tokens(self, tokens, mouse_scale=1.0):
        action = decode_actions(tokens)
        if not math.isfinite(mouse_scale) or not 0 < mouse_scale <= 2:
            raise ValueError("mouse_scale must be in (0,2]")
        with self._lock:
            self.check()
            keys, buttons = set(action["keys"]), set(action["buttons"])
            for key in sorted(self.held_keys - keys):
                self.backend.key(key, False)
                self.held_keys.discard(key)
            for button in sorted(self.buttons - buttons):
                self.backend.mouse_button(button, False)
                self.buttons.discard(button)
            for key in sorted(keys - self.held_keys):
                self.check()
                self.backend.key(key, True)
                self.held_keys.add(key)
            for button in sorted(buttons - self.buttons):
                self.check()
                self.backend.mouse_button(button, True)
                self.buttons.add(button)
            self.check()
            window = self.backend.window()
            self.backend.mouse_move(action["mouse_dx_pixels"] * mouse_scale / window.width,
                                    action["mouse_dy_pixels"] * mouse_scale / window.height)
        return action

    def release_all(self):
        errors = []
        with self._lock:
            try:
                super().release_all()
            except Exception as exc:
                errors.append(exc)
            for button in tuple(self.buttons):
                try:
                    self.backend.mouse_button(button, False)
                    self.buttons.discard(button)
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise DesktopError(f"Input cleanup failed: {errors[0]}")


def run_trial(backend, policy, output, *, seconds=30, fps=15, mouse_scale=1.0, live=False):
    if not math.isfinite(seconds) or not 0 < seconds <= 600:
        raise ValueError("seconds must be in (0,600]")
    if not math.isfinite(fps) or not 1 <= fps <= 30:
        raise ValueError("fps must be in [1,30]")
    if not math.isfinite(mouse_scale) or not 0 < mouse_scale <= 2:
        raise ValueError("mouse_scale must be in (0,2]")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Refusing to overwrite existing trial")
    frames = output / "frames"
    frames.mkdir()
    controller = P2PController(backend)
    def memory():
        return process_rss_mb() + (torch.mps.driver_allocated_memory() / 1e6 if policy.device == "mps" else 0)
    controller.check()
    frame = backend.capture()
    frame.save(output / "start.png")
    # Compile/warm device kernels before holding any controls.
    for _ in range(3):
        controller.check()
        policy.predict(frame)
    policy.reset()
    start = time.monotonic()
    last_evidence = -1
    steps, timings, peak_memory, reason = 0, [], memory(), "duration_limit"
    evidence_count = 0
    with (output / "actions.jsonl").open("w", buffering=1) as log:
        try:
            with controller, Watchdog(controller, timeout=2, memory_limit_mb=5500, rss=memory) as guard:
                while time.monotonic() - start < seconds:
                    begin = time.monotonic()
                    guard.raise_if_stopped()
                    controller.check()
                    guard.heartbeat()
                    frame = backend.capture()
                    captured = time.monotonic()
                    prediction = policy.predict(frame)
                    predicted = time.monotonic()
                    guard.raise_if_stopped()
                    controller.check()
                    if time.monotonic() - start >= seconds:
                        break
                    if live:
                        controller.apply_tokens(prediction["tokens"], mouse_scale)
                    latency = (time.monotonic() - begin) * 1000
                    timings.append(latency)
                    peak_memory = max(peak_memory, memory())
                    if peak_memory > 5500:
                        raise DesktopError("Agent memory exceeded 5500MB")
                    elapsed = time.monotonic() - start
                    frame_path = None
                    if elapsed - last_evidence >= .2:
                        relative = f"frames/{evidence_count:06d}.jpg"
                        evidence = frame.copy()
                        evidence.thumbnail((960, 540))
                        evidence.save(output / relative, quality=80)
                        frame_path = relative
                        evidence_count += 1
                        last_evidence = elapsed
                    log.write(json.dumps({"step": steps, "elapsed_seconds": elapsed,
                        "capture_to_control_ms": latency, "prediction": prediction,
                        "capture_ms": (captured - begin) * 1000,
                        "policy_ms": (predicted - captured) * 1000,
                        "control_ms": latency - (predicted - begin) * 1000,
                        "live_input": live, "frame_path": frame_path}) + "\n")
                    steps += 1
                    if steps % 30 == 0:
                        print(json.dumps({"steps": steps, "elapsed_seconds": elapsed,
                            "decisions_per_second": steps / elapsed, "memory_mb": peak_memory}), flush=True)
                    while time.monotonic() < min(start + seconds, begin + 1 / fps):
                        controller.check()
                        guard.raise_if_stopped()
                        time.sleep(.002)
        except (Exception, KeyboardInterrupt) as exc:
            reason = f"stopped: {type(exc).__name__}: {exc}"
        finally:
            controller.release_all()
            if backend.is_focused():
                backend.capture().save(output / "stop.png")
    duration = time.monotonic() - start
    report = {"model": "Open Pixel2Play 150M (experimental Mac port)", "checkpoint_sha256": CHECKPOINT_SHA256,
        "live_input": live, "steps": steps, "duration_seconds": duration,
        "decisions_per_second": steps / duration, "max_conservative_agent_memory_mb": peak_memory,
        "capture_to_control_p50_ms": float(np.percentile(timings, 50)) if timings else None,
        "capture_to_control_p95_ms": float(np.percentile(timings, 95)) if timings else None,
        "reason": reason, "outcome": "unverified", "evidence_frames": evidence_count,
        "latency_excludes": ["game response/rendering", "network"], "recording": "timestamped JPEG frames and action log"}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--countdown", type=float, default=8)
    parser.add_argument("--history", type=int, default=200)
    parser.add_argument("--mouse-scale", type=float, default=1)
    parser.add_argument("--live", action="store_true", help="Send predicted controls; default only observes")
    args = parser.parse_args(argv)
    def stop(*_): raise KeyboardInterrupt("Termination requested")
    signal.signal(signal.SIGTERM, stop)
    policy = P2PPolicy(args.checkpoint, "mps", history=args.history)
    countdown(args.countdown)
    report = run_trial(MacDesktop(), policy, args.output, seconds=args.seconds, fps=args.fps,
                       mouse_scale=args.mouse_scale, live=args.live)
    return int(report["reason"].startswith("stopped"))


if __name__ == "__main__":
    raise SystemExit(main())
