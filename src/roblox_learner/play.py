"""Run a compact learned policy against the visible foreground Roblox window.

No Roblox code is injected. Escape, focus loss, memory limits, and the watchdog
release held controls. An episode log is evidence of actions, not proof of wins;
use the separate outcome evaluator to review saved frames against game criteria.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from .desktop import (
    Action, DesktopBackend, DesktopError, InputController, MacDesktop,
    ROBLOX_BUNDLE_ID, Watchdog, countdown, parse_actions, portable_diagnostics,
    process_rss_mb,
)


@dataclass(frozen=True)
class Prediction:
    action: int
    confidence: float
    inference_ms: float


class Policy(Protocol):
    def predict(self, image: Any) -> Prediction: ...


class TorchPolicy:
    """Frame-stacked compact CNN, batch size one, with no gradient allocations."""

    def __init__(self, checkpoint: Path, device: str = "cpu", threads: int = 2) -> None:
        import torch
        from .model import FrameStack, load_checkpoint

        if not checkpoint.is_file():
            raise ValueError(f"Checkpoint does not exist: {checkpoint}")
        if checkpoint.stat().st_size > 250_000_000:
            raise ValueError("Checkpoint exceeds the compact-policy 250 MB file limit")
        if device not in {"cpu", "mps"}:
            raise ValueError("Runtime supports cpu or mps only")
        if not 1 <= threads <= 16:
            raise ValueError("threads must be between 1 and 16")
        torch.set_num_threads(threads)
        if device == "mps":
            if not torch.backends.mps.is_available():
                raise ValueError("MPS is not available on this host")
            # Limit model-side unified-memory allocation. Full process resident
            # memory is measured independently by the runtime watchdog.
            recommended = torch.mps.recommended_max_memory()
            torch.mps.set_per_process_memory_fraction(min(0.8, 2_000_000_000 / recommended))
        self.model, self.metadata = load_checkpoint(checkpoint, device=device)
        self.model.eval()
        self.stack = FrameStack(image_size=self.model.config.image_size, stack_size=self.model.config.stack_size)
        self.device = device
        self.torch = torch

    def predict(self, image: Any) -> Prediction:
        start = time.perf_counter()
        tensor = self.stack.push(image).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            probabilities = self.torch.softmax(self.model(tensor), dim=-1)
            confidence, index = probabilities.max(dim=-1)
            # .item() synchronizes MPS so measured time includes completion.
            result = int(index.item()), float(confidence.item())
        return Prediction(*result, (time.perf_counter() - start) * 1000)

    def accounted_memory_mb(self) -> float:
        # Adding driver memory to RSS can double-count unified allocations, but
        # is conservative: an MPS policy cannot hide its budget outside RSS.
        accelerator = self.torch.mps.driver_allocated_memory() / 1_000_000 if self.device == "mps" else 0.0
        return process_rss_mb() + accelerator


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_episode(backend: DesktopBackend, policy: Policy, actions: Sequence[Action], output: Path,
                *, seconds: float = 120, fps: float = 10, memory_limit_mb: float = 5500,
                watchdog_seconds: float = 2, save_every: int = 10,
                evidence_on_error: bool = True, min_confidence: float = 0,
                model_path: str = "", model_sha256: str = "", config: dict[str, Any] | None = None,
                max_evidence_mb: float = 512) -> dict[str, Any]:
    """Finite desktop episode with injectable policy/backend for portable tests."""
    if not math.isfinite(seconds) or not 0 < seconds <= 86400:
        raise ValueError("seconds must be finite, positive, and at most 86400")
    if not math.isfinite(fps) or not 1 <= fps <= 60:
        raise ValueError("fps must be between 1 and 60")
    if not math.isfinite(memory_limit_mb) or not 100 <= memory_limit_mb <= 5500:
        raise ValueError("memory_limit_mb must be between 100 and 5500")
    if not math.isfinite(watchdog_seconds) or not 0.1 <= watchdog_seconds <= 10:
        raise ValueError("watchdog_seconds must be between 0.1 and 10")
    if save_every < 0 or not isinstance(save_every, int):
        raise ValueError("save_every must be a nonnegative integer")
    if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be in [0,1]")
    if not math.isfinite(max_evidence_mb) or max_evidence_mb <= 0:
        raise ValueError("max_evidence_mb must be finite and positive")
    if not actions:
        raise ValueError("A nonempty action vocabulary is required")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite an existing episode: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "frames").mkdir()
    episode_id = str(uuid.uuid4())
    started = time.monotonic()
    deadline = started + seconds
    steps = 0
    frame = None
    reason = "duration_limit"
    maximum_rss = process_rss_mb()
    accounted_memory = getattr(policy, "accounted_memory_mb", process_rss_mb)
    maximum_accounted = accounted_memory()
    evidence_bytes = 0
    cfg = config if isinstance(config, dict) else {}
    controller = InputController(backend)
    summary: dict[str, Any] = {}
    with (output / "episode.jsonl").open("w", buffering=1) as log:
        def write(kind: str, **fields: Any) -> None:
            log.write(json.dumps({"type": kind, "timestamp": time.time(),
                "elapsed_seconds": time.monotonic() - started, "episode_id": episode_id,
                **fields}, allow_nan=False) + "\n")

        def save_frame(image: Any, name: str) -> str:
            nonlocal evidence_bytes
            relative = f"frames/{name}"
            target = output / relative
            image.save(target, format="PNG" if name.endswith(".png") else "JPEG", **({} if name.endswith(".png") else {"quality": 88}))
            evidence_bytes += target.stat().st_size
            return relative

        write("start", model=model_path, model_sha256=model_sha256, frame_path="frames/start.png",
            target_bundle=ROBLOX_BUNDLE_ID, fps=fps, seconds_limit=seconds,
            memory_limit_mb=memory_limit_mb, watchdog_seconds=watchdog_seconds,
            game_id=cfg.get("game_id", cfg.get("name")),
            criterion_id=cfg.get("evaluation", {}).get("criterion_id"),
            actions=[a.to_dict() for a in actions], observation="external_window_pixels",
            outcome="unverified", live_input=True)
        try:
            with controller, Watchdog(controller, timeout=watchdog_seconds, memory_limit_mb=memory_limit_mb, rss=accounted_memory) as guard:
                controller.check()
                frame = backend.capture()
                start_frame = save_frame(frame, "start.png")
                write("initial_frame", frame_path=start_frame)
                while time.monotonic() < deadline:
                    guard.raise_if_stopped()
                    controller.check()
                    step_start = time.monotonic()
                    guard.heartbeat()
                    frame = backend.capture()
                    guard.raise_if_stopped()
                    prediction = policy.predict(frame)
                    guard.raise_if_stopped()
                    if not 0 <= prediction.action < len(actions):
                        raise DesktopError(f"Policy returned invalid action index {prediction.action}")
                    if not math.isfinite(prediction.confidence) or not 0 <= prediction.confidence <= 1:
                        raise DesktopError("Policy returned invalid confidence")
                    if time.monotonic() >= deadline:
                        break
                    action = actions[prediction.action]
                    abstained = prediction.confidence < min_confidence
                    if abstained:
                        controller.release_all()
                    else:
                        controller.apply(action)
                    rss = process_rss_mb()
                    maximum_rss = max(maximum_rss, rss)
                    accounted = accounted_memory()
                    maximum_accounted = max(maximum_accounted, accounted)
                    if accounted > memory_limit_mb:
                        raise DesktopError(f"Accounted process/accelerator memory exceeded {memory_limit_mb:.0f} MB")
                    frame_path = save_frame(frame, f"step-{steps:07d}.jpg") if save_every and steps % save_every == 0 else None
                    write("step", step=steps, action=prediction.action, action_name=action.name,
                        abstained=abstained, confidence=prediction.confidence,
                        inference_ms=prediction.inference_ms,
                        end_to_end_ms=(time.monotonic() - step_start) * 1000,
                        rss_mb=rss, accounted_memory_mb=accounted, frame_path=frame_path)
                    steps += 1
                    if evidence_bytes > max_evidence_mb * 1_000_000:
                        reason = "evidence_disk_limit"
                        break
                    # Action duration is a maximum control interval, and the
                    # deadline always wins. Escape/focus checks run at 100Hz.
                    period = action.duration if action.duration is not None else 1 / fps
                    until = min(deadline, time.monotonic() + period) if action.duration is not None else min(deadline, step_start + period)
                    while time.monotonic() < until:
                        controller.check()
                        guard.raise_if_stopped()
                        guard.heartbeat()
                        time.sleep(min(0.01, max(0, until - time.monotonic())))
        except KeyboardInterrupt:
            reason = "keyboard_interrupt"
            write("interrupted", reason=reason)
        except Exception as exc:
            reason = f"error: {exc}"
            error_frame = None
            if evidence_on_error and frame is not None:
                try:
                    error_frame = save_frame(frame, "error-last-observation.png")
                except Exception:
                    pass
            write("error", message=str(exc), frame_path=error_frame)
        finally:
            try:
                controller.release_all()
            except Exception as exc:
                reason += f"; cleanup_error: {exc}"
                write("error", message=f"Input cleanup failed: {exc}")
            terminal_frame = None
            terminal_is_current = False
            try:
                if backend.is_focused():
                    frame = backend.capture()
                    terminal_is_current = True
                if frame is not None:
                    terminal_frame = save_frame(frame, "stop.png")
            except Exception as exc:
                write("evidence_error", message=str(exc))
            summary = {"episode_id": episode_id, "reason": reason, "steps": steps,
                "duration_seconds": time.monotonic() - started,
                "max_rss_mb": maximum_rss, "frame_path": terminal_frame,
                "max_accounted_memory_mb": maximum_accounted,
                "terminal_frame_current": terminal_is_current,
                "outcome": "unverified", "evidence_bytes": evidence_bytes}
            write("stop", **summary)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--config", type=Path, help="Optional config; vocabulary must match checkpoint exactly")
    parser.add_argument("--output", type=Path, default=Path("logs") / time.strftime("episode-%Y%m%d-%H%M%S"))
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--countdown", type=float, default=5)
    parser.add_argument("--memory-limit-mb", type=float, default=5500)
    parser.add_argument("--watchdog-seconds", type=float, default=2)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--max-evidence-mb", type=float, default=512)
    parser.add_argument("--no-error-evidence", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Validate/infer a synthetic frame without live capture or input")
    parser.add_argument("--diagnose", action="store_true", help="Inspect macOS permissions/focus only; never requests permissions")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text()) if args.config else {}
        if args.diagnose:
            print(json.dumps(MacDesktop().diagnostics(), indent=2))
            return 0
        if args.dry_run and args.model is None:
            actions = parse_actions(config) if config else []
            print(json.dumps({**portable_diagnostics(), "action_count": len(actions), "mode": "play_dry_run"}, indent=2))
            return 0
        if args.model is None:
            parser.error("--model is required except for --dry-run or --diagnose")
        policy = TorchPolicy(args.model, args.device, args.threads)
        actions = parse_actions(policy.metadata.get("actions"))
        if len(actions) != policy.model.config.action_count:
            raise ValueError("Checkpoint action vocabulary does not match model output size")
        if config and parse_actions(config) != actions:
            raise ValueError("Config action vocabulary differs from checkpoint; action order/meaning must match")
        if args.dry_run:
            from PIL import Image
            prediction = policy.predict(Image.new("RGB", (384, 216), (60, 90, 120)))
            print(json.dumps({**portable_diagnostics(), "mode": "model_dry_run", "action_count": len(actions),
                "model_sha256": file_sha256(args.model), "synthetic_frame_inference_ms": prediction.inference_ms,
                "prediction": prediction.action, "confidence": prediction.confidence,
                "game_success_tested": False}, indent=2))
            return 0
        training = policy.metadata.get("training", {})
        if (training.get("method") != "behavior_cloning" or policy.metadata.get("epoch", 0) < 1
                or not policy.metadata.get("split", {}).get("train_episodes")):
            raise ValueError("Live play requires a checkpoint produced by the training command; random weights are not a trained agent")
        backend = MacDesktop()
        print(json.dumps(backend.diagnostics(), indent=2), flush=True)
        print("Learned policy episode; game success is unverified until outcome evidence is reviewed.", flush=True)
        countdown(args.countdown)
        summary = run_episode(backend, policy, actions, args.output, seconds=args.seconds, fps=args.fps,
            memory_limit_mb=args.memory_limit_mb, watchdog_seconds=args.watchdog_seconds,
            save_every=args.save_every, evidence_on_error=not args.no_error_evidence,
            min_confidence=args.min_confidence, model_path=str(args.model.resolve()),
            model_sha256=file_sha256(args.model), config=config, max_evidence_mb=args.max_evidence_mb)
        print(json.dumps(summary, indent=2))
        return 1 if summary["reason"].startswith("error") else 0
    except (DesktopError, ValueError, OSError, ImportError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
