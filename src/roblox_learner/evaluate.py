"""Audit externally observed play; never infer a game win from action counts.

This is evidence bookkeeping, not an image-based win detector. A verified win
means an identified reviewer marked visible success and every required evidence
file is present. Image meaning and reviewer honesty remain outside this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
from typing import Any


class EvaluationError(ValueError):
    """Malformed or ambiguous evidence must not silently change denominators."""


def wilson_interval(wins: int, trials: int, z: float = 1.959963984540054) -> list[float] | None:
    """Two-sided 95% Wilson binomial interval (None when no trials exist)."""
    if not isinstance(wins, int) or not isinstance(trials, int) or trials < 0 or not 0 <= wins <= trials:
        raise ValueError("Require integer 0 <= wins <= trials")
    if trials == 0:
        return None
    p = wins / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"{path}:{number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise EvaluationError(f"{path}:{number}: expected a JSON object")
            records.append(value)
    return records


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def _image_evidence(path: Path) -> dict[str, Any]:
    """Validate decoding, not semantics. Pillow is shared with the runtime."""
    result: dict[str, Any] = {"path": str(path), "valid": False}
    try:
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        if width < 16 or height < 16:
            result["error"] = "image smaller than 16x16"
            return result
        result.update(valid=True, width=width, height=height,
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    except (OSError, ValueError, ImportError) as exc:
        result["error"] = str(exc)
    return result


def _resolve_frame(log_path: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (log_path.parent / path).resolve()


def _episode(log_path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    ids = {str(row["episode_id"]) for row in records if row.get("episode_id")}
    if len(ids) != 1:
        raise EvaluationError(f"{log_path}: require exactly one nonempty episode_id")
    episode_id = ids.pop()
    if any(str(row.get("episode_id", "")) != episode_id for row in records):
        raise EvaluationError(f"{log_path}: every event must identify its episode")
    starts = [row for row in records if row.get("type") == "start"]
    stops = [row for row in records if row.get("type") == "stop"]
    if len(starts) > 1 or len(stops) > 1:
        raise EvaluationError(f"{log_path}: repeated start or stop event")
    start = starts[0] if starts else {}
    stop = stops[0] if stops else {}
    steps = [row for row in records if row.get("type") == "step"]
    timestamps = [_number(row.get("timestamp")) for row in records]
    known_times = [value for value in timestamps if value is not None]
    timestamp_ordered = all(a <= b for a, b in zip(known_times, known_times[1:]))
    duration = _number(stop.get("duration"))
    if duration is None:
        duration = _number(stop.get("elapsed_seconds"))
    if duration is None and len(known_times) > 1:
        duration = max(0.0, known_times[-1] - known_times[0])
    duration = max(0.0, duration or 0.0)
    frame_paths = {_resolve_frame(log_path, row.get("frame_path")) for row in records}
    frame_paths.discard(None)
    return {
        "episode_id": episode_id, "log_path": str(log_path.resolve()),
        "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
        "start": start, "stop": stop, "steps": steps,
        "errors": [row for row in records if row.get("type") == "error"],
        "duration_seconds": duration, "frame_paths": frame_paths,
        "timestamp_ordered": timestamp_ordered,
        "valid_timestamps": bool(timestamps) and all(value is not None for value in timestamps),
        "complete_log": bool(starts and stops and steps and records[0].get("type") == "start"
                             and records[-1].get("type") == "stop"),
    }


def _review(episode: dict[str, Any], outcome: dict[str, Any] | None, criterion_id: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "episode_id": episode["episode_id"], "log_path": episode["log_path"],
        "log_sha256": episode["log_sha256"], "duration_seconds": episode["duration_seconds"],
        "steps": len(episode["steps"]), "reported_outcome": "unannotated",
        "runtime_stop_reason": episode["stop"].get("reason"),
        "verified_win": False, "reasons": [], "evidence": [],
    }
    if outcome is None:
        result["reasons"].append("no outcome annotation")
        return result
    result["reported_outcome"] = outcome["outcome"]
    result["criterion_id"] = outcome.get("criterion_id")
    result["reviewed_by"] = outcome.get("reviewed_by")
    if outcome["outcome"] != "win":
        return result
    reasons = result["reasons"]
    if not episode["complete_log"]:
        reasons.append("missing complete start/step/stop log")
    if not episode["timestamp_ordered"]:
        reasons.append("timestamps are out of order")
    if not episode["valid_timestamps"]:
        reasons.append("event timestamps must all be finite numbers")
    if episode["errors"]:
        reasons.append("runtime error recorded")
    if not isinstance(outcome.get("reviewed_by"), str) or not outcome["reviewed_by"].strip():
        reasons.append("no identified visual reviewer")
    if outcome.get("success_visible") is not True:
        reasons.append("reviewer did not confirm visible success")
    if outcome.get("unassisted") is not True or outcome.get("interventions") != 0 or isinstance(outcome.get("interventions"), bool):
        reasons.append("zero interventions and unassisted=true are required")
    if not isinstance(outcome.get("criterion_id"), str) or not outcome["criterion_id"].strip():
        reasons.append("no success criterion")
    elif criterion_id and outcome["criterion_id"] != criterion_id:
        reasons.append("annotation uses a different success criterion")
    start_criterion = episode["start"].get("criterion_id")
    if start_criterion and outcome.get("criterion_id") != start_criterion:
        reasons.append("criterion differs from the recorded trial")
    model_hash = episode["start"].get("model_sha256")
    if not isinstance(model_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", model_hash):
        reasons.append("log lacks a valid model_sha256")
    elif outcome.get("checkpoint_sha256") != model_hash:
        reasons.append("annotation checkpoint does not match play log")
    log_path = Path(episode["log_path"])
    start_frame = _resolve_frame(log_path, episode["start"].get("frame_path"))
    terminal_frame = _resolve_frame(log_path, outcome.get("terminal_frame"))
    if start_frame is None:
        reasons.append("log lacks initial-state screenshot")
    if terminal_frame is None:
        reasons.append("annotation lacks terminal screenshot")
    elif terminal_frame not in episode["frame_paths"]:
        reasons.append("terminal screenshot was not referenced by this play log")
    stop_frame = _resolve_frame(log_path, episode["stop"].get("frame_path"))
    if terminal_frame is not None and terminal_frame == stop_frame:
        if episode["stop"].get("terminal_frame_current") is not True:
            reasons.append("stop screenshot is not confirmed as a fresh terminal capture")
    if terminal_frame is not None and terminal_frame == start_frame:
        reasons.append("initial and terminal screenshots must be distinct")
    for role, frame in (("initial", start_frame), ("terminal", terminal_frame)):
        if frame is not None:
            evidence = _image_evidence(frame)
            evidence["role"] = role
            result["evidence"].append(evidence)
            if not evidence["valid"]:
                reasons.append(f"{role} screenshot is missing or undecodable")
    if len(result["evidence"]) == 2 and all(item["valid"] for item in result["evidence"]):
        if result["evidence"][0]["sha256"] == result["evidence"][1]["sha256"]:
            reasons.append("initial and terminal screenshots have identical bytes")
        expected_hash = outcome.get("terminal_sha256")
        if expected_hash != result["evidence"][1]["sha256"]:
            reasons.append("annotation lacks matching terminal_sha256")
    result["verified_win"] = not reasons
    return result


def evaluate(log_paths: list[Path], outcomes_path: Path | None = None,
             criterion_id: str | None = None) -> dict[str, Any]:
    """Count every supplied trial, including crashes, missing reviews and losses."""
    episodes: dict[str, dict[str, Any]] = {}
    for log_path in log_paths:
        episode = _episode(log_path, read_jsonl(log_path))
        if episode["episode_id"] in episodes:
            raise EvaluationError(f"duplicate episode_id: {episode['episode_id']}")
        episodes[episode["episode_id"]] = episode
    outcomes: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(outcomes_path) if outcomes_path else []:
        episode_id = record.get("episode_id")
        if not isinstance(episode_id, str) or episode_id not in episodes:
            raise EvaluationError(f"outcome references unknown episode_id: {episode_id!r}")
        if episode_id in outcomes:
            raise EvaluationError(f"duplicate outcome for {episode_id}")
        if record.get("outcome") not in {"win", "loss", "incomplete"}:
            raise EvaluationError(f"{episode_id}: outcome must be win, loss or incomplete")
        outcomes[episode_id] = record
    reviews = [_review(episode, outcomes.get(key), criterion_id) for key, episode in episodes.items()]
    trials = len(episodes)
    reported = sum(item["reported_outcome"] == "win" for item in reviews)
    verified = sum(item["verified_win"] for item in reviews)
    steps = [step for episode in episodes.values() for step in episode["steps"]]
    durations = [episode["duration_seconds"] for episode in episodes.values()]
    metrics = {}
    for field in ("inference_ms", "end_to_end_ms", "rss_mb"):
        values = [value for step in steps if (value := _number(step.get(field))) is not None and value >= 0]
        metrics[field] = {"samples": len(values), "median": median(values) if values else None,
                          "p95": _percentile(values, 0.95), "max": max(values) if values else None}
    checkpoint_hashes = sorted({str(ep["start"].get("model_sha256")) for ep in episodes.values()
                                if ep["start"].get("model_sha256")})
    criteria = sorted({str(row.get("criterion_id")) for row in outcomes.values() if row.get("criterion_id")})
    warnings = ["Visual success is a reviewer attestation; this evaluator does not recognize wins in pixels.",
                "Confidence intervals assume independent trials; repeated saved states may violate that assumption.",
                "RSS measurements cover the policy process only, not Roblox or total macOS memory."]
    if len(checkpoint_hashes) > 1:
        warnings.append("Multiple checkpoints are mixed; evaluate each checkpoint separately before comparing policies.")
    if len(criteria) > 1:
        warnings.append("Multiple success criteria are mixed; the combined rate is not a single-task benchmark.")
    return {
        "schema_version": 1, "status": "no_trials" if not trials else "audited",
        "criterion_id": criterion_id, "trials": trials,
        "reported_wins": reported, "verified_wins": verified,
        "unannotated_trials": sum(item["reported_outcome"] == "unannotated" for item in reviews),
        "reported_success_rate": reported / trials if trials else None,
        "verified_success_rate": verified / trials if trials else None,
        "verified_success_wilson_95": wilson_interval(verified, trials),
        "total_play_seconds": sum(durations), "longest_episode_seconds": max(durations) if durations else None,
        "longest_verified_success_episode_seconds": max(
            (row["duration_seconds"] for row in reviews if row["verified_win"]), default=None),
        "total_actions": len(steps),
        "observed_actions_per_second": len(steps) / sum(durations) if sum(durations) > 0 else None,
        "performance": metrics, "checkpoint_sha256": checkpoint_hashes,
        "warnings": warnings, "episodes": reviews,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, nargs="+", required=True, help="One episode JSONL file per trial")
    parser.add_argument("--outcomes", type=Path, help="Reviewer outcome annotations; omitted means no verified wins")
    parser.add_argument("--criterion-id", help="Only this exact criterion can count as a verified win")
    parser.add_argument("--output", type=Path, help="Save JSON report (also printed to stdout)")
    args = parser.parse_args(argv)
    try:
        report = evaluate(args.logs, args.outcomes, args.criterion_id)
    except (EvaluationError, OSError) as exc:
        parser.error(str(exc))
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
