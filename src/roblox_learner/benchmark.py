"""Measure batch-one policy latency and actual process memory on this machine.

This checks runtime cost only. Random benchmark pixels are never demonstrations,
and inference speed does not establish parkour or game completion ability.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch

from .model import FrameStack, load_checkpoint, resolve_device, synchronize


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _latencies(values: list[float]) -> dict[str, float]:
    p50 = _percentile(values, 50)
    p95 = _percentile(values, 95)
    return {"p50_ms": p50, "p95_ms": p95, "mean_ms": statistics.mean(values),
            "max_ms": max(values), "p95_equivalent_fps": 1000.0 / p95}


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _current_rss_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return None


def _cpu_name() -> str:
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine()


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.iterations < 1 or args.warmup < 0 or args.threads < 1 or args.fps_target <= 0:
        raise ValueError("iterations, threads and fps-target must be positive; warmup must be nonnegative")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    before_load_rss = _current_rss_bytes()
    model, metadata = load_checkpoint(args.checkpoint, device)
    if args.image:
        with Image.open(args.image) as source:
            frame = source.convert("RGB").copy()
        input_source = "provided screenshot"
    else:
        # Workload pixels only; no labels, optimization or gameplay claims.
        pixels = np.random.default_rng(7).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        frame = Image.fromarray(pixels)
        input_source = "random workload pixels (not a gameplay evaluation)"
    stack = FrameStack(model.config.image_size, model.config.stack_size)
    sample = stack.push(frame).unsqueeze(0).to(device)
    inference_times: list[float] = []
    pipeline_times: list[float] = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(sample).argmax(dim=1).cpu().item()
        synchronize(device)
        for _ in range(args.iterations):
            started = time.perf_counter_ns()
            model(sample).argmax(dim=1).cpu().item()
            synchronize(device)
            inference_times.append((time.perf_counter_ns() - started) / 1e6)
        for _ in range(args.iterations):
            started = time.perf_counter_ns()
            sample = stack.push(frame).unsqueeze(0).to(device)
            model(sample).argmax(dim=1).cpu().item()
            synchronize(device)
            pipeline_times.append((time.perf_counter_ns() - started) / 1e6)
    memory: dict[str, Any] = {
        "process_rss_before_checkpoint_bytes": before_load_rss,
        "process_rss_after_benchmark_bytes": _current_rss_bytes(),
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "rss_scope": "this benchmark Python process, including imported libraries; excludes Roblox",
    }
    device_memory = 0
    if device.type == "mps":
        memory["mps_allocated_bytes"] = int(torch.mps.current_allocated_memory())
        memory["mps_driver_allocated_bytes"] = int(torch.mps.driver_allocated_memory())
        device_memory = memory["mps_driver_allocated_bytes"]
    elif device.type == "cuda":
        memory["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        device_memory = memory["cuda_peak_allocated_bytes"]
    conservative_memory = memory["process_peak_rss_bytes"] + device_memory
    memory["conservative_process_plus_device_bytes"] = conservative_memory
    memory["under_6_decimal_gb"] = conservative_memory < 6_000_000_000
    memory["device_accounting_note"] = "Device bytes are added conservatively; unified memory may overlap RSS."
    pipeline_summary = _latencies(pipeline_times)
    report = {
        "schema_version": 1,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_bytes": Path(args.checkpoint).stat().st_size,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_bytes_float32": sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()),
        "hardware": {"cpu": _cpu_name(), "machine": platform.machine(), "platform": platform.platform(),
                     "logical_cpu_count": os.cpu_count(), "device": str(device), "torch_threads": args.threads,
                     "python": platform.python_version(), "torch": str(torch.__version__)},
        "input": {"source": input_source, "screenshot_size": list(frame.size),
                  "policy_image_size": model.config.image_size, "stack_size": model.config.stack_size,
                  "action_count": model.config.action_count, "batch_size": 1},
        "iterations_per_measurement": args.iterations, "warmup_iterations": args.warmup,
        "inference_and_action_selection": _latencies(inference_times),
        "preprocessing_inference_and_action_selection": pipeline_summary,
        "memory": memory,
        "target_fps": args.fps_target,
        "meets_policy_compute_budget_at_p95": pipeline_summary["p95_ms"] <= 1000.0 / args.fps_target,
        "excluded_from_latency": ["screen capture", "input injection", "Roblox rendering and networking"],
        "gameplay_success": "NOT EVALUATED",
        "training_epoch": metadata.get("epoch"),
    }
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--fps-target", type=float, default=20)
    parser.add_argument("--image", help="Optional real screenshot to use as the inference workload")
    parser.add_argument("--output", help="Optional JSON report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        benchmark(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Benchmark error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
