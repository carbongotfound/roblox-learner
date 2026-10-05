"""Compact screen-only policies and safe, portable inference checkpoints.

The network consumes RGB screenshots, never Roblox process memory or game APIs.
Its logits describe a user-supplied finite vocabulary of ordinary input actions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn


CHECKPOINT_FORMAT = "roblox-learner.visual-policy"
CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class PolicyConfig:
    action_count: int
    stack_size: int = 4
    image_size: int = 96

    def __post_init__(self) -> None:
        if self.action_count < 2:
            raise ValueError("A policy requires at least two actions")
        if not 1 <= self.stack_size <= 16:
            raise ValueError("stack_size must be between 1 and 16")
        if not 32 <= self.image_size <= 512:
            raise ValueError("image_size must be between 32 and 512")


class CompactPolicy(nn.Module):
    """Small convolutional behavior-cloning policy with temporal frame stacking."""

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        self.features = nn.Sequential(
            nn.Conv2d(3 * config.stack_size, 16, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(96 * 3 * 3, 192), nn.ReLU(inplace=True),
            nn.Linear(192, config.action_count),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Return unnormalized action logits for [batch, stacked RGB, H, W]."""
        return self.classifier(self.features(frames))


def preprocess_frame(frame: Image.Image | np.ndarray, image_size: int = 96) -> torch.Tensor:
    """Resize one RGB screenshot to a contiguous [3, H, W] float32 tensor.

    Full-frame resizing is intentional. Training and playback share this final
    transform; recordings may already be JPEG-compressed or downsampled. NumPy
    inputs are RGB (not OpenCV BGR), uint8 H x W x 3/4.
    """
    if not isinstance(frame, Image.Image):
        array = np.asarray(frame)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError("Frame arrays must be uint8 RGB/RGBA with shape H x W x 3/4")
        frame = Image.fromarray(array)
    resized = frame.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    pixels = np.array(resized, dtype=np.uint8, copy=True)
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous().float().div_(255.0)


class FrameStack:
    """Causal frame history, padding a new episode with its first observation."""

    def __init__(self, image_size: int = 96, stack_size: int = 4):
        if stack_size < 1:
            raise ValueError("stack_size must be positive")
        self.image_size = image_size
        self.stack_size = stack_size
        self._frames: deque[torch.Tensor] = deque(maxlen=stack_size)

    def reset(self) -> None:
        self._frames.clear()

    def push(self, frame: Image.Image | np.ndarray) -> torch.Tensor:
        current = preprocess_frame(frame, self.image_size)
        if not self._frames:
            self._frames.extend([current] * self.stack_size)
        else:
            self._frames.append(current)
        return torch.cat(tuple(self._frames), dim=0)


def resolve_device(requested: str = "auto") -> torch.device:
    """Prefer Apple Metal on supported Macs; CPU remains a complete fallback."""
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    if requested not in ("cpu", "mps", "cuda"):
        raise ValueError("device must be auto, cpu, mps or cuda")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this Python/PyTorch installation")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this Python/PyTorch installation")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def save_checkpoint(
    path: str | Path,
    model: CompactPolicy,
    metadata: dict[str, Any] | None = None,
    training_state: dict[str, Any] | None = None,
) -> None:
    """Atomically save tensor weights and JSON metadata, with optional resume state.

    Inference artifacts omit ``training_state``; optimizer buffers are unnecessary
    for standalone playback. No model instances or custom Python classes are saved.
    """
    clean_metadata = json.loads(json.dumps(metadata or {}, allow_nan=False))
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "config": asdict(model.config),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metadata": clean_metadata,
    }
    if training_state is not None:
        payload["training_state"] = training_state
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a local checkpoint without allowing arbitrary pickle object execution."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Not a roblox-learner visual-policy checkpoint")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version: {payload.get('version')}")
    for key in ("config", "state_dict", "metadata"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"Checkpoint has invalid {key}")
    return payload


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[CompactPolicy, dict[str, Any]]:
    """Restore an eval-mode standalone policy and its JSON-compatible metadata."""
    payload = read_checkpoint(path)
    model = CompactPolicy(PolicyConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(resolve_device(str(device))).eval()
    actions = payload["metadata"].get("actions")
    if actions is not None and len(actions) != model.config.action_count:
        raise ValueError("Checkpoint action vocabulary does not match output dimension")
    return model, payload["metadata"]
