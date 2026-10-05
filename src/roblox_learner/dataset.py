"""Episode-isolated screenshot/action datasets for real demonstrations."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

from PIL import Image
import torch
from torch.utils.data import Dataset

from .model import preprocess_frame


@dataclass(frozen=True)
class FrameRecord:
    frame: Path
    action: int
    timestamp: float
    history_reset: bool = False
    interval_seconds: float | None = None


@dataclass(frozen=True)
class Episode:
    name: str
    directory: Path
    records: tuple[FrameRecord, ...]
    manifest_digest: str
    content_digest: str


def load_actions(path: str | Path) -> list[dict[str, Any]]:
    """Read the same canonical finite input vocabulary used during playback."""
    from .desktop import parse_actions

    content = json.loads(Path(path).read_text())
    actions = parse_actions(content)
    if len(actions) < 2:
        raise ValueError("Action vocabulary must contain at least two actions")
    return [action.to_dict() for action in actions]


def discover_episodes(
    root: str | Path, action_count: int, actions: list[dict[str, Any]] | None = None
) -> list[Episode]:
    """Validate aligned actions.jsonl manifests and find their screenshot files.

    Each line contains ``frame`` (relative path), integer ``action``, and finite
    nondecreasing ``timestamp``. One directory is one independent episode.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    episodes: list[Episode] = []
    for manifest in sorted(root.rglob("actions.jsonl")):
        directory = manifest.parent
        recorded_vocabulary = directory / "actions.json"
        if actions is not None and recorded_vocabulary.is_file() and load_actions(recorded_vocabulary) != actions:
            raise ValueError(f"Episode action vocabulary differs from --actions: {recorded_vocabulary}")
        records: list[FrameRecord] = []
        manifest_bytes = manifest.read_bytes()
        previous_timestamp = -math.inf
        for line_number, line in enumerate(manifest_bytes.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            location = f"{manifest}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{location}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{location}: each record must be an object")
            action = record.get("action")
            if type(action) is not int or not 0 <= action < action_count:
                raise ValueError(f"{location}: action must be an integer in [0, {action_count})")
            timestamp = record.get("timestamp")
            if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
                raise ValueError(f"{location}: timestamp must be finite and numeric")
            if timestamp < previous_timestamp:
                raise ValueError(f"{location}: timestamps must be nondecreasing")
            previous_timestamp = timestamp
            relative = record.get("frame")
            if not isinstance(relative, str) or not relative:
                raise ValueError(f"{location}: frame must be a relative filename")
            frame = (directory / relative).resolve()
            if Path(relative).is_absolute() or not frame.is_relative_to(directory):
                raise ValueError(f"{location}: frame must stay inside its episode directory")
            if not frame.is_file():
                raise ValueError(f"{location}: frame does not exist: {relative}")
            history_reset = record.get("history_reset", False)
            if not isinstance(history_reset, bool):
                raise ValueError(f"{location}: history_reset must be boolean")
            interval_seconds = record.get("interval_seconds")
            if interval_seconds is not None and (type(interval_seconds) not in (float, int) or
                    not math.isfinite(interval_seconds) or interval_seconds <= 0):
                raise ValueError(f"{location}: interval_seconds must be finite and positive")
            records.append(FrameRecord(frame, action, float(timestamp), history_reset, interval_seconds))
        if not records:
            raise ValueError(f"Empty episode manifest: {manifest}")
        positive_gaps = [b.timestamp - a.timestamp for a, b in zip(records, records[1:])
                         if b.timestamp > a.timestamp]
        nominal_gap = statistics.median(positive_gaps) if positive_gaps else None
        records[0] = replace(records[0], history_reset=True)
        for index in range(1, len(records)):
            prior = records[index - 1]
            expected_gap = prior.interval_seconds or nominal_gap
            if expected_gap and records[index].timestamp - prior.timestamp > max(expected_gap * 1.8, expected_gap + 0.025):
                records[index] = replace(records[index], history_reset=True)
        # Include image content, not only labels, so modified screenshots cannot
        # silently change a resumed run. Ignore filenames/timestamps in this
        # second digest to recognize byte-identical copied demonstrations.
        content_hasher = hashlib.sha256()
        for record in records:
            image_hasher = hashlib.sha256()
            with record.frame.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    image_hasher.update(chunk)
            content_hasher.update(json.dumps([record.action, record.history_reset,
                                               image_hasher.hexdigest()]).encode())
        name = str(directory.relative_to(root))
        episodes.append(Episode(name, directory, tuple(records), hashlib.sha256(manifest_bytes).hexdigest(),
                                content_hasher.hexdigest()))
    if not episodes:
        raise ValueError(f"No episode actions.jsonl manifests found in {root}")
    return episodes


def split_episodes(
    episodes: list[Episode], validation_fraction: float = 0.2, seed: int = 7
) -> tuple[list[Episode], list[Episode]]:
    """Split whole episodes, so adjacent frames can never leak into validation."""
    if len(episodes) < 2:
        raise ValueError("At least two independent episodes are required for training and validation")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    groups: dict[str, list[Episode]] = {}
    for episode in episodes:
        groups.setdefault(episode.content_digest, []).append(episode)
    if len(groups) < 2:
        raise ValueError("At least two distinct episodes are required; exact copies cannot form a validation split")
    shuffled_groups = list(groups.values())
    random.Random(seed).shuffle(shuffled_groups)
    validation_count = max(1, min(len(shuffled_groups) - 1, round(len(shuffled_groups) * validation_fraction)))
    return ([episode for group in shuffled_groups[validation_count:] for episode in group],
            [episode for group in shuffled_groups[:validation_count] for episode in group])


def dataset_digest(episodes: list[Episode]) -> str:
    """Fingerprint labels, screenshots and episode membership for strict resume."""
    material = [(e.name, e.manifest_digest, e.content_digest) for e in sorted(episodes, key=lambda e: e.name)]
    return hashlib.sha256(json.dumps(material, separators=(",", ":")).encode()).hexdigest()


def refinement_split(
    episodes: list[Episode], previous_metadata: dict[str, Any]
) -> tuple[list[Episode], list[Episode]]:
    """Preserve a checkpoint's untouched holdout while adding training episodes.

    Every earlier held-out episode must still exist byte-for-byte. Known training
    episodes may be retained or omitted; every genuinely new episode is assigned
    to training. Exact copies of validation demonstrations are rejected, even if
    moved to another folder. Historical fingerprints persist across refinements.
    """
    split = previous_metadata.get("split", {})
    held_out = split.get("validation_episodes")
    prior_training = split.get("train_episodes")
    identities = previous_metadata.get("episode_fingerprints")
    if (not isinstance(held_out, list) or not held_out or not isinstance(prior_training, list) or
            not prior_training or not isinstance(identities, dict)):
        raise ValueError("Initialization checkpoint lacks an auditable episode split and fingerprints")
    if set(held_out) & set(prior_training):
        raise ValueError("Initialization checkpoint contains overlapping train and validation episodes")
    by_name = {episode.name: episode for episode in episodes}
    missing = set(held_out) - set(by_name)
    if missing:
        raise ValueError(f"Refinement must retain all prior validation episodes: {sorted(missing)}")
    for episode in episodes:
        if episode.name in identities:
            expected = identities[episode.name]
            if expected != {"manifest": episode.manifest_digest, "content": episode.content_digest}:
                raise ValueError(f"Previously recorded episode changed: {episode.name}; add corrections as new episodes")
    validation_hashes: set[str] = set()
    for name in held_out:
        if name not in identities:
            raise ValueError(f"Missing previous validation fingerprint: {name}")
        validation_hashes.add(identities[name]["content"])
    for episode in episodes:
        if episode.name not in held_out and episode.content_digest in validation_hashes:
            raise ValueError(f"Episode copies prior validation data into training: {episode.name}")
    train = [episode for episode in episodes if episode.name not in held_out]
    if not train:
        raise ValueError("Refinement needs training episodes in addition to its preserved holdout")
    return train, [by_name[name] for name in held_out]


class EpisodeDataset(Dataset):
    """Lazy, bounded-cache visual dataset; histories never cross episode edges."""

    def __init__(
        self,
        episodes: list[Episode],
        image_size: int = 96,
        stack_size: int = 4,
        cache_size: int = 256,
    ):
        if stack_size < 1 or cache_size < 0:
            raise ValueError("stack_size must be positive and cache_size nonnegative")
        self.episodes = list(episodes)
        self.image_size = image_size
        self.stack_size = stack_size
        self.cache_size = cache_size
        self.samples = [(episode_index, frame_index)
                        for episode_index, episode in enumerate(episodes)
                        for frame_index in range(len(episode.records))]
        self._cache: OrderedDict[Path, torch.Tensor] = OrderedDict()
        self._history_starts: list[list[int]] = []
        for episode in self.episodes:
            starts: list[int] = []
            latest_reset = 0
            for index, record in enumerate(episode.records):
                if record.history_reset:
                    latest_reset = index
                starts.append(latest_reset)
            self._history_starts.append(starts)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def labels(self) -> list[int]:
        return [self.episodes[e].records[f].action for e, f in self.samples]

    def _frame(self, path: Path) -> torch.Tensor:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        try:
            with Image.open(path) as image:
                tensor = preprocess_frame(image, self.image_size)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot decode demonstration image: {path}") from exc
        if self.cache_size:
            self._cache[path] = tensor
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return tensor

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        episode_index, frame_index = self.samples[index]
        episode = self.episodes[episode_index]
        history_start = self._history_starts[episode_index][frame_index]
        history = [self._frame(episode.records[max(history_start, offset)].frame)
                   for offset in range(frame_index - self.stack_size + 1, frame_index + 1)]
        return torch.cat(history, dim=0), torch.tensor(episode.records[frame_index].action, dtype=torch.long)
