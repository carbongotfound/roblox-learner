"""Unit fixtures validate mechanics only; they do not establish gameplay skill."""

import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest
import torch

from roblox_learner.dataset import EpisodeDataset, dataset_digest, discover_episodes, load_actions, refinement_split, split_episodes
from roblox_learner.model import CompactPolicy, FrameStack, PolicyConfig, load_checkpoint, save_checkpoint
from roblox_learner.train import classification_metrics, main as train_main


@pytest.fixture(autouse=True)
def bound_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _episode(root: Path, name: str, colors=None):
    mark_episode = colors is None
    if colors is None:
        colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
    directory = root / name
    (directory / "frames").mkdir(parents=True)
    records = []
    for index, color in enumerate(colors):
        filename = f"frames/{index:06d}.png"
        image = Image.new("RGB", (40, 30), color)
        if mark_episode:
            image.putpixel((0, 0), (sum(name.encode()) % 256, len(name) % 256, index))
        image.save(directory / filename)
        records.append({"frame": filename, "action": index % 2, "timestamp": 100.0 + index / 10})
    (directory / "actions.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return directory


def test_frame_stack_is_causal_and_resets():
    frames = FrameStack(image_size=32, stack_size=2)
    red = np.full((40, 50, 3), (255, 0, 0), dtype=np.uint8)
    green = Image.new("RGB", (50, 40), (0, 255, 0))
    first = frames.push(red)
    second = frames.push(green)
    assert first.shape == (6, 32, 32)
    assert torch.equal(first[:3], first[3:])
    assert torch.equal(second[:3], first[:3])
    assert second[4].eq(1).all()
    frames.reset()
    reset = frames.push(green)
    assert torch.equal(reset[:3], reset[3:])


def test_episode_split_never_mixes_neighboring_frames(tmp_path):
    for name in ("one", "two", "three", "four"):
        _episode(tmp_path, name)
    episodes = discover_episodes(tmp_path, 2)
    train, validation = split_episodes(episodes, 0.25, seed=3)
    assert {e.name for e in train}.isdisjoint({e.name for e in validation})
    assert len(train) == 3 and len(validation) == 1
    again = split_episodes(episodes, 0.25, seed=3)
    assert [e.name for e in train] == [e.name for e in again[0]]
    with pytest.raises(ValueError, match="At least two"):
        split_episodes(episodes[:1])


def test_dataset_history_matches_runtime_and_never_crosses_episodes(tmp_path):
    _episode(tmp_path, "one")
    _episode(tmp_path, "two", colors=((255, 255, 255), (0, 0, 0)))
    dataset = EpisodeDataset(discover_episodes(tmp_path, 2), image_size=32, stack_size=4, cache_size=2)
    runtime_stack = FrameStack(image_size=32, stack_size=4)
    for index in range(3):
        with Image.open(tmp_path / "one" / "frames" / f"{index:06d}.png") as frame:
            assert torch.equal(dataset[index][0], runtime_stack.push(frame))
    first_second_episode, target = dataset[3]
    assert first_second_episode.eq(1).all()
    assert target.item() == 0
    assert len(dataset._cache) <= 2


@pytest.mark.parametrize("change, message", [
    ({"action": 2}, "action must be"),
    ({"action": True}, "action must be"),
    ({"timestamp": float("nan")}, "timestamp must be"),
    ({"frame": "../outside.png"}, "inside its episode"),
    ({"frame": "frames/missing.png"}, "does not exist"),
])
def test_bad_demonstrations_fail_with_context(tmp_path, change, message):
    directory = _episode(tmp_path, "one")
    manifest = directory / "actions.jsonl"
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    records[0].update(change)
    manifest.write_text("\n".join(json.dumps(r) for r in records))
    with pytest.raises(ValueError, match=message):
        discover_episodes(tmp_path, 2)


def test_checkpoint_roundtrip_is_standalone_and_action_aligned(tmp_path):
    model = CompactPolicy(PolicyConfig(action_count=3, stack_size=2, image_size=32)).eval()
    frames = torch.rand(2, 6, 32, 32)
    checkpoint = tmp_path / "policy.pt"
    actions = [{"name": name, "keys": []} for name in ("idle", "forward", "jump")]
    save_checkpoint(checkpoint, model, {"actions": actions, "epoch": 3})
    restored, metadata = load_checkpoint(checkpoint)
    with torch.inference_mode():
        assert torch.equal(model(frames), restored(frames))
    assert metadata["actions"] == actions
    assert restored.training is False
    assert checkpoint.stat().st_size < 5_000_000
    payload = torch.load(checkpoint, weights_only=True)
    assert "training_state" not in payload
    save_checkpoint(checkpoint, model, {"actions": actions[:2]})
    with pytest.raises(ValueError, match="vocabulary"):
        load_checkpoint(checkpoint)


def test_balanced_accuracy_does_not_hide_ignored_rare_actions():
    metrics = classification_metrics(torch.tensor([[90, 0], [10, 0]]))
    assert metrics["accuracy"] == 0.9
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["per_action_recall"] == [1.0, 0.0]


def test_training_saves_resume_and_episode_isolated_metrics(tmp_path):
    # Tiny generated fixtures exercise I/O and optimizer continuation only.
    dataset = tmp_path / "unit-fixtures"
    for name in ("one", "two", "three"):
        _episode(dataset, name)
    vocabulary = tmp_path / "actions.json"
    vocabulary.write_text(json.dumps({"actions": [{"name": "idle", "keys": []},
                                                {"name": "forward", "keys": ["w"]}]}))
    assert len(load_actions(vocabulary)) == 2
    output = tmp_path / "policy.pt"
    arguments = ["--dataset", str(dataset), "--actions", str(vocabulary), "--output", str(output),
                 "--device", "cpu", "--image-size", "32", "--stack-size", "2", "--threads", "1",
                 "--batch-size", "3", "--epochs", "1"]
    assert train_main(arguments) == 0
    first = json.loads(output.with_name("policy.metrics.json").read_text())
    assert first["gameplay_success"] == "NOT EVALUATED"
    assert first["completed_epochs"] == 1
    assert set(first["split"]["train_episodes"]).isdisjoint(first["split"]["validation_episodes"])
    arguments[-1] = "2"
    assert train_main(arguments + ["--resume", str(output.with_name("policy.last.pt"))]) == 0
    resumed = json.loads(output.with_name("policy.metrics.json").read_text())
    assert resumed["completed_epochs"] == 2
    assert len(resumed["epochs"]) == 2
    assert resumed["split"] == first["split"]


def _split_metadata(episodes):
    return {"split": {"train_episodes": [e.name for e in episodes[:-1]],
                      "validation_episodes": [episodes[-1].name]},
            "episode_fingerprints": {e.name: {"manifest": e.manifest_digest, "content": e.content_digest}
                                     for e in episodes}}


def test_refinement_preserves_holdout_and_puts_new_episodes_only_in_training(tmp_path):
    for name in ("one", "two", "three"):
        _episode(tmp_path, name)
    original = discover_episodes(tmp_path, 2)
    previous = _split_metadata(original)
    _episode(tmp_path, "new-corrective-demo")
    train, validation = refinement_split(discover_episodes(tmp_path, 2), previous)
    assert [e.name for e in validation] == previous["split"]["validation_episodes"]
    assert "new-corrective-demo" in {e.name for e in train}
    assert {e.content_digest for e in train}.isdisjoint({e.content_digest for e in validation})


@pytest.mark.parametrize("mutation, message", [
    ("missing", "retain all prior validation"),
    ("changed_pixels", "Previously recorded episode changed"),
    ("renamed_copy", "copies prior validation"),
])
def test_refinement_rejects_invalid_or_leaking_holdout(tmp_path, mutation, message):
    for name in ("one", "two", "three"):
        _episode(tmp_path, name)
    original = discover_episodes(tmp_path, 2)
    previous = _split_metadata(original)
    held_out = original[-1].directory
    if mutation == "missing":
        shutil.rmtree(held_out)
    elif mutation == "changed_pixels":
        Image.new("RGB", (40, 30), (127, 127, 127)).save(held_out / "frames/000000.png")
    else:
        shutil.copytree(held_out, tmp_path / "new-name-copy")
    with pytest.raises(ValueError, match=message):
        refinement_split(discover_episodes(tmp_path, 2), previous)


def test_initial_split_keeps_exact_copies_together(tmp_path):
    source = _episode(tmp_path, "one")
    shutil.copytree(source, tmp_path / "copy-of-one")
    _episode(tmp_path, "two")
    train, validation = split_episodes(discover_episodes(tmp_path, 2), seed=3)
    assert {e.content_digest for e in train}.isdisjoint({e.content_digest for e in validation})


def test_dataset_fingerprint_detects_changed_images(tmp_path):
    episode = _episode(tmp_path, "one")
    first = dataset_digest(discover_episodes(tmp_path, 2))
    Image.new("RGB", (40, 30), (128, 64, 32)).save(episode / "frames/000000.png")
    assert dataset_digest(discover_episodes(tmp_path, 2)) != first


def test_recorded_action_vocabulary_cannot_be_silently_reordered(tmp_path):
    episode = _episode(tmp_path, "one")
    canonical = [{"name": "idle", "keys": []}, {"name": "forward", "keys": ["w"]}]
    vocabulary = tmp_path / "vocabulary.json"
    vocabulary.write_text(json.dumps(canonical))
    (episode / "actions.json").write_text(json.dumps(list(reversed(canonical))))
    with pytest.raises(ValueError, match="vocabulary differs"):
        discover_episodes(tmp_path, 2, actions=load_actions(vocabulary))


@pytest.mark.parametrize("explicit", [True, False])
def test_frame_history_resets_after_pause_or_dropped_interval(tmp_path, explicit):
    episode = _episode(tmp_path, "one")
    manifest = episode / "actions.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    if explicit:
        rows[2]["history_reset"] = True
    else:
        rows[1]["interval_seconds"] = 0.1
        rows[2]["timestamp"] += 5
    manifest.write_text("\n".join(json.dumps(row) for row in rows))
    dataset = EpisodeDataset(discover_episodes(tmp_path, 2), image_size=32, stack_size=4)
    history, _ = dataset[2]
    for index in range(1, 4):
        assert torch.equal(history[:3], history[index * 3:(index + 1) * 3])


def test_weight_only_refinement_resets_optimizer_and_keeps_validation(tmp_path):
    dataset = tmp_path / "unit-fixtures"
    for name in ("one", "two", "three"):
        _episode(dataset, name)
    vocabulary = tmp_path / "actions.json"
    vocabulary.write_text(json.dumps([{"name": "idle", "keys": []}, {"name": "forward", "keys": ["w"]}]))
    original = tmp_path / "base.pt"
    common = ["--dataset", str(dataset), "--actions", str(vocabulary), "--device", "cpu",
              "--image-size", "32", "--stack-size", "2", "--threads", "1", "--batch-size", "3"]
    assert train_main(common + ["--output", str(original), "--epochs", "2"]) == 0
    _, prior = load_checkpoint(original)
    _episode(dataset, "new-corrective")
    refined = tmp_path / "refined.pt"
    assert train_main(common + ["--output", str(refined), "--epochs", "1", "--init-checkpoint", str(original)]) == 0
    _, metadata = load_checkpoint(refined)
    assert metadata["epoch"] == 1
    assert metadata["split"]["validation_episodes"] == prior["split"]["validation_episodes"]
    assert "new-corrective" in metadata["split"]["train_episodes"]
    assert metadata["lineage"][-1]["optimizer_reset"] is True
    last = torch.load(tmp_path / "refined.last.pt", weights_only=True)
    assert len(last["training_state"]["history"]) == 1
    assert all(state["step"].item() == 3 for state in last["training_state"]["optimizer_state"]["state"].values())
