"""Train a compact behavior-cloning policy from actual screen/input episodes.

Held-out action accuracy is a diagnostic, not evidence of winning a game. A
policy must separately be evaluated through the ordinary Roblox client.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import EpisodeDataset, dataset_digest, discover_episodes, load_actions, refinement_split, split_episodes
from .model import CompactPolicy, PolicyConfig, read_checkpoint, resolve_device, save_checkpoint


def classification_metrics(confusion: torch.Tensor) -> dict[str, Any]:
    """Summarize action agreement, including minority-class performance."""
    confusion = confusion.cpu()
    support = confusion.sum(dim=1)
    represented = support > 0
    recalls = confusion.diag().float() / support.clamp_min(1)
    total = int(confusion.sum().item())
    return {
        "accuracy": float(confusion.diag().sum().item() / max(1, total)),
        "balanced_accuracy": float(recalls[represented].mean().item()) if represented.any() else 0.0,
        "per_action_recall": [float(recalls[i]) if support[i] else None for i in range(len(support))],
        "action_support": support.tolist(),
        "confusion_matrix": confusion.tolist(),
        "samples": total,
    }


def run_epoch(
    model: CompactPolicy,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    samples = 0
    confusion = torch.zeros(model.config.action_count, model.config.action_count, dtype=torch.int64)
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            batch_size = len(targets)
            total_loss += float(loss.detach().cpu().item()) * batch_size
            samples += batch_size
            predictions = logits.detach().argmax(dim=1).cpu()
            target_cpu = targets.detach().cpu()
            pairs = target_cpu * model.config.action_count + predictions
            confusion += torch.bincount(pairs, minlength=model.config.action_count**2).reshape(confusion.shape)
    result = classification_metrics(confusion)
    result["loss"] = total_loss / max(1, samples)
    return result


def _write_json(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1:
        raise ValueError("epochs, batch-size and threads must be positive")
    if args.learning_rate <= 0 or args.cache_size < 0 or args.patience < 0:
        raise ValueError("learning-rate must be positive; cache-size and patience must be nonnegative")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    actions = load_actions(args.actions)
    episodes = discover_episodes(args.dataset, len(actions), actions=actions)
    fingerprint = dataset_digest(episodes)
    config = PolicyConfig(len(actions), args.stack_size, args.image_size)
    if args.init_checkpoint:
        initial_payload = read_checkpoint(args.init_checkpoint)
        if initial_payload["config"] != asdict(config) or initial_payload["metadata"].get("actions") != actions:
            raise ValueError("Initialization requires identical architecture and ordered action vocabulary")
        train_episodes, validation_episodes = refinement_split(episodes, initial_payload["metadata"])
    else:
        initial_payload = None
        train_episodes, validation_episodes = split_episodes(episodes, args.validation_fraction, args.seed)
    model = CompactPolicy(config).to(device)
    if initial_payload is not None:
        model.load_state_dict(initial_payload["state_dict"], strict=True)
    # Weight-only refinement intentionally starts a fresh optimizer and epoch 1.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output = Path(args.output)
    last_output = output.with_name(output.stem + ".last" + output.suffix)
    metrics_output = output.with_name(output.stem + ".metrics.json")
    start_epoch, best_loss, stale_epochs = 0, float("inf"), 0
    history: list[dict[str, Any]] = []
    resume_payload = None
    if args.resume:
        resume_payload = read_checkpoint(args.resume)
        if resume_payload["config"] != asdict(config):
            raise ValueError("Resume architecture differs from --image-size, --stack-size or action count")
        previous = resume_payload["metadata"]
        if previous.get("actions") != actions:
            raise ValueError("Resume requires the identical ordered action vocabulary")
        if previous.get("dataset_digest") != fingerprint:
            raise ValueError("Dataset labels or screenshots changed; use --init-checkpoint for new corrective episodes")
        previous_training = previous.get("training", {})
        for name, requested in (("seed", args.seed), ("batch_size", args.batch_size),
                                ("class_weighting", args.class_weighting)):
            if previous_training.get(name) != requested:
                raise ValueError(f"Resume requires the previous {name}: {previous_training.get(name)}")
        split = previous.get("split", {})
        by_name = {episode.name: episode for episode in episodes}
        try:
            train_episodes = [by_name[name] for name in split["train_episodes"]]
            validation_episodes = [by_name[name] for name in split["validation_episodes"]]
        except KeyError as exc:
            raise ValueError("Resume checkpoint has an invalid episode split") from exc
        if (set(split["train_episodes"]) & set(split["validation_episodes"]) or
                set(split["train_episodes"]) | set(split["validation_episodes"]) != set(by_name)):
            raise ValueError("Resume checkpoint episode split leaks or omits episodes")
        state = resume_payload.get("training_state")
        if not isinstance(state, dict) or "optimizer_state" not in state:
            raise ValueError("Use the .last.pt checkpoint for resume; the compact policy omits optimizer state")
        model.load_state_dict(resume_payload["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state["epoch"])
        if args.epochs <= start_epoch:
            raise ValueError(f"--epochs is a total target and must exceed completed epoch {start_epoch}")
        best_loss = float(state["best_validation_loss"]) if output.exists() else float("inf")
        stale_epochs = int(state.get("stale_epochs", 0))
        history = list(state.get("history", []))

    train_data = EpisodeDataset(train_episodes, args.image_size, args.stack_size, args.cache_size)
    validation_data = EpisodeDataset(validation_episodes, args.image_size, args.stack_size, args.cache_size)
    label_counts = torch.bincount(torch.tensor(train_data.labels), minlength=len(actions))
    majority_action = int(label_counts.argmax())
    baseline = sum(label == majority_action for label in validation_data.labels) / len(validation_data)
    weights = None
    if args.class_weighting == "inverse-sqrt":
        weights = label_counts.clamp_min(1).float().rsqrt()
        weights /= weights.mean()
        weights = weights.to(device)
    train_criterion = nn.CrossEntropyLoss(weight=weights)
    validation_criterion = nn.CrossEntropyLoss()
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    inherited = (initial_payload or resume_payload or {}).get("metadata", {})
    episode_fingerprints = dict(inherited.get("episode_fingerprints", {}))
    episode_fingerprints.update({e.name: {"manifest": e.manifest_digest, "content": e.content_digest}
                                 for e in episodes})
    lineage = list(inherited.get("lineage", []))
    if initial_payload is not None:
        lineage.append({"method": "weight_only_refinement", "optimizer_reset": True,
                        "parent_checkpoint_sha256": hashlib.sha256(Path(args.init_checkpoint).read_bytes()).hexdigest(),
                        "parent_epoch": initial_payload["metadata"].get("epoch"),
                        "validation_policy": "prior held-out episodes preserved unchanged; new episodes train only"})
    metadata = {
        "actions": actions,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": "compact-cnn-causal-frame-stack-v1",
        "observation": "external RGB screenshot only",
        "preprocessing": {"resize": "bilinear", "channels": "RGB", "range": [0, 1]},
        "dataset_digest": fingerprint,
        "dataset_fingerprint_scope": "episode names, actions.jsonl contents and all referenced image bytes",
        "episode_fingerprints": episode_fingerprints,
        "lineage": lineage,
        "split": {"train_episodes": [e.name for e in train_episodes],
                  "validation_episodes": [e.name for e in validation_episodes]},
        "training": {"method": "behavior_cloning", "seed": args.seed,
                     "batch_size": args.batch_size, "class_weighting": args.class_weighting,
                     "learning_rate": optimizer.param_groups[0]["lr"],
                     "device": str(device), "torch_version": str(torch.__version__)},
        "gameplay_success": "NOT EVALUATED by the training command",
    }
    summary: dict[str, Any] = {
        "schema_version": 1, "task": "held-out action classification",
        "gameplay_success": "NOT EVALUATED",
        "policy_config": asdict(config), "device": str(device),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "train_samples": len(train_data), "validation_samples": len(validation_data),
        "split": metadata["split"], "train_action_counts": label_counts.tolist(),
        "validation_majority_baseline_accuracy": baseline,
        "unseen_training_actions": [actions[i]["name"] for i, count in enumerate(label_counts) if not count],
        "lineage": lineage,
        "epochs": history,
    }
    print(json.dumps({"status": "training", "device": str(device),
                      "train_episodes": len(train_episodes), "validation_episodes": len(validation_episodes),
                      "train_frames": len(train_data), "validation_frames": len(validation_data),
                      "parameters": summary["parameter_count"]}), flush=True)
    started = time.perf_counter()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Epoch-specific shuffling makes resumed runs use the same next order.
        generator = torch.Generator().manual_seed(args.seed + epoch)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, generator=generator)
        epoch_started = time.perf_counter()
        training_metrics = run_epoch(model, train_loader, train_criterion, device, optimizer)
        validation_metrics = run_epoch(model, validation_loader, validation_criterion, device)
        record = {"epoch": epoch, "seconds": time.perf_counter() - epoch_started,
                  "train": training_metrics, "validation": validation_metrics}
        history.append(record)
        improved = validation_metrics["loss"] < best_loss
        if improved:
            best_loss = validation_metrics["loss"]
            stale_epochs = 0
            best_metadata = {**metadata, "epoch": epoch, "validation": validation_metrics}
            save_checkpoint(output, model, best_metadata)
        else:
            stale_epochs += 1
        save_checkpoint(last_output, model, {**metadata, "epoch": epoch, "validation": validation_metrics},
                        {"epoch": epoch, "optimizer_state": optimizer.state_dict(),
                         "best_validation_loss": best_loss, "stale_epochs": stale_epochs, "history": history})
        summary.update({"best_validation_loss": best_loss, "completed_epochs": epoch,
                        "elapsed_seconds_this_run": time.perf_counter() - started,
                        "policy_checkpoint": str(output), "resume_checkpoint": str(last_output)})
        _write_json(metrics_output, summary)
        print(json.dumps({"epoch": epoch, "train_loss": training_metrics["loss"],
                          "validation_loss": validation_metrics["loss"],
                          "validation_accuracy": validation_metrics["accuracy"],
                          "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
                          "new_best": improved}), flush=True)
        if args.patience and stale_epochs >= args.patience:
            break
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Directory containing independent episode folders")
    parser.add_argument("--actions", required=True, help="Ordered action vocabulary JSON")
    parser.add_argument("--output", required=True, help="Compact best-validation checkpoint, e.g. models/policy.pt")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", help="Continue the identical dataset and optimizer from a .last.pt checkpoint")
    initialization.add_argument("--init-checkpoint", help="Refine weights with new training episodes; preserve prior holdout, reset optimizer")
    parser.add_argument("--epochs", type=int, default=20, help="Total target epochs, including resumed epochs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--stack-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-size", type=int, default=256, help="Maximum decoded frames cached per split")
    parser.add_argument("--class-weighting", choices=("none", "inverse-sqrt"), default="none")
    parser.add_argument("--patience", type=int, default=0, help="Stop after N epochs without improvement; 0 disables")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        train(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Training error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
