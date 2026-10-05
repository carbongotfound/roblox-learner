# Roblox Learner

A small, local visual-policy training toolkit for macOS. It records gameplay demonstrations, trains a convolutional policy, runs that policy using screenshots and ordinary keyboard/mouse events, and evaluates recorded trials against explicit game objectives.

**Current status: training/runtime toolkit under live validation. No policy has yet demonstrated successful completion of Peel a Potato, Deadly Delivery, or an obby.** A working pipeline and fast inference do not establish gameplay skill. See [validation](reports/validation.json) for measured evidence.

This is an independent MIT-licensed project, not an official Roblox model. Supporting a game's controls does not mean solving that game. New games require demonstrations, a useful action vocabulary, and a verifiable success condition.

## Install

Use Python 3.11 or newer on Apple Silicon macOS:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[mac,dev]'
pytest -q
```

The compact four-frame RGB CNN has approximately 250,000 parameters. The exact count depends on the action vocabulary. CPU inference is supported; training can use PyTorch MPS when available. The standalone runtime requires no API key, server, Roblox plugin, or script injection.

## Run the workflow

```sh
# Inspect existing OS permissions without changing them.
roblox-play --diagnose

# Capture a separate episode per attempt; F8 toggles recording, Escape stops.
roblox-record --config configs/peel_a_potato.json --output data/potato/episode-001

# Repeat recording with distinct episodes, then train.
roblox-train --dataset data/potato --actions configs/peel_a_potato.json \
  --output models/potato.pt --epochs 30 --device auto

# Measure actual policy compute; this does not measure game completion.
roblox-benchmark --checkpoint models/potato.pt --device cpu --iterations 500

# Bring Roblox gameplay to the foreground during the countdown.
roblox-play --model models/potato.pt --config configs/peel_a_potato.json \
  --output runs/potato/attempt-001 --seconds 120 --fps 10
```

The app controlling the Mac needs Screen Recording and Accessibility already enabled. Demonstration recording of a locked camera also needs Input Monitoring. The code diagnoses missing access and does not silently change system permissions. Escape stops the agent; changing the foreground app releases controls. Runs have explicit duration and memory limits.

Read the [complete training and refinement workflow](docs/TRAINING.md), [game objectives](docs/GAME_TARGETS.md), and each command's `--help`. Evaluation keeps failed, crashed, and unannotated attempts in its denominator. Visible success evidence must be reviewed; low training loss is never recorded as a win.

## Limits

- Behavior cloning learns from correctly aligned demonstrations. It cannot discover long-horizon game objectives from an empty dataset.
- The current CNN uses a short frame history. Complex navigation, inventory strategy, audio-only threats, and long-term planning may need a recurrent policy, additional modalities, or a hierarchical controller.
- The action profiles are editable starting points. Validate controls in the actual game before collecting training data.
- A sub-6-GB policy process does not imply the whole Mac plus Roblox uses less than 6 GB. Reports separate the policy process from the client.
- Recorded frames can contain account names and chat. Datasets and run recordings are local and ignored by Git by default.

## Reproducibility

Training writes compact standalone checkpoints, optimizer-resume checkpoints, dataset fingerprints, fixed episode splits, and metrics. `--init-checkpoint` supports new corrective demonstrations while preserving validation data. `--resume` continues the exact original training run. Tests cover temporal alignment, held-out data isolation, checkpoint loading, input cleanup, watchdogs, and evaluation evidence handling.

The MIT license covers this repository's code. Roblox and the third-party games remain the property of their respective owners.
