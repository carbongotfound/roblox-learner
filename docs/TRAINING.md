# Train from real external play

This repository learns a finite keyboard/mouse action from four recent RGB screenshots. It is an imitation-learning baseline with a compact convolutional policy. It does not read Roblox memory, use game remotes, inject scripts, or contain universal Roblox knowledge. A model trained on one camera layout and task does not automatically understand another game. Four frames provide short-term motion information; long-horizon navigation, inventory planning and exact peeling may require a richer policy and action vocabulary.

Use the setup commands in the repository README first. All commands below run from the repository root with the environment activated. The supplied configs describe starting action vocabularies, not confirmed working strategies. `configs/peel_a_potato.json` and `configs/deadly_delivery.json` name the exact game targets and provisional goals; see [GAME_TARGETS.md](GAME_TARGETS.md).

## 1. Validate the controls and record episodes

Open Roblox normally, join the target experience, set a consistent window size/camera sensitivity and bring the game to the intended start state. The desktop recorder requires macOS Screen Recording and Accessibility access for the terminal/Python host. Camera demonstrations also require Input Monitoring for raw mouse deltas, because Roblox may lock or recenter its cursor. The recorder checks existing permissions without requesting or changing them. It stops when Roblox loses foreground focus. Escape is the emergency stop; F8 toggles recording. Begin recording only after normal login and loading are complete.

```sh
python -m roblox_learner.record --config configs/peel_a_potato.json --dry-run
python -m roblox_learner.record \
  --config configs/peel_a_potato.json \
  --output data/potato/episode-001 \
  --seconds 180 --fps 10
```

After the countdown, bring Roblox forward and press F8. Play a successful, concise segment with actual keyboard and mouse inputs. Repeat into different directories such as `episode-002`. At least two independent episodes are required by the trainer; two is a technical minimum, not sufficient evidence of mastery. Cover different starting views, approach errors, recovery, timing variations and relevant game states. Record both complete intended behaviors and corrections of failures observed during policy play.

The recorder matches physical input combinations to the config's discrete actions. It drops unmapped combinations and records their counts; inspect `episode.json` and the recording events. A high drop count means the action vocabulary does not cover the way the game is being played. Add the missing combinations or finer mouse increments, then record a consistent new dataset. Do not silently relabel unrepresentable controls as idle. Keep the action order fixed: the integer label is its array index.

Each directory contains `actions.jsonl` and the corresponding images. One line has this structure (illustrative schema, not a supplied gameplay sample):

```json
{"frame":"frames/0000001.jpg","action":1,"timestamp":1791200000.0}
```

The screenshot precedes the demonstrated action interval. Arbitrary online gameplay videos generally do not contain exact input labels or synchronized cursor deltas; downloading them is not equivalent to a usable demonstration dataset. They can inform task analysis, but claims of training require identifiable samples and a completed optimizer run.

## 2. Train and inspect what was learned

```sh
python -m roblox_learner.train \
  --dataset data/potato \
  --actions configs/peel_a_potato.json \
  --output models/potato-v1.pt \
  --epochs 20 --batch-size 32 --device auto \
  --class-weighting inverse-sqrt
```

Training splits whole episode directories so neighboring frames stay in the same split. It saves the compact best-validation checkpoint and a `.last.pt` training checkpoint, plus metrics and split metadata. Inspect action counts, validation loss and per-action metrics. High accuracy dominated by idle does not prove a useful player. A high action-prediction score is not a game win.

Use `--resume models/potato-v1.last.pt` only to continue the same dataset, config and training setup to a larger total epoch count. Resume verifies both label manifests and screenshot contents. After adding corrective demonstrations, use `--init-checkpoint models/potato-v1.pt` as described below. Version checkpoints, config files and outcomes together. Keep a separate frozen live-play evaluation set; development episodes are not final benchmarks.

## 3. Measure the compact model, then play

```sh
python -m roblox_learner.benchmark \
  --checkpoint models/potato-v1.pt --device cpu \
  --iterations 200 --fps-target 10 \
  --output reports/potato-v1-inference.json
```

A model benchmark measures a controlled inference workload. It does not include Roblox rendering, screenshot capture, input delivery, or game progress. Measure the complete desktop loop separately. The requested memory cap is 6 GB for the AI process; count framework buffers and capture memory as well as weight size. Report Roblox and total system memory separately when measured. The runtime stops before its configured process-memory limit.

Run a bounded first rollout after returning Roblox to the declared start state:

```sh
python -m roblox_learner.play \
  --model models/potato-v1.pt --config configs/peel_a_potato.json \
  --output logs/potato-trial-001 --seconds 120 --fps 10 \
  --device cpu --save-every 10 --memory-limit-mb 5500
```

The countdown gives time to bring Roblox forward. Preserve the initial and terminal screenshots and inspect emitted actions. Keep the physical Escape stop available. Increase duration only after actual task progress is visible. Dry-run validation does not move the character and cannot satisfy a gameplay criterion. A stop screenshot marked `terminal_frame_current: false` is a cached observation, not a fresh terminal capture; the evaluator rejects it as terminal proof.

## 4. Correct failures and retrain

After a failed rollout, identify the first wrong decision, the visual state and the missing behavior. Recreate that state in Roblox and record the correct recovery into a new episode. Useful corrections include turning toward the bench, reacquiring the potato, stopping near an edge, returning to a safe route and recovering after a blocked interaction. Preserve earlier successful demonstrations to avoid teaching only recovery states.

Refine the weights while preserving the original held-out episodes:

```sh
python -m roblox_learner.train \
  --dataset data/potato --actions configs/peel_a_potato.json \
  --init-checkpoint models/potato-v1.pt --output models/potato-v2.pt \
  --epochs 10 --batch-size 32 --device auto \
  --class-weighting inverse-sqrt
```

`--init-checkpoint` requires the same architecture and ordered action vocabulary. It starts a fresh optimizer and epoch count, keeps the prior held-out episodes unchanged, and assigns new episodes only to training. Renamed copies of held-out demonstrations are rejected. Preserve earlier successful training episodes when adding corrections.

Repeat the same declared evaluation conditions with fresh trials. Compare actual success counts and completion times, not a single attractive clip. If controls or camera settings change, record that change and recheck all relevant data. Stop adding training epochs when validation and real play stop improving; investigate observation quality, action precision, missing memory and demonstration coverage. This implements an iterative demonstration → policy rollout → corrective demonstration → retraining process; it is not autonomous reward-driven reinforcement learning.

The framework can be adapted to other Roblox games by supplying an adequate action set, real demonstrations and a measurable success criterion. It cannot promise that any game is solvable by this small architecture. Tasks requiring information absent from screenshots, very long memory, or continuous precision beyond the chosen actions need additional modeling work.

## 5. Audit actual outcomes

Every attempted play episode stays in the denominator. Use `logs/<run>/episode.jsonl`, not a demonstration manifest. Annotate success only after inspecting the actual starting state and visible result. The following is a template; replace every placeholder from the corresponding real log and screenshot before using it:

```json
{"episode_id":"UUID_FROM_LOG","outcome":"win","criterion_id":"potato_shipment_complete","reviewed_by":"IDENTIFIED_VISUAL_REVIEWER","success_visible":true,"unassisted":true,"interventions":0,"checkpoint_sha256":"SHA256_FROM_START_EVENT","terminal_frame":"frames/ACTUAL_SAVED_FRAME.jpg","terminal_sha256":"ACTUAL_IMAGE_SHA256"}
```

Paths resolve relative to the episode log's directory. The terminal frame must already be referenced in that episode's events. Obtain its hash with `shasum -a 256 logs/<run>/frames/<frame>.jpg`. The initial screenshot must exist in the start event, and the checkpoint hash must match. Use `outcome: "loss"` or `"incomplete"` for other outcomes. Missing annotations remain unverified trials. A manual recovery must set `unassisted: false` and an accurate `interventions` count.

```sh
python -m roblox_learner.evaluate \
  --logs logs/potato-trial-001/episode.jsonl logs/potato-trial-002/episode.jsonl \
  --outcomes reports/potato-outcomes.jsonl \
  --criterion-id potato_shipment_complete \
  --output reports/potato-v1-live-evaluation.json
```

The report distinguishes reported wins from reviewed wins backed by complete log/screenshot files. It reports the trial count, 95% Wilson interval, unannotated episodes, timing samples, process RSS and rejection reasons. It refuses duplicate episode IDs or duplicate annotations. No annotations means no verified wins. Neither a successful unit test nor a synthetic learning smoke test contributes to live success counts.

The default potato config's full goal is `potato_shipment_complete`. For initial paid-cycle experiments, make a separately named copy of the config with `evaluation.criterion_id` set to `potato_one_paid_cycle` before recording the trial; the annotation must match the logged criterion. Do not change a trial's target after seeing its outcome.
