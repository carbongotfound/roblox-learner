# Pretrained real-time game policy investigation

Open Pixel2Play is a stronger starting point for fast games than the toolkit's
untrained compact CNN. Its authors report training on over 8,000 hours and
testing Roblox Rivals, Hypershot and Be a Shark. Their released live interaction
setup uses Windows with NVIDIA GPUs; it is not a supported Mac product.

Source: https://github.com/elefant-ai/open-p2p

`roblox_learner.p2p` is an experimental, local inference port of the 150M model.
It uses Apple GPU PyTorch operations, bounded temporal history and the learned
no-text embedding. No Gemma model or gated tokenizer is required for this mode.
The default history is 200 frames (ten seconds at the source model's 20Hz rate).
Inference results alone do not establish successful autonomous gameplay.

Install the optional dependency with `pip install -e '.[p2p]'`. Download the
upstream `150M/checkpoint-step=00500000.ckpt` from
https://huggingface.co/guaguaa/open-p2p . It is a 2.2GB training checkpoint;
optimizer tensors are memory mapped rather than loaded into the model.
The adapter checks the upstream LFS SHA256 before loading with
`torch.load(weights_only=True)`.

```sh
python -m roblox_learner.p2p \
  --checkpoint /path/to/checkpoint-step=00500000.ckpt \
  --image /path/to/roblox-screenshot.png \
  --device mps --iterations 100 --warmup 10 \
  --output reports/p2p-m4.json
```

This command predicts actions from a saved image and measures latency and
conservative agent memory. It does **not** send inputs. It excludes screen
capture, OS input delivery, game rendering and network latency. It is unsuitable
as evidence of a game win or a full real-time player. No autonomous playing
ability is claimed for this port until live evaluation succeeds.

Port differences: FP16 on MPS instead of upstream BF16; standard dense SDPA
instead of CUDA FlexAttention; Pillow Hamming resize instead of the original
Rust Hamming interpolation; mouse bin centers instead of truncated-normal
sampling within bins. Full-model numerical parity remains unverified. The
reports directory contains any measured results, with their explicit scope.

See [THIRD_PARTY.md](../THIRD_PARTY.md) for attribution and license.

## External live trials

The player runs locally and needs no language-model round trips. Enter actual
gameplay before starting it. The model was not designed to navigate Roblox's
home page, join servers, dismiss dialogs or select matches.

```sh
roblox-play p2p --checkpoint /path/to/checkpoint-step=00500000.ckpt \
  --output runs/p2p/trial-001 --seconds 60 --fps 15 --live
```

Omit `--live` to capture and predict without injecting controls. Live trials
hold simultaneous movement keys and mouse buttons, retain a bounded frame
history, and record timestamped actions plus JPEG evidence at up to 5fps.
Escape, foreground focus loss, a stalled loop, the 5.5GB conservative agent
memory limit or the duration limit release held inputs. The duration is capped
at ten minutes per invocation. The native macOS launcher can run this through
its already-supported `roblox_learner.play` module with `p2p` as the first argument.

The output summary reports measured control-loop throughput and
capture-to-input latency. It never labels a run a win automatically. Review the
frames and in-game result independently; teammate wins are not agent skill.

## Measured M4 results, 2026-10-06

The adapter was tested with the real Roblox Rivals client, an external screen
capture loop and ordinary OS keyboard/mouse events. It runs without chat-model
calls. All numbers below refer to the agent; Roblox and the OS use additional
memory. A separate 5fps recording was running during the live trials.

| Workload | Decisions/sec | Capture-to-control p95 | Peak accounted agent memory |
| --- | ---: | ---: | ---: |
| Live match, automatic graphics | 3.66 | 251ms | 1.83GB |
| Range, automatic graphics | 3.92 | 237ms | 2.13GB |
| Range, graphics quality 1 | 4.80 | 195ms | 2.19GB |

The isolated saved-image benchmark reached about 51ms median prediction time;
it did **not** predict live throughput. In the running game, prediction and
capture were slower. The lowest-graphics trial still failed the 15–20Hz target.
The original automatic graphics mode was restored after testing.

Movement and camera changes occurred during autonomous range trials, but the
target counter stayed at **0/5**. There are **zero verified agent wins**. The
match trial started after manual setup and AFK deaths, so it is not a controlled
skill measurement. The read-only shadow trial saw the match result screen and
does not establish gameplay ability. No human corrections were injected during
the finite policy trials.

The checkpoint is pretrained by the upstream authors. This project has **not**
fine-tuned it or established full-model numerical parity. Needed next steps are
whole-model/preprocessing parity, lower live latency, and game-specific training
with held-out success evaluations. This experimental adapter should not be
described as industrial-grade or capable of playing arbitrary Roblox games.

See [machine-readable assessment](../reports/p2p-live-assessment.json) for the
raw run summaries, timing breakdowns, action counts and evaluation scope.
