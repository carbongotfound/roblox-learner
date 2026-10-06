"""Experimental Apple GPU inference for the pretrained Open Pixel2Play 150M.

Port of elefant-ai/open-p2p (MIT), revision a329d98. See THIRD_PARTY.md.
This diagnostic emits predictions only; it does not inject game controls. It
uses PyTorch SDPA instead of CUDA FlexAttention and the learned no-text token,
so the separate Gemma text encoder and training dependencies are unnecessary.
The port must be evaluated in games before making any gameplay claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

KEYS = (None, "space", "1", "2", "3", "4", "a", "d", "e", "f", "q", "w",
        "s", "z", "down", "up", "left", "right", "shift", "shift")
BUTTONS = (None, "left", "right", "middle")
MOUSE_X = (-501, -322, -110, -61, -38, -24, -15, -9, -5, -2, -1, 0,
           1, 2, 5, 9, 15, 24, 38, 61, 110, 322, 501)
MOUSE_Y = (-151, -87, -18, -10, -6, -4, -2, -1, 0, 1, 2, 4, 6, 10, 18, 87, 151)
CHECKPOINT_SHA256 = "2a3b43121144ca2d9c2b8aa605e313184c98ac5ef6d2c3b530c2d353d886c5bc"


def policy_mask(query_positions, key_positions):
    """Upstream 12-token step: text, image, thinking, output, 8 actions.

    Excludes every previous output token. Current image/text/thinking attend
    mutually, output sees those plus itself, and actions see all current tokens
    except output. Keeping this last property requires a second forward pass.
    """
    q = query_positions[:, None]
    k = key_positions[None, :]
    same = q // 12 == k // 12
    history = (k // 12 < q // 12) & (q // 12 - k // 12 <= 200) & (k % 12 != 3)
    current = same & (((q % 12 < 3) & (k % 12 < 3)) |
                      ((q % 12 == 3) & (k % 12 <= 3)) |
                      ((q % 12 > 3) & (k % 12 != 3)))
    return history | current


def decode_actions(tokens):
    values = list(tokens)
    if len(values) != 8 or any(isinstance(v, bool) or not isinstance(v, int) for v in values):
        raise ValueError("Expected eight integer action tokens")
    limits = [20] * 4 + [4] * 2 + [23, 17]
    if any(not 0 <= v < n for v, n in zip(values, limits)):
        raise ValueError("Action token outside pretrained vocabulary")
    return {"keys": sorted({KEYS[v] for v in values[:4]} - {None}),
            "buttons": sorted({BUTTONS[v] for v in values[4:6]} - {None}),
            "mouse_dx_pixels": MOUSE_X[values[6]], "mouse_dy_pixels": MOUSE_Y[values[7]]}


class P2PPolicy:
    """Fixed 150M architecture with a bounded 200-frame KV history."""
    def __init__(self, checkpoint: Path, device="mps", *, verify=True, history=200):
        from torchvision.models import efficientnet_b0
        if device not in {"cpu", "mps"} or not 1 <= history <= 200:
            raise ValueError("device must be cpu/mps; history must be 1..200")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("Apple GPU unavailable")
        if verify:
            with checkpoint.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != CHECKPOINT_SHA256:
                raise ValueError("Checkpoint does not match upstream 150M SHA256")
        # Never execute downloaded pickle code; mmap avoids loading optimizer
        # tensors from the 2.2GB training checkpoint into resident memory.
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        self.training_step = int(checkpoint_data["global_step"])
        source = checkpoint_data["state_dict"]
        self.device = device
        self.dtype = torch.float16 if device == "mps" else torch.float32
        self.history = history
        torch.set_num_threads(4)
        if device == "mps":
            torch.mps.set_per_process_memory_fraction(min(.75, 3_500_000_000 / torch.mps.recommended_max_memory()))
        self.vision = efficientnet_b0(weights=None).features[:6]
        prefix = "image_tokenizer.efficientnet_preprocess."
        self.vision.load_state_dict({k[len(prefix):]: v for k, v in source.items() if k.startswith(prefix)}, strict=True)
        self.vision.eval().to(device=device, dtype=self.dtype)
        # The checkpoint stores the same tokenizer under two module paths.
        self.weights = {k: v.to(device=device, dtype=self.dtype) for k, v in source.items()
                        if not k.startswith(prefix) and not k.startswith("bc_transformer.image_tokenizer.")
                        and not k.startswith("_orig_mod.")}
        self.tensor_elements = sum(v.numel() for v in self.weights.values()) + sum(v.numel() for v in self.vision.state_dict().values())
        del source, checkpoint_data
        self.prefix = "bc_transformer."
        if self.weights[self.prefix + "img_pos_tokens"].shape != (1, 1, 1024):
            raise ValueError("This port supports the upstream 150M architecture only")
        self.reset()

    def reset(self):
        self.cache = [None] * 10
        self.position = 0

    def linear(self, x, name):
        return F.linear(x, self.weights[name + ".weight"], self.weights.get(name + ".bias"))

    def norm(self, x, name):
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)).to(x.dtype) * self.weights[name + ".scale"]

    @staticmethod
    def rotations(dim, positions):
        freq = 1 / (10000 ** (torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32) / dim))
        phase = positions.float()[:, None] * freq[None, :]
        return phase.cos()[None, :, None, :], phase.sin()[None, :, None, :]

    @staticmethod
    def rope(x, positions, rotations=None):
        cos, sin = rotations if rotations is not None else P2PPolicy.rotations(x.shape[-1], positions)
        paired = x.float().reshape(*x.shape[:-1], -1, 2)
        return torch.stack((paired[..., 0] * cos - paired[..., 1] * sin,
                            paired[..., 1] * cos + paired[..., 0] * sin), -1).flatten(-2).to(x.dtype)

    def layer(self, x, prefix, positions, cache, mask, heads, sinks=False, rotations=None):
        sa = prefix + "self_attention."
        q, k, v = self.linear(self.norm(x, prefix + "self_attention_norm"), sa + "c_attn").chunk(3, -1)
        q = self.norm(q, sa + "q_norm")
        k = self.norm(k, sa + "k_norm")
        shape = (*q.shape[:2], heads, q.shape[-1] // heads)
        q = self.rope(q.reshape(shape), positions, rotations).transpose(1, 2)
        k = self.rope(k.reshape(shape), positions, rotations).transpose(1, 2)
        v = v.reshape(shape).transpose(1, 2)
        if cache is not None:
            k, v = torch.cat((cache[0], k), 2), torch.cat((cache[1], v), 2)
        new_cache = (k, v)
        if sinks:
            k = torch.cat((self.weights[sa + "k_sinks"], k), 2)
            v = torch.cat((self.weights[sa + "v_sinks"], v), 2)
        attended = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(), attn_mask=mask)
        x = x + self.linear(attended.transpose(1, 2).reshape_as(x), sa + "c_proj")
        left, right = self.linear(self.norm(x, prefix + "ffn_norm"), prefix + "ffn.w13").chunk(2, -1)
        return x + self.linear(F.silu(left) * right, prefix + "ffn.w2"), new_cache

    def transformer(self, x, positions, caches, prefix, heads, mask, sinks=False):
        result = []
        rotations = self.rotations(x.shape[-1] // heads, positions)
        for i, cache in enumerate(caches):
            x, updated = self.layer(x, prefix + f"transformer_layers.{i}.", positions, cache, mask, heads, sinks, rotations)
            result.append(updated)
        return x, result

    @torch.inference_mode()
    def predict(self, image, *, sample=True):
        # The upstream Rust resizer also uses Hamming; numerical parity of the
        # two resizing implementations is not established.
        pixels = np.array(image.convert("RGB").resize((192, 192), Image.Resampling.HAMMING), copy=True)
        frame = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=self.dtype) / 255
        features = self.vision(frame).reshape(1, -1)
        features = self.linear(features, "image_tokenizer.mlp.0")
        features = F.layer_norm(features, (1024,), self.weights["image_tokenizer.mlp.1.weight"], self.weights["image_tokenizer.mlp.1.bias"], 1e-5)
        p = self.prefix
        base = torch.cat((self.weights[p + "text_embedding_for_no_text_input"] + self.weights[p + "text_pos_tokens"],
                          features[:, None, :] + self.weights[p + "img_pos_tokens"],
                          self.weights[p + "thinking_pos_tokens"], self.weights[p + "action_out_token"]), 1)
        old_len = 0 if self.cache[0] is None else self.cache[0][0].shape[2]
        if old_len >= self.history * 12:
            self.cache = [(k[:, :, 12:], v[:, :, 12:]) for k, v in self.cache]
            old_len -= 12
        positions = torch.arange(self.position, self.position + 12, device=self.device)
        key_positions = torch.arange(self.position - old_len, self.position + 12, device=self.device)
        mask = policy_mask(positions, key_positions)
        # Observation/output queries cannot attend to current action tokens.
        # Avoid computing eight masked-out dummy queries in the first pass.
        y, _ = self.transformer(base, positions[:4], self.cache, p + "_transformer.", 16, mask[:4, :old_len + 4])
        decoder = p + "action_decoder."
        token = self.linear(y[:, 3:4], decoder + "input_proj")
        decoder_cache = [None] * 3
        embedded, sampled = [], []
        for i in range(8):
            token = token + self.weights[decoder + "pos_tokens"][None, i:i+1]
            token, decoder_cache = self.transformer(token, torch.tensor([i], device=self.device), decoder_cache, decoder, 8, None, sinks=True)
            kind = "key_action" if i < 4 else "mouse_button" if i < 6 else "mouse_delta_x" if i == 6 else "mouse_delta_y"
            head = "keyboard" if i < 4 else kind
            logits = self.linear(token, head + "_out_logits")
            index = torch.multinomial(logits.float().softmax(-1).reshape(1, -1), 1) if sample else logits.argmax(-1)
            sampled.append(index)
            token = F.embedding(index, self.weights[kind + "_embedding.weight"])
            embedded.append(token)
        actions = torch.cat(embedded, 1) + self.weights[p + "action_pos_tokens"]
        _, self.cache = self.transformer(torch.cat((base, actions), 1), positions, self.cache, p + "_transformer.", 16, mask)
        self.position += 12
        tokens = torch.cat(sampled, -1).squeeze(0).tolist()
        return {"tokens": tokens, **decode_actions(tokens)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--history", type=int, default=200)
    args = parser.parse_args(argv)
    if not 1 <= args.iterations <= 10000 or not 0 <= args.warmup <= 1000:
        parser.error("iterations must be 1..10000; warmup 0..1000")
    from .desktop import process_rss_mb
    started = time.monotonic()
    policy = P2PPolicy(args.checkpoint, args.device, history=args.history)
    print(json.dumps({"loaded_seconds": time.monotonic() - started, "rss_mb": process_rss_mb()}), flush=True)
    with Image.open(args.image) as source:
        frame = source.convert("RGB")
    timings, examples = [], []
    peak_memory = 0
    for i in range(args.warmup + args.iterations):
        start = time.perf_counter()
        prediction = policy.predict(frame)
        elapsed = (time.perf_counter() - start) * 1000
        memory = process_rss_mb()
        if args.device == "mps":
            memory += torch.mps.driver_allocated_memory() / 1e6
        peak_memory = max(memory, peak_memory)
        if memory > 5500:
            raise RuntimeError(f"Conservative agent memory exceeded 5500MB: {memory:.0f}")
        if i >= args.warmup:
            timings.append(elapsed)
            if len(examples) < 5:
                examples.append(prediction)
        if i % 20 == 0:
            print(json.dumps({"step": i, "ms": elapsed, "memory_mb": memory}), flush=True)
    report = {"model": "Open Pixel2Play 150M", "upstream": "https://github.com/elefant-ai/open-p2p",
              "checkpoint_sha256": CHECKPOINT_SHA256, "device": args.device,
              "upstream_training_step": policy.training_step, "tensor_elements_including_buffers": policy.tensor_elements,
              "iterations": args.iterations, "history_limit_frames": args.history,
              "p50_ms": float(np.percentile(timings, 50)), "p95_ms": float(np.percentile(timings, 95)),
              "mean_ms": float(np.mean(timings)), "max_ms": max(timings),
              "compute_meets_20hz_p95": bool(np.percentile(timings, 95) <= 50),
              "max_conservative_agent_memory_mb": peak_memory, "examples": examples,
              "pretrained": True, "port_numerical_parity_verified": False,
              "gameplay_success": "NOT EVALUATED", "live_input": False,
              "excludes": ["screen capture", "OS input", "game rendering", "network"],
              "resize_difference": "Pillow Hamming instead of upstream Rust Hamming interpolation",
              "mouse_decode": "bin centers rather than upstream truncated normal sampling"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
