#!/usr/bin/env python3
"""Profile candidate model sizes on an RTX 2070-style 8GB GPU.

Runs one synthetic train micro-step for a model config and reports actual CUDA
memory use.  This intentionally profiles the largest experiment arm
(`mhc_mtp_mix`) by default, since any config that fits that arm should fit the
controls.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
from pathlib import Path

import torch

from experiment import ExperimentConfig, build_model
from controls import build_control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="mhc_mtp", choices=[
        "mhc_mtp", "mhc_only", "residual_mtp", "residual_only",
    ])
    parser.add_argument("--reduction", default="mix", choices=["sum", "mix"])
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--n_streams", type=int, default=4)
    parser.add_argument("--n_mtp", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--ffn_mult", type=int, default=4)
    parser.add_argument("--sinkhorn_iters", type=int, default=20)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def build_profile_model(args: argparse.Namespace):
    cfg = ExperimentConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_streams=args.n_streams,
        ffn_mult=args.ffn_mult,
        max_seq_len=args.seq_len + 64,
        n_mtp=args.n_mtp,
        reduction=args.reduction,
        sinkhorn_iters=args.sinkhorn_iters,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    if args.model == "mhc_mtp":
        return build_model(cfg), cfg
    return build_control(args.model, cfg), cfg


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for VRAM profiling")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")

    model, cfg = build_profile_model(args)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    input_ids = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.seq_len),
        device=device,
    )

    use_amp = args.dtype == "float16"
    amp_dtype = torch.float16 if use_amp else torch.float32

    optimizer.zero_grad(set_to_none=True)
    try:
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            loss, metrics = model(input_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as exc:
        result = {
            "ok": False,
            "error": str(exc).splitlines()[0],
            "config": dataclasses.asdict(cfg),
        }
        print(json.dumps(result, indent=2))
        if args.json:
            args.json.write_text(json.dumps(result, indent=2) + "\n")
        return 2

    params = sum(p.numel() for p in model.parameters())
    result = {
        "ok": True,
        "model": args.model,
        "reduction": args.reduction,
        "params": params,
        "params_m": params / 1e6,
        "loss": float(loss.detach().cpu()),
        "main_loss": float(metrics["main_loss"].detach().cpu()),
        "allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        "config": dataclasses.asdict(cfg),
    }
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")

    del optimizer, model, input_ids, loss
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
