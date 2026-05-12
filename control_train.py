#!/usr/bin/env python3
"""
train.py — Train mHC + Sequential MTP on WikiText-103
======================================================

Usage
-----
    # Main experiment — mHC trunk + sequential MTP:
    python train.py --model mhc_mtp --reduction sum   # Condition A (baseline sum)
    python train.py --model mhc_mtp --reduction mix   # Condition B (mixing matrix)

    # Control experiments:
    python train.py --model residual_only             # Control 0: standard trunk, no MTP
    python train.py --model mhc_only                  # Control A: mHC, no MTP
    python train.py --model residual_mtp              # Control B: standard trunk + MTP

    # Resume from a checkpoint:
    python train.py --model mhc_mtp --reduction sum --resume runs/mhc_mtp_sum/checkpoint_latest.pt

    # Override any hyperparameter:
    python train.py --model mhc_mtp --reduction mix --d_model 256 --n_layers 6 --lr 2e-4

    # Evaluate a saved checkpoint:
    python train.py --model mhc_only --eval_only --resume runs/mhc_only/checkpoint_best.pt

Model scale (default, fits comfortably on 12 GB GPU)
-----------------------------------------------------
    d_model     = 512
    n_layers    = 8
    n_heads     = 8
    n_streams   = 4       (mHC parallel streams)
    ffn_mult    = 4
    seq_len     = 512
    batch_size  = 16      (effective: 16 × 512 = 8192 tokens/step)
    grad_accum  = 4       (effective batch: 32768 tokens/step)
    n_mtp       = 2       (predict offsets +2, +3)

    Approximate parameter count: ~85M (trunk) + ~6M (MTP modules)

    Expected training time on a single RTX 3080/3090/4080 (12 GB):
        ~6–10 hours for 20 000 steps (~650M tokens seen)
        Use --n_steps 10000 for a ~3–5 hour run with still meaningful results.

Dataset
-------
    WikiText-103 via HuggingFace datasets.
    ~103M training tokens, standard benchmark for small LM research.
    Downloaded and cached automatically on first run (~180 MB).
    Falls back to WikiText-2 (~2M tokens) if WikiText-103 is unavailable.

Checkpointing
-------------
    Saves to  runs/<reduction>/checkpoint_latest.pt   every --save_every steps
    Saves to  runs/<reduction>/checkpoint_best.pt     whenever val loss improves
    Checkpoint contains: model state, optimizer state, step, best_val_loss, config

Logging
-------
    Prints to stdout.
    Writes a CSV log to  runs/<reduction>/metrics.csv  (step, train_loss,
    val_loss, main_loss, mtp_loss, depth_0_loss, ..., mix_weight_diag, lr,
    tokens_per_sec).

Dependencies
------------
    pip install torch datasets transformers tiktoken
    mhc.py, mtp.py, experiment.py in the same directory.
"""

import argparse
import csv
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

# ---------------------------------------------------------------------------
# Local imports — must be in the same directory
# ---------------------------------------------------------------------------
from experiment import ExperimentConfig, build_model, mHCWithMTP
from controls import build_control

# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def get_tokenizer():
    """
    Use GPT-2 BPE tokenizer (vocab size 50257) via tiktoken.
    Falls back to a character-level tokenizer if tiktoken is unavailable.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return enc, enc.n_vocab
    except ImportError:
        print("tiktoken not found; falling back to character-level tokenizer.")
        print("  Install with:  pip install tiktoken")
        return None, 256   # byte-level fallback


def encode_text(text: str, tokenizer) -> list[int]:
    """Encode a string to a list of integer token ids."""
    if tokenizer is None:
        # Byte-level fallback
        return list(text.encode("utf-8"))
    return tokenizer.encode(text)


# ---------------------------------------------------------------------------
# Dataset: WikiText-103 token stream
# ---------------------------------------------------------------------------

class TokenStreamDataset(IterableDataset):
    """
    Wraps a flat list of token ids as an infinite stream of fixed-length chunks.

    Each item yielded is a LongTensor of length (seq_len + 1): the first seq_len
    tokens are the input, the last seq_len tokens (offset by 1) are the targets.
    We return seq_len + 1 so the caller can slice input_ids = item[:-1] and
    targets = item[1:] if needed; in practice we pass the full seq_len+1 chunk
    and let the model/loss handle indexing.

    The stream wraps around when exhausted (infinite epoch).

    Args:
        token_ids : flat list or 1-D LongTensor of all token ids
        seq_len   : context window length
        shuffle   : if True, start at a random offset each wrap (training only)
        seed      : RNG seed for shuffle
    """

    def __init__(
        self,
        token_ids: torch.Tensor,
        seq_len: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.token_ids = token_ids
        self.seq_len   = seq_len
        self.shuffle   = shuffle
        self.seed      = seed
        self.chunk_len = seq_len + 1   # +1 so we have a target for every input

    def __iter__(self) -> Iterator[torch.Tensor]:
        rng   = torch.Generator()
        epoch = 0
        while True:
            n      = len(self.token_ids)
            n_full = (n - 1) // self.chunk_len   # number of full non-overlapping chunks

            if self.shuffle:
                rng.manual_seed(self.seed + epoch)
                # Random start offset so we don't always start at position 0
                offset = int(torch.randint(0, self.chunk_len, (1,), generator=rng).item())
            else:
                offset = 0

            for i in range(n_full):
                start = offset + i * self.chunk_len
                end   = start + self.chunk_len
                if end > n:
                    break
                yield self.token_ids[start:end]

            epoch += 1


def load_wikitext(
    seq_len: int,
    tokenizer,
    seed: int = 42,
    dataset: str = "wikitext103",
    max_train_tokens: Optional[int] = None,
    max_val_tokens: Optional[int] = None,
):
    """
    Download and tokenise WikiText-103 (falls back to WikiText-2).

    Returns (train_dataset, val_dataset, vocab_size).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not found.")
        print("  Install with:  pip install datasets transformers")
        sys.exit(1)

    dataset_configs = {
        "wikitext103": "wikitext-103-raw-v1",
        "wikitext2": "wikitext-2-raw-v1",
    }
    requested_name = dataset_configs[dataset]

    print(f"Loading {requested_name} ... ", end="", flush=True)
    try:
        ds = load_dataset("wikitext", requested_name)
        dataset_name = requested_name
    except Exception:
        if dataset == "wikitext2":
            raise
        print("(falling back to wikitext-2-raw-v1) ", end="", flush=True)
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        dataset_name = "wikitext-2-raw-v1"

    print(f"done ({dataset_name})")

    def tokenise_split(split_name: str, max_tokens: Optional[int]) -> torch.Tensor:
        ids: list[int] = []
        for text in ds[split_name]["text"]:
            if not text.strip():
                continue
            ids.extend(encode_text(text + "\n", tokenizer))
            if max_tokens is not None and len(ids) >= max_tokens:
                del ids[max_tokens:]
                break
        return torch.tensor(ids, dtype=torch.long)

    print("Tokenising ... ", end="", flush=True)
    t0 = time.time()
    train_ids = tokenise_split("train", max_train_tokens)
    val_ids   = tokenise_split("validation", max_val_tokens)
    print(f"done in {time.time()-t0:.1f}s  "
          f"(train={len(train_ids)/1e6:.1f}M tokens, "
          f"val={len(val_ids)/1e6:.1f}M tokens)")

    train_ds = TokenStreamDataset(train_ids, seq_len, shuffle=True,  seed=seed)
    val_ds   = TokenStreamDataset(val_ids,   seq_len, shuffle=False, seed=seed)

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Learning rate schedule: cosine decay with linear warmup
# ---------------------------------------------------------------------------

def cosine_lr(step: int, warmup_steps: int, total_steps: int, lr_max: float,
              lr_min_ratio: float = 0.1) -> float:
    """
    Linear warmup for warmup_steps, then cosine decay to lr_max * lr_min_ratio.
    """
    if step < warmup_steps:
        return lr_max * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_max * (lr_min_ratio + (1 - lr_min_ratio) * cosine)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: mHCWithMTP,
    val_ds: TokenStreamDataset,
    device: torch.device,
    n_batches: int = 50,
    batch_size: int = 16,
) -> dict:
    """
    Estimate validation loss over n_batches random batches from the val set.

    Returns a dict with val_main_loss, val_mtp_loss, val_total_loss, val_ppl.
    """
    model.eval()
    seq_len    = val_ds.seq_len
    loader_it  = iter(DataLoader(val_ds, batch_size=batch_size))

    total_loss = main_sum = mtp_sum = 0.0
    depth_sums = None
    count      = 0

    for _ in range(n_batches):
        try:
            batch = next(loader_it)
        except StopIteration:
            break

        # batch shape: (B, seq_len+1); we use only first seq_len tokens as input
        input_ids = batch[:, :seq_len].to(device)

        loss, metrics = model(input_ids)

        main_sum  += metrics["main_loss"].item()
        mtp_sum   += metrics["mtp_loss"].item()
        total_loss += loss.item()

        if depth_sums is None:
            depth_sums = [0.0] * len(metrics["per_depth_losses"])
        for k, dl in enumerate(metrics["per_depth_losses"]):
            depth_sums[k] += dl.item()

        count += 1

    if count == 0:
        return {}

    model.train()
    return {
        "val_total_loss":  total_loss / count,
        "val_main_loss":   main_sum   / count,
        "val_mtp_loss":    mtp_sum    / count,
        "val_ppl":         math.exp(min(main_sum / count, 20)),  # clamp against overflow
        "val_depth_losses": [s / count for s in (depth_sums or [])],
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: mHCWithMTP,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: float,
    cfg: ExperimentConfig,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step":           step,
        "best_val_loss":  best_val_loss,
        "model_state":    model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config":         dataclasses.asdict(cfg),
    }, path)


def load_checkpoint(
    path: Path,
    model: mHCWithMTP,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> tuple[int, float]:
    """Returns (step, best_val_loss)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt.get("step", 0), ckpt.get("best_val_loss", float("inf"))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train mHC + Sequential MTP language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # The one argument that distinguishes the two experimental conditions
    p.add_argument(
        "--reduction",
        choices=["sum", "mix"],
        default="sum",
        help=(
            "'sum' = baseline plain stream summation; "
            "'mix' = learned Sinkhorn doubly-stochastic mixing matrix. "
            "Only used when --model is 'mhc_mtp'."
        ),
    )

    p.add_argument(
        "--model",
        choices=["mhc_mtp", "residual_only", "mhc_only", "residual_mtp"],
        default="mhc_mtp",
        help=(
            "'mhc_mtp'     = mHC trunk + sequential MTP (main experiment, use --reduction to pick sum/mix); "
            "'residual_only'= Control 0: standard residual trunk, no MTP; "
            "'mhc_only'    = Control A: mHC trunk, no MTP; "
            "'residual_mtp'= Control B: standard residual trunk + sequential MTP"
        ),
    )

    # Model architecture
    p.add_argument("--vocab_size",  type=int, default=50257,
                   help="Vocabulary size (default matches GPT-2 / tiktoken)")
    p.add_argument("--d_model",     type=int, default=512)
    p.add_argument("--n_layers",    type=int, default=8,
                   help="Number of mHC transformer blocks in the trunk")
    p.add_argument("--n_heads",     type=int, default=8)
    p.add_argument("--n_streams",   type=int, default=4,
                   help="Number of parallel mHC residual streams")
    p.add_argument("--ffn_mult",    type=int, default=4)
    p.add_argument("--n_mtp",       type=int, default=2,
                   help="Number of sequential MTP prediction depths")
    p.add_argument("--mtp_loss_scale", type=float, default=0.3,
                   help="Weight applied to MTP loss before adding to main loss")
    p.add_argument("--sinkhorn_iters", type=int, default=20,
                   help="Sinkhorn iterations for mHC residuals and mix reduction")
    p.add_argument("--mhc_identity_epsilon", type=float, default=1e-3,
                   help="Initial off-diagonal mass before Sinkhorn for mHC near-identity mixers")
    p.add_argument("--mhc_gate_init", type=float, default=0.01,
                   help="Initial mHC gate alpha for identity/Sinkhorn residual mixer blend")

    # Training
    p.add_argument("--seq_len",     type=int, default=512)
    p.add_argument("--batch_size",  type=int, default=16,
                   help="Per-step batch size (before gradient accumulation)")
    p.add_argument("--grad_accum",  type=int, default=4,
                   help="Gradient accumulation steps")
    p.add_argument("--n_steps",     type=int, default=20000,
                   help="Total optimiser update steps")
    p.add_argument("--max_train_minutes", type=float, default=None,
                   help="Stop training after this many wall-clock minutes")
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--lr_min_ratio", type=float, default=0.1,
                   help="LR decays to lr * lr_min_ratio at end of training")
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--dataset", choices=["wikitext103", "wikitext2"],
                   default="wikitext103",
                   help="Training corpus; wikitext2 is useful for smoke tests")
    p.add_argument("--max_train_tokens", type=int, default=None,
                   help="Optional cap on tokenized train tokens for smoke tests")
    p.add_argument("--max_val_tokens", type=int, default=None,
                   help="Optional cap on tokenized validation tokens for smoke tests")

    # Checkpointing / logging
    p.add_argument("--run_dir",     type=str, default="runs",
                   help="Root directory for all run outputs")
    p.add_argument("--save_every",  type=int, default=500,
                   help="Save latest checkpoint every N steps")
    p.add_argument("--val_every",   type=int, default=250,
                   help="Run validation every N steps")
    p.add_argument("--log_every",   type=int, default=50,
                   help="Print training metrics every N steps")
    p.add_argument("--val_batches", type=int, default=50,
                   help="Number of validation batches to average")

    # Resume / eval
    p.add_argument("--resume",    type=str, default=None,
                   help="Path to a checkpoint to resume from")
    p.add_argument("--eval_only", action="store_true",
                   help="Only run evaluation on --resume checkpoint, then exit")

    # Hardware
    p.add_argument("--profile", choices=["default", "rtx2070"], default="default",
                   help="Apply hardware-aware defaults before user overrides")
    p.add_argument("--device", type=str, default=None,
                   help="Force device (e.g. 'cpu', 'cuda:0'); auto-detected if omitted")
    p.add_argument("--compile", action="store_true",
                   help="Use torch.compile() for ~20%% speedup (requires PyTorch 2+)")
    p.add_argument("--dtype",   type=str, default="auto",
                   choices=["auto", "float32", "float16", "bfloat16"],
                   help="'auto' uses float16 on pre-Ampere CUDA GPUs, bfloat16 on Ampere+")
    p.add_argument("--allow_tf32", action="store_true",
                   help="Enable TF32 matmul/cudnn on Ampere+ GPUs; ignored on RTX 2070")
    p.add_argument("--empty_cache_every", type=int, default=0,
                   help="Call torch.cuda.empty_cache() every N optimizer steps; 0 disables it")
    return p.parse_args()


def apply_hardware_profile(args):
    """
    Apply conservative defaults for known hardware profiles.

    argparse cannot tell which values came from the user with the current
    parser shape, so the profile only changes values that are still at the
    script defaults.
    """
    if args.profile != "rtx2070":
        return args

    profile_defaults = {
        "seq_len": (512, 256),
        "batch_size": (16, 2),
        "grad_accum": (4, 16),
        "d_model": (512, 384),
        "n_layers": (8, 6),
        "n_heads": (8, 6),
        "val_batches": (50, 10),
        "log_every": (50, 20),
    }
    for name, (old_default, profile_value) in profile_defaults.items():
        if getattr(args, name) == old_default:
            setattr(args, name, profile_value)
    if args.dtype == "auto":
        args.dtype = "float16"
    return args


def resolve_amp_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name != "auto":
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype_name]

    if device.type != "cuda":
        return torch.float32

    major, _minor = torch.cuda.get_device_capability(device)
    if major >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def warn_if_unsupported_dtype(dtype: torch.dtype, device: torch.device) -> torch.dtype:
    if device.type == "cuda" and dtype == torch.bfloat16:
        major, _minor = torch.cuda.get_device_capability(device)
        if major < 8 or not torch.cuda.is_bf16_supported():
            print("WARNING: bfloat16 is not supported well on this CUDA GPU; using float16.")
            return torch.float16
    return dtype


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = apply_hardware_profile(parse_args())

    # ---- Device ----
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU detected. Training will be very slow.")

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(device)}")
        print(f"VRAM:   {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        major, minor = torch.cuda.get_device_capability(device)
        print(f"CUDA capability: {major}.{minor}")
        if major < 8 and args.dtype in ("auto", "bfloat16"):
            print("Using float16 autocast; RTX 20-series GPUs do not have native bfloat16.")

    # ---- Output directory ----
    # Directory includes both model type and reduction so all four runs
    # land in separate folders and never overwrite each other.
    if args.model == "mhc_mtp":
        run_name = f"mhc_mtp_{args.reduction}"
    else:
        run_name = args.model
    run_dir = Path(args.run_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # ---- Tokenizer ----
    tokenizer, vocab_size_detected = get_tokenizer()
    # If user overrode vocab_size, respect it; otherwise use detected value
    vocab_size = args.vocab_size if args.vocab_size != 50257 else vocab_size_detected

    # ---- Build config ----
    cfg = ExperimentConfig(
        vocab_size    = vocab_size,
        d_model       = args.d_model,
        n_layers      = args.n_layers,
        n_heads       = args.n_heads,
        n_streams     = args.n_streams,
        ffn_mult      = args.ffn_mult,
        max_seq_len   = args.seq_len + 64,   # small buffer above seq_len
        n_mtp         = args.n_mtp,
        mtp_loss_scale = args.mtp_loss_scale,
        reduction     = args.reduction,
        sinkhorn_iters = args.sinkhorn_iters,
        mhc_identity_epsilon = args.mhc_identity_epsilon,
        mhc_gate_init = args.mhc_gate_init,
        lr            = args.lr,
        batch_size    = args.batch_size,
        seq_len       = args.seq_len,
        n_steps       = args.n_steps,
        log_every     = args.log_every,
        seed          = args.seed,
    )

    # Save config alongside checkpoints
    with open(run_dir / "config.json", "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    # ---- Model ----
    torch.manual_seed(cfg.seed)
    if args.model == "mhc_mtp":
        model = build_model(cfg).to(device)
    else:
        model = build_control(args.model, cfg).to(device)

    if args.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile() ...")
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel:  --model='{args.model}'  reduction='{cfg.reduction}'")
    print(f"Params: {n_params/1e6:.1f}M total")
    counts = model.parameter_count()
    print(f"        trunk={counts['trunk_total']/1e6:.1f}M  "
          f"mtp={counts['mtp_modules']/1e6:.1f}M  "
          f"mix_matrix={counts['mix_matrix']}")

    # Rough VRAM estimate
    bytes_per_param = 4   # float32; halved under CUDA autocast for many activations
    activation_multiplier = 8   # rough: activations ~8x params during forward
    estimated_gb = n_params * bytes_per_param * activation_multiplier / 1e9
    print(f"Estimated VRAM (rough): ~{estimated_gb:.1f} GB at fp32")

    # ---- Optimizer ----
    # Separate weight decay: apply to weight matrices, not biases or norms
    decay_params     = []
    no_decay_params  = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "norm" not in name and "embed" not in name:
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # ---- Dataset ----
    train_ds, val_ds = load_wikitext(
        args.seq_len,
        tokenizer,
        seed=cfg.seed,
        dataset=args.dataset,
        max_train_tokens=args.max_train_tokens,
        max_val_tokens=args.max_val_tokens,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=0,   # IterableDataset works fine with 0 workers
        pin_memory=(device.type == "cuda"),
    )
    train_iter = iter(train_loader)

    # ---- Resume ----
    start_step     = 0
    best_val_loss  = float("inf")

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"Resuming from {resume_path}")
            start_step, best_val_loss = load_checkpoint(
                resume_path, model, optimizer, device
            )
            print(f"  Resumed at step {start_step}, best_val_loss={best_val_loss:.4f}")
        else:
            print(f"WARNING: --resume path {resume_path} not found; starting fresh.")

    # ---- Eval-only mode ----
    if args.eval_only:
        if not args.resume:
            print("ERROR: --eval_only requires --resume")
            sys.exit(1)
        print("\nRunning evaluation ...")
        val_metrics = evaluate(model, val_ds, device,
                               n_batches=args.val_batches,
                               batch_size=args.batch_size)
        print(f"  val_total_loss : {val_metrics['val_total_loss']:.4f}")
        print(f"  val_main_loss  : {val_metrics['val_main_loss']:.4f}")
        print(f"  val_ppl        : {val_metrics['val_ppl']:.2f}")
        return

    # ---- Autocast dtype ----
    amp_dtype = warn_if_unsupported_dtype(resolve_amp_dtype(args.dtype, device), device)
    use_amp   = (device.type == "cuda") and (amp_dtype != torch.float32)
    print(f"Autocast dtype: {str(amp_dtype).replace('torch.', '') if use_amp else 'disabled'}")

    # ---- CSV logger ----
    csv_path = run_dir / "metrics.csv"
    csv_file = open(csv_path, "a", newline="")
    csv_cols = (
        ["step", "train_total_loss", "train_main_loss", "train_mtp_loss"]
        + [f"train_depth_{k}_loss" for k in range(cfg.n_mtp)]
        + ["val_total_loss", "val_main_loss", "val_ppl"]
        + [f"val_depth_{k}_loss" for k in range(cfg.n_mtp)]
        + [
            "mix_weight_diag",
            "mhc_row_err",
            "mhc_col_err",
            "mhc_diag_mass",
            "mhc_entropy",
            "mhc_sigma_1",
            "mhc_sigma_2",
            "mhc_reduction_row_err",
            "mhc_reduction_col_err",
            "mhc_reduction_diag_mass",
            "mhc_reduction_entropy",
            "mhc_reduction_sigma_1",
            "mhc_reduction_sigma_2",
            "lr",
            "tokens_per_sec",
            "elapsed_hours",
        ]
    )
    writer = csv.DictWriter(csv_file, fieldnames=csv_cols, extrasaction="ignore")
    if start_step == 0:
        writer.writeheader()
        csv_file.flush()

    # ---- Training loop ----
    print(f"\nTraining for {args.n_steps} steps  "
          f"(effective batch: {args.batch_size * args.grad_accum * args.seq_len:,} tokens/step)")
    print(f"Checkpoints -> {run_dir}")
    print(f"Metrics CSV -> {csv_path}\n")

    model.train()
    optimizer.zero_grad()

    scaler            = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    step              = start_step
    accum_count       = 0
    accum_loss        = 0.0
    accum_main        = 0.0
    accum_mtp         = 0.0
    accum_depths      = [0.0] * cfg.n_mtp
    last_mix_diag     = None
    last_mhc_diag     = {}
    t_loop_start      = time.perf_counter()
    t_step_start      = time.perf_counter()
    total_tokens      = 0
    train_start_time  = time.perf_counter()

    stop_reason = None
    max_train_seconds = (
        args.max_train_minutes * 60.0
        if args.max_train_minutes is not None
        else None
    )

    while step < args.n_steps:

        # ---- Fetch batch ----
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # batch: (B, seq_len+1); take first seq_len tokens as input
        input_ids = batch[:, :args.seq_len].to(device, non_blocking=True)

        # ---- Forward ----
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            total_loss, metrics = model(input_ids)
            # Scale loss for gradient accumulation
            loss = total_loss / args.grad_accum

        # ---- Backward ----
        if use_amp and amp_dtype == torch.float16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Accumulate metrics (unscaled for logging)
        accum_loss  += total_loss.item()
        accum_main  += metrics["main_loss"].item()
        accum_mtp   += metrics["mtp_loss"].item()
        for k, dl in enumerate(metrics["per_depth_losses"]):
            accum_depths[k] += dl.item()
        if metrics["mix_weights"] is not None:
            last_mix_diag = metrics["mix_weights"].diagonal().tolist()
        last_mhc_diag = metrics.get("mhc_diagnostics", {}) or {}
        total_tokens += input_ids.numel()
        accum_count  += 1

        # ---- Optimizer step (every grad_accum micro-steps) ----
        if accum_count < args.grad_accum:
            continue

        # Update learning rate
        current_lr = cosine_lr(step, args.warmup_steps, args.n_steps, args.lr,
                                args.lr_min_ratio)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Gradient clipping + optimizer step
        if use_amp and amp_dtype == torch.float16:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if use_amp and amp_dtype == torch.float16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        step        += 1
        if args.empty_cache_every and device.type == "cuda" and step % args.empty_cache_every == 0:
            torch.cuda.empty_cache()
        accum_count  = 0

        t_now     = time.perf_counter()
        step_time = t_now - t_step_start
        t_step_start = t_now
        tps       = total_tokens / max(t_now - t_loop_start, 1e-6)
        elapsed_h = (t_now - train_start_time) / 3600

        # Average accumulated metrics
        avg_loss   = accum_loss   / args.grad_accum
        avg_main   = accum_main   / args.grad_accum
        avg_mtp    = accum_mtp    / args.grad_accum
        avg_depths = [d / args.grad_accum for d in accum_depths]

        accum_loss   = 0.0
        accum_main   = 0.0
        accum_mtp    = 0.0
        accum_depths = [0.0] * cfg.n_mtp
        total_tokens = 0
        t_loop_start = t_now

        # ---- Logging ----
        if step % args.log_every == 0 or step == 1:
            depth_str = "  ".join(
                f"d{k}={avg_depths[k]:.3f}" for k in range(cfg.n_mtp)
            )
            mix_str = ""
            if last_mix_diag is not None:
                mix_str = f"  mix_diag=[{', '.join(f'{x:.3f}' for x in last_mix_diag)}]"
            mhc_str = ""
            if last_mhc_diag:
                mhc_str = (
                    f"  mhc_diag={last_mhc_diag.get('mhc_diag_mass', 0.0):.3f}"
                    f" ent={last_mhc_diag.get('mhc_entropy', 0.0):.3f}"
                    f" s2={last_mhc_diag.get('mhc_sigma_2', 0.0):.3f}"
                )
            eta_h = (args.n_steps - step) * (elapsed_h / max(step - start_step, 1))
            print(
                f"step {step:5d}/{args.n_steps} | "
                f"loss={avg_loss:.4f}  main={avg_main:.4f}  mtp={avg_mtp:.4f}  "
                f"[{depth_str}]{mix_str}{mhc_str}  "
                f"lr={current_lr:.2e}  "
                f"tps={tps:,.0f}  "
                f"elapsed={elapsed_h:.2f}h  eta={eta_h:.1f}h"
            )

        # ---- Validation ----
        val_metrics = {}
        if step % args.val_every == 0:
            val_metrics = evaluate(
                model, val_ds, device,
                n_batches=args.val_batches,
                batch_size=args.batch_size,
            )
            depth_val_str = "  ".join(
                f"d{k}={val_metrics['val_depth_losses'][k]:.3f}"
                for k in range(cfg.n_mtp)
            ) if val_metrics.get("val_depth_losses") else ""
            print(
                f"  [VAL] step {step:5d} | "
                f"val_loss={val_metrics['val_total_loss']:.4f}  "
                f"main={val_metrics['val_main_loss']:.4f}  "
                f"ppl={val_metrics['val_ppl']:.2f}  "
                f"[{depth_val_str}]"
            )

            # Save best checkpoint
            if val_metrics["val_total_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_total_loss"]
                save_checkpoint(
                    run_dir / "checkpoint_best.pt",
                    model, optimizer, step, best_val_loss, cfg,
                )
                print(f"  [BEST] New best val loss: {best_val_loss:.4f} — saved.")

        # ---- Periodic checkpoint ----
        if step % args.save_every == 0:
            save_checkpoint(
                run_dir / "checkpoint_latest.pt",
                model, optimizer, step, best_val_loss, cfg,
            )

        # ---- CSV row ----
        row = {
            "step":              step,
            "train_total_loss":  avg_loss,
            "train_main_loss":   avg_main,
            "train_mtp_loss":    avg_mtp,
            "val_total_loss":    val_metrics.get("val_total_loss", ""),
            "val_main_loss":     val_metrics.get("val_main_loss", ""),
            "val_ppl":           val_metrics.get("val_ppl", ""),
            "mix_weight_diag":   str(last_mix_diag) if last_mix_diag else "",
            "mhc_row_err":       last_mhc_diag.get("mhc_row_err", ""),
            "mhc_col_err":       last_mhc_diag.get("mhc_col_err", ""),
            "mhc_diag_mass":     last_mhc_diag.get("mhc_diag_mass", ""),
            "mhc_entropy":       last_mhc_diag.get("mhc_entropy", ""),
            "mhc_sigma_1":       last_mhc_diag.get("mhc_sigma_1", ""),
            "mhc_sigma_2":       last_mhc_diag.get("mhc_sigma_2", ""),
            "mhc_reduction_row_err":   last_mhc_diag.get("mhc_reduction_row_err", ""),
            "mhc_reduction_col_err":   last_mhc_diag.get("mhc_reduction_col_err", ""),
            "mhc_reduction_diag_mass": last_mhc_diag.get("mhc_reduction_diag_mass", ""),
            "mhc_reduction_entropy":   last_mhc_diag.get("mhc_reduction_entropy", ""),
            "mhc_reduction_sigma_1":   last_mhc_diag.get("mhc_reduction_sigma_1", ""),
            "mhc_reduction_sigma_2":   last_mhc_diag.get("mhc_reduction_sigma_2", ""),
            "lr":                current_lr,
            "tokens_per_sec":    tps,
            "elapsed_hours":     elapsed_h,
        }
        for k in range(cfg.n_mtp):
            row[f"train_depth_{k}_loss"] = avg_depths[k]
            if val_metrics.get("val_depth_losses"):
                row[f"val_depth_{k}_loss"] = val_metrics["val_depth_losses"][k]
        writer.writerow(row)
        csv_file.flush()

        if max_train_seconds is not None:
            elapsed_s = time.perf_counter() - train_start_time
            if elapsed_s >= max_train_seconds:
                stop_reason = (
                    f"Reached --max_train_minutes={args.max_train_minutes:g} "
                    f"after step {step}."
                )
                break

    # ---- End of training ----
    total_time_h = (time.perf_counter() - train_start_time) / 3600
    if stop_reason:
        print(f"\nStopping early: {stop_reason}")
    print(f"\nTraining complete. {step} steps in {total_time_h:.2f} hours.")

    # Final checkpoint
    save_checkpoint(
        run_dir / "checkpoint_final.pt",
        model, optimizer, step, best_val_loss, cfg,
    )
    print(f"Final checkpoint saved to {run_dir / 'checkpoint_final.pt'}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Metrics: {csv_path}")

    csv_file.close()


if __name__ == "__main__":
    main()
