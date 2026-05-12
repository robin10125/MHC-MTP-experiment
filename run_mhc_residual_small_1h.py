#!/usr/bin/env python3
"""Run a one-hour paired residual-vs-mHC small-model test.

This script is intentionally narrower than the all-baselines smoke test: it
only runs the no-MTP residual baseline and the no-MTP mHC model with matching
architecture/training settings, then prints a compact comparison from the
resulting metrics.csv files.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path


RUNS = [
    ("residual_only", ["--model", "residual_only"]),
    ("mhc_only", ["--model", "mhc_only"]),
]


def default_python() -> str:
    venv_python = Path(".venv/bin/python")
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small residual-vs-mHC comparison inside a wall-clock budget.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--python", default=default_python())
    parser.add_argument("--run_dir", default="runs_mhc_residual_small_1h")
    parser.add_argument("--keep_outputs", action="store_true")
    parser.add_argument("--time_budget_minutes", type=float, default=60.0)
    parser.add_argument("--minutes_per_model", type=float, default=27.0)
    parser.add_argument("--dataset", choices=["wikitext2", "wikitext103"], default="wikitext2")
    parser.add_argument("--max_train_tokens", type=int, default=2_000_000)
    parser.add_argument("--max_val_tokens", type=int, default=100_000)
    parser.add_argument("--n_steps", type=int, default=10_000)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_streams", type=int, default=4)
    parser.add_argument("--ffn_mult", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--sinkhorn_iters", type=int, default=20)
    parser.add_argument("--mhc_gate_init", type=float, default=0.01)
    parser.add_argument("--val_every", type=int, default=100)
    parser.add_argument("--val_batches", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def read_rows(metrics_path: Path) -> list[dict[str, str]]:
    with metrics_path.open(newline="") as f:
        return list(csv.DictReader(f))


def best_val(rows: list[dict[str, str]]) -> dict[str, str] | None:
    val_rows = [row for row in rows if row.get("val_main_loss")]
    if not val_rows:
        return None
    return min(val_rows, key=lambda row: float(row["val_main_loss"]))


def last_train(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        raise RuntimeError("metrics.csv has no rows")
    return rows[-1]


def fmt_float(value: str | None, digits: int = 4) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.{digits}f}"


def summarize(root: Path, timings: dict[str, float]) -> None:
    print("\nSmall mHC vs residual summary")
    print(
        "model,steps,minutes,train_main_loss,best_val_main_loss,"
        "best_val_step,tokens_per_sec,mhc_diag_mass,mhc_entropy,mhc_sigma_2"
    )
    for label, _args in RUNS:
        rows = read_rows(root / label / "metrics.csv")
        last = last_train(rows)
        best = best_val(rows)
        print(
            ",".join(
                [
                    label,
                    last["step"],
                    f"{timings[label] / 60.0:.1f}",
                    fmt_float(last.get("train_main_loss")),
                    fmt_float(best.get("val_main_loss") if best else None),
                    best.get("step", "") if best else "",
                    fmt_float(last.get("tokens_per_sec"), digits=0),
                    fmt_float(last.get("mhc_diag_mass")),
                    fmt_float(last.get("mhc_entropy")),
                    fmt_float(last.get("mhc_sigma_2")),
                ]
            )
        )


def main() -> int:
    args = parse_args()
    root = Path(args.run_dir)
    if root.exists() and not args.keep_outputs:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    common = [
        "train.py",
        "--profile", "rtx2070",
        "--dataset", args.dataset,
        "--max_train_tokens", str(args.max_train_tokens),
        "--max_val_tokens", str(args.max_val_tokens),
        "--n_steps", str(args.n_steps),
        "--max_train_minutes", str(args.minutes_per_model),
        "--seq_len", str(args.seq_len),
        "--batch_size", str(args.batch_size),
        "--grad_accum", str(args.grad_accum),
        "--d_model", str(args.d_model),
        "--n_layers", str(args.n_layers),
        "--n_heads", str(args.n_heads),
        "--n_streams", str(args.n_streams),
        "--ffn_mult", str(args.ffn_mult),
        "--n_mtp", "1",
        "--lr", str(args.lr),
        "--warmup_steps", str(args.warmup_steps),
        "--weight_decay", str(args.weight_decay),
        "--sinkhorn_iters", str(args.sinkhorn_iters),
        "--mhc_gate_init", str(args.mhc_gate_init),
        "--val_every", str(args.val_every),
        "--val_batches", str(args.val_batches),
        "--log_every", str(args.log_every),
        "--save_every", str(args.save_every),
        "--seed", str(args.seed),
        "--run_dir", str(root),
    ]
    if args.compile:
        common.append("--compile")

    started = time.perf_counter()
    budget_seconds = args.time_budget_minutes * 60.0
    timings: dict[str, float] = {}
    for label, model_args in RUNS:
        remaining = budget_seconds - (time.perf_counter() - started)
        if remaining <= 0:
            raise TimeoutError(f"Time budget exceeded before starting {label}")
        cmd = [args.python, *common, *model_args]
        log_path = root / f"{label}.log"
        print(f"\n=== {label} ===")
        print(" ".join(cmd))
        print(f"Log -> {log_path}")
        run_started = time.perf_counter()
        with log_path.open("w") as log_file:
            subprocess.run(
                cmd,
                check=True,
                timeout=remaining,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        timings[label] = time.perf_counter() - run_started

    summarize(root, timings)
    total_minutes = (time.perf_counter() - started) / 60.0
    print(f"total_minutes,{total_minutes:.1f}")
    print(f"run_dir,{root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
