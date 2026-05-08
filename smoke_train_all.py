#!/usr/bin/env python3
"""Run a bounded smoke training sweep across all experiment baselines.

The goal is not model quality. This script checks that every comparison arm can
instantiate, load data, train, validate, write metrics, and exit on the target
GPU within a short wall-clock budget.
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
    ("residual_mtp", ["--model", "residual_mtp"]),
    ("mhc_mtp_sum", ["--model", "mhc_mtp", "--reduction", "sum"]),
    ("mhc_mtp_mix", ["--model", "mhc_mtp", "--reduction", "mix"]),
]


def default_python() -> str:
    venv_python = Path(".venv/bin/python")
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke train every baseline with a tiny model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--python", default=default_python(), help="Python executable")
    parser.add_argument("--run_dir", default="runs_smoke_all", help="Output directory")
    parser.add_argument("--time_budget_minutes", type=float, default=30.0)
    parser.add_argument("--keep_outputs", action="store_true", help="Do not delete old run_dir first")
    parser.add_argument("--dataset", choices=["wikitext2", "wikitext103"], default="wikitext2")
    parser.add_argument("--max_train_tokens", type=int, default=100_000)
    parser.add_argument("--max_val_tokens", type=int, default=20_000)
    parser.add_argument("--n_steps", type=int, default=3)
    parser.add_argument("--val_batches", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_streams", type=int, default=4)
    parser.add_argument("--n_mtp", type=int, default=1)
    parser.add_argument("--sinkhorn_iters", type=int, default=5)
    return parser.parse_args()


def read_last_metrics(run_dir: Path, run_name: str) -> dict[str, str]:
    metrics_path = run_dir / run_name / "metrics.csv"
    with metrics_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No metrics rows found in {metrics_path}")
    return rows[-1]


def main() -> int:
    args = parse_args()
    root = Path(args.run_dir)
    if root.exists() and not args.keep_outputs:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    budget_seconds = args.time_budget_minutes * 60.0
    results = []

    common = [
        "train.py",
        "--profile", "rtx2070",
        "--dataset", args.dataset,
        "--max_train_tokens", str(args.max_train_tokens),
        "--max_val_tokens", str(args.max_val_tokens),
        "--n_steps", str(args.n_steps),
        "--val_every", str(args.n_steps),
        "--save_every", "999999",
        "--log_every", str(args.n_steps),
        "--val_batches", str(args.val_batches),
        "--seq_len", str(args.seq_len),
        "--batch_size", str(args.batch_size),
        "--grad_accum", str(args.grad_accum),
        "--d_model", str(args.d_model),
        "--n_layers", str(args.n_layers),
        "--n_heads", str(args.n_heads),
        "--n_streams", str(args.n_streams),
        "--n_mtp", str(args.n_mtp),
        "--sinkhorn_iters", str(args.sinkhorn_iters),
        "--run_dir", str(root),
    ]

    for label, model_args in RUNS:
        elapsed = time.perf_counter() - started
        remaining = budget_seconds - elapsed
        if remaining <= 0:
            raise TimeoutError(f"Smoke test budget exceeded before {label}")

        cmd = [args.python, *common, *model_args]
        print(f"\n=== {label} ===", flush=True)
        print(" ".join(cmd), flush=True)
        run_started = time.perf_counter()
        subprocess.run(cmd, check=True, timeout=remaining)
        run_seconds = time.perf_counter() - run_started

        run_name = "mhc_mtp_sum" if label == "mhc_mtp_sum" else (
            "mhc_mtp_mix" if label == "mhc_mtp_mix" else label
        )
        row = read_last_metrics(root, run_name)
        results.append((label, run_seconds, row))

    total_seconds = time.perf_counter() - started
    print("\nSmoke test summary")
    print("model,total_loss,main_loss,mtp_loss,val_total_loss,tokens_per_sec,seconds")
    for label, run_seconds, row in results:
        print(
            ",".join([
                label,
                row.get("train_total_loss", ""),
                row.get("train_main_loss", ""),
                row.get("train_mtp_loss", ""),
                row.get("val_total_loss", ""),
                row.get("tokens_per_sec", ""),
                f"{run_seconds:.1f}",
            ])
        )
    print(f"total_seconds,{total_seconds:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
