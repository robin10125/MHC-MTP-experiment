#!/usr/bin/env python3
"""Plot validation loss comparisons across training runs.

Reads each run's `metrics.csv` and produces:
  - validation loss versus optimizer step
  - validation loss versus elapsed wall-clock minutes
  - train loss versus optimizer step
  - train loss versus elapsed wall-clock minutes

Example:
    python plot_validation_comparison.py --run_dir runs_baselines_20min
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Series:
    name: str
    steps: list[int]
    minutes: list[float]
    val_total: list[float]
    val_main: list[float]
    train_total_steps: list[int]
    train_total_minutes: list[float]
    train_total: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare validation loss over training step and time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run_dir", default="runs_baselines_20min")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Specific model run directories to plot. Defaults to every metrics.csv under run_dir.",
    )
    parser.add_argument("--title", default="Validation Loss Comparison")
    parser.add_argument("--format", choices=["png", "svg", "pdf"], default="png")
    parser.add_argument(
        "--train_smooth",
        type=int,
        default=50,
        help="Moving-average window for train loss plots. Use 1 for raw per-step loss.",
    )
    return parser.parse_args()


def read_series(metrics_path: Path) -> Series:
    name = metrics_path.parent.name
    steps: list[int] = []
    minutes: list[float] = []
    val_total: list[float] = []
    val_main: list[float] = []
    train_steps: list[int] = []
    train_minutes: list[float] = []
    train_total: list[float] = []

    with metrics_path.open(newline="") as f:
        for row in csv.DictReader(f):
            step = int(row["step"])
            elapsed_minutes = float(row["elapsed_hours"]) * 60.0
            if row.get("train_total_loss"):
                train_steps.append(step)
                train_minutes.append(elapsed_minutes)
                train_total.append(float(row["train_total_loss"]))
            if row.get("val_total_loss"):
                steps.append(step)
                minutes.append(elapsed_minutes)
                val_total.append(float(row["val_total_loss"]))
                val_main.append(float(row["val_main_loss"]))

    if not val_total:
        raise ValueError(f"No validation rows found in {metrics_path}")

    return Series(
        name=name,
        steps=steps,
        minutes=minutes,
        val_total=val_total,
        val_main=val_main,
        train_total_steps=train_steps,
        train_total_minutes=train_minutes,
        train_total=train_total,
    )


def find_metric_files(run_dir: Path, models: list[str] | None) -> list[Path]:
    if models:
        paths = [run_dir / model / "metrics.csv" for model in models]
    else:
        paths = sorted(run_dir.glob("*/metrics.csv"))
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing metrics files:\n" + "\n".join(missing))
    return paths


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    smoothed: list[float] = []
    running = 0.0
    for idx, value in enumerate(values):
        running += value
        if idx >= window:
            running -= values[idx - window]
        denom = min(idx + 1, window)
        smoothed.append(running / denom)
    return smoothed


def plot_lines(
    series: list[Series],
    x_attr: str,
    y_attr: str,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    smooth_window: int = 1,
) -> None:
    import matplotlib.pyplot as plt

    styles = {
        "residual_only": {"linestyle": "-", "marker": "o", "zorder": 2},
        "mhc_only": {"linestyle": "-", "marker": "s", "zorder": 2},
        "residual_mtp": {"linestyle": "-", "marker": "^", "zorder": 2},
        "mhc_mtp_sum": {"linestyle": "--", "marker": "D", "zorder": 3},
        "mhc_mtp_mix": {"linestyle": ":", "marker": "x", "zorder": 4},
    }
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    for item in series:
        x = getattr(item, x_attr)
        y = getattr(item, y_attr)
        if not x or not y:
            continue
        y = moving_average(y, smooth_window)
        style = styles.get(item.name, {"linestyle": "-", "marker": "o", "zorder": 2})
        ax.plot(
            x,
            y,
            linewidth=1.8,
            markersize=4.0,
            label=item.name,
            **style,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    series = [read_series(path) for path in find_metric_files(run_dir, args.models)]

    plot_lines(
        series,
        "steps",
        "val_total",
        "Training step",
        "Validation total loss",
        f"{args.title}: validation loss vs step",
        out_dir / f"validation_loss_vs_step.{args.format}",
    )
    plot_lines(
        series,
        "minutes",
        "val_total",
        "Elapsed minutes",
        "Validation total loss",
        f"{args.title}: validation loss vs time",
        out_dir / f"validation_loss_vs_time.{args.format}",
    )
    plot_lines(
        series,
        "minutes",
        "val_main",
        "Elapsed minutes",
        "Validation main loss",
        f"{args.title}: validation main loss vs time",
        out_dir / f"validation_main_loss_vs_time.{args.format}",
    )
    plot_lines(
        series,
        "train_total_minutes",
        "train_total",
        "Elapsed minutes",
        f"Training total loss ({args.train_smooth}-step moving average)",
        f"{args.title}: training loss vs time",
        out_dir / f"training_loss_vs_time.{args.format}",
        smooth_window=args.train_smooth,
    )
    plot_lines(
        series,
        "train_total_steps",
        "train_total",
        "Training step",
        f"Training total loss ({args.train_smooth}-step moving average)",
        f"{args.title}: training loss vs step",
        out_dir / f"training_loss_vs_step.{args.format}",
        smooth_window=args.train_smooth,
    )

    summary_path = out_dir / "validation_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "best_val_step",
                "best_val_minutes",
                "best_val_total",
                "best_val_main",
                "final_val_step",
                "final_val_minutes",
                "final_val_total",
                "final_val_main",
            ],
        )
        writer.writeheader()
        for item in series:
            best_idx = min(range(len(item.val_total)), key=item.val_total.__getitem__)
            final_idx = len(item.val_total) - 1
            writer.writerow({
                "model": item.name,
                "best_val_step": item.steps[best_idx],
                "best_val_minutes": item.minutes[best_idx],
                "best_val_total": item.val_total[best_idx],
                "best_val_main": item.val_main[best_idx],
                "final_val_step": item.steps[final_idx],
                "final_val_minutes": item.minutes[final_idx],
                "final_val_total": item.val_total[final_idx],
                "final_val_main": item.val_main[final_idx],
            })

    print(f"Wrote plots to {out_dir}")
    print(f"Wrote summary to {summary_path}")

    train_summary_path = out_dir / "training_summary.csv"
    with train_summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "first_train_step",
                "first_train_minutes",
                "first_train_total",
                "best_train_step",
                "best_train_minutes",
                "best_train_total",
                "final_train_step",
                "final_train_minutes",
                "final_train_total",
            ],
        )
        writer.writeheader()
        for item in series:
            if not item.train_total:
                continue
            best_idx = min(range(len(item.train_total)), key=item.train_total.__getitem__)
            final_idx = len(item.train_total) - 1
            writer.writerow({
                "model": item.name,
                "first_train_step": item.train_total_steps[0],
                "first_train_minutes": item.train_total_minutes[0],
                "first_train_total": item.train_total[0],
                "best_train_step": item.train_total_steps[best_idx],
                "best_train_minutes": item.train_total_minutes[best_idx],
                "best_train_total": item.train_total[best_idx],
                "final_train_step": item.train_total_steps[final_idx],
                "final_train_minutes": item.train_total_minutes[final_idx],
                "final_train_total": item.train_total[final_idx],
            })
    print(f"Wrote train summary to {train_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
