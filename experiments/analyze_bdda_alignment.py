"""Utility script to inspect BDDA batches and measure timestamp alignment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from args import parse_args_main
import datasets


def parse_alignment_args():
    """
    Parse custom alignment args first, then delegate the rest to the main Symile parser.
    """
    original_argv = sys.argv[:]
    helper = argparse.ArgumentParser(add_help=False)
    helper.add_argument("--batch-index", type=int, default=0,
                        help="Which training batch to inspect (0 = first batch).")
    helper.add_argument("--max-negative-samples", type=int, default=5000,
                        help="How many negative-pair offsets to subsample when plotting.")
    helper.add_argument("--plot-dir", type=Path, default=Path("alignment_plots"),
                        help="Directory where offset histograms will be saved.")
    helper.add_argument("--positive-threshold", type=float, default=0.2,
                        help="Threshold (seconds) for reporting positive alignment success.")
    helper_args, remaining = helper.parse_known_args()

    try:
        sys.argv = [original_argv[0]] + remaining
        args = parse_args_main()
    finally:
        sys.argv = original_argv

    if args.experiment != "symile_bdda":
        raise ValueError("This analysis script currently only supports --experiment symile_bdda.")

    setattr(args, "batch_index_to_inspect", max(helper_args.batch_index, 0))
    setattr(args, "max_negative_samples", max(helper_args.max_negative_samples, 1))
    setattr(args, "plot_dir", helper_args.plot_dir)
    setattr(args, "positive_threshold", max(helper_args.positive_threshold, 0.0))
    return args


def fetch_batch(dm: datasets.SymileBDDADataModule, batch_index: int):
    loader = dm.train_dataloader()
    for idx, batch in enumerate(loader):
        if idx == batch_index:
            return batch
    raise IndexError(f"Requested batch_index {batch_index} but train loader exhausted.")


def print_batch_overview(batch):
    timestamps = batch["timestamp"].cpu().numpy()
    frame_idx = batch["frame_idx"].cpu().numpy()
    sample_ids = batch["sample_id"]
    print("\n=== Sample overview ===")
    print("idx\tframe_idx\ttimestamp(s)\tsample_id")
    for idx, (frame, ts, sid) in enumerate(zip(frame_idx, timestamps, sample_ids)):
        print(f"{idx:02d}\t{int(frame):06d}\t{ts:8.3f}\t{sid}")


def compute_offsets(timestamps: torch.Tensor):
    ts = timestamps.cpu().numpy().astype(np.float64)
    diff = ts[:, None] - ts[None, :]
    abs_diff = np.abs(diff)
    positive = np.diag(abs_diff)
    mask = ~np.eye(abs_diff.shape[0], dtype=bool)
    negatives = abs_diff[mask]
    return positive, negatives


def summarise_offsets(name: str, positives: np.ndarray, negatives: np.ndarray, threshold: float):
    def describe(values: np.ndarray):
        return {
            "count": int(values.size),
            "mean": float(values.mean()) if values.size else float("nan"),
            "std": float(values.std()) if values.size else float("nan"),
            "p50": float(np.percentile(values, 50)) if values.size else float("nan"),
            "p90": float(np.percentile(values, 90)) if values.size else float("nan"),
            "max": float(values.max()) if values.size else float("nan"),
        }

    within_thresh = float((positives <= threshold).sum()) / max(positives.size, 1)
    print(f"\n[{name}] positive offsets stats: {describe(positives)}")
    print(f"[{name}] fraction of positives within {threshold:.3f}s: {within_thresh:.3f}")
    print(f"[{name}] negative offsets stats: {describe(negatives)}")


def plot_offsets(name: str,
                 positives: np.ndarray,
                 negatives: np.ndarray,
                 max_negative_samples: int,
                 plot_dir: Path):
    plot_dir.mkdir(parents=True, exist_ok=True)
    neg = negatives
    if neg.size > max_negative_samples:
        rng = np.random.default_rng(seed=0)
        neg = rng.choice(neg, size=max_negative_samples, replace=False)

    plt.figure(figsize=(8, 4))
    bins = max(10, min(50, positives.size))
    plt.hist(positives, bins=bins, alpha=0.8, label="positive pairs")
    plt.hist(neg, bins=bins, alpha=0.5, label="negative pairs")
    plt.xlabel("Absolute time offset (s)")
    plt.ylabel("Count")
    plt.title(f"{name} offsets")
    plt.legend()
    out_path = plot_dir / f"{name}_offsets.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[{name}] Saved histogram to {out_path}")


def main():
    args = parse_alignment_args()
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    dm = datasets.SymileBDDADataModule(args)
    dm.setup(stage="fit")

    batch = fetch_batch(dm, args.batch_index_to_inspect)
    print_batch_overview(batch)

    timestamps = batch["timestamp"]
    for pair_name in ("camera_gaze", "camera_gps", "gaze_gps"):
        positives, negatives = compute_offsets(timestamps)
        summarise_offsets(pair_name, positives, negatives, args.positive_threshold)
        plot_offsets(pair_name, positives, negatives, args.max_negative_samples, args.plot_dir)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
