#!/usr/bin/env python3
"""Quality checks for gesture HDF5 datasets with lightweight WandB logging."""

from __future__ import annotations

import argparse
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import wandb

from conditioned_gesture_generator.train_cnn_gesture_classifier_no_int import (
    H5ActionSequenceDataset,
)


class IndexedSubset(Dataset):
    """Wrap a dataset so that each sample also returns its original index."""

    def __init__(self, base: H5ActionSequenceDataset, indices: Sequence[int]):
        self._base = base
        self._indices = list(indices)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        base_idx = self._indices[idx]
        return self._base[base_idx], base_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quality checks over an augmented gesture dataset")
    parser.add_argument("--data-dir", required=True, help="Directory containing HDF5 trajectory files")
    parser.add_argument("--manifest", help="Optional manifest JSON limiting which files to scan")
    parser.add_argument("--split", default="train", help="Manifest split to inspect (default: train)")
    parser.add_argument("--max-samples", type=int, default=0, help="Stop after this many sequences (0 = all)")
    parser.add_argument("--limit-files", type=int, help="Only inspect the first N files referenced in the manifest")
    parser.add_argument(
        "--max-segments-per-file",
        type=int,
        help="Limit number of segments pulled from each file (after filtering)",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size used for streaming")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--zero-eps", type=float, default=1e-6, help="Absolute tolerance when counting zeros")
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=0.0,
        help="Pressure/action channel is counted active when greater than this",
    )
    parser.add_argument("--hist-bins", type=int, default=50, help="Number of bins for XY histograms")
    parser.add_argument(
        "--hist-range",
        type=float,
        nargs=2,
        default=(0.0, 1.0),
        metavar=("MIN", "MAX"),
        help="Value range for XY histograms",
    )
    parser.add_argument(
        "--duplicate-sample-size",
        type=int,
        default=5000,
        help="Number of sequences to hash when estimating duplicates",
    )
    parser.add_argument(
        "--duplicate-top-k",
        type=int,
        default=10,
        help="Number of duplicate hashes to surface in the report",
    )
    parser.add_argument(
        "--file-table-limit",
        type=int,
        default=20,
        help="Limit of per-file rows to log (keeps WandB tables small)",
    )
    parser.add_argument("--wandb-project", default="dataset_qa", help="WandB project name")
    parser.add_argument("--wandb-entity", help="Optional WandB entity/org")
    parser.add_argument("--wandb-run-name", help="Optional WandB run name override")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
        help="Logging mode (disabled prints results locally)",
    )
    parser.add_argument("--tag", dest="tags", action="append", default=[], help="Extra WandB tags")
    return parser.parse_args()


def build_index_list(
    dataset: H5ActionSequenceDataset,
    max_samples: int,
    limit_files: Optional[int],
    max_segments_per_file: Optional[int],
) -> List[int]:
    allowed_files: Optional[set[int]] = None
    if limit_files is not None:
        allowed_files = set(range(min(limit_files, len(dataset.files))))

    per_file_counter: Dict[int, int] = defaultdict(int)
    indices: List[int] = []

    for base_idx, (file_id, _) in enumerate(dataset.segment_index):
        if allowed_files is not None and file_id not in allowed_files:
            continue
        if max_segments_per_file is not None and per_file_counter[file_id] >= max_segments_per_file:
            continue
        indices.append(base_idx)
        per_file_counter[file_id] += 1
        if max_samples and len(indices) >= max_samples:
            break

    return indices


def describe_segment(dataset: H5ActionSequenceDataset, base_idx: int) -> str:
    file_id, entry = dataset.segment_index[base_idx]
    file_name = dataset.files[file_id].name
    if entry[1] is None:
        segment_desc = f"segment={entry[0]}"
    else:
        segment_desc = f"grid=({entry[0]}, {entry[1]})"
    return f"{file_name}:{segment_desc}"


def close_dataset_handles(dataset: H5ActionSequenceDataset) -> None:
    for worker_handles in dataset._file_handles.values():
        for handle in worker_handles.values():
            handle.close()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    manifest_path = Path(args.manifest) if args.manifest else None

    dataset = H5ActionSequenceDataset(
        processed_data_dir=data_dir,
        manifest_path=str(manifest_path) if manifest_path else None,
        split=args.split,
    )

    indices = build_index_list(dataset, args.max_samples, args.limit_files, args.max_segments_per_file)
    if not indices:
        raise RuntimeError("No segments selected for QA; adjust sampling parameters")

    subset = IndexedSubset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    wandb_run = None
    if args.wandb_mode != "disabled":
        init_kwargs = {
            "project": args.wandb_project,
            "config": {
                "data_dir": str(data_dir.resolve()),
                "manifest": str(manifest_path.resolve()) if manifest_path else None,
                "split": args.split,
                "max_samples": args.max_samples,
                "limit_files": args.limit_files,
                "max_segments_per_file": args.max_segments_per_file,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "zero_eps": args.zero_eps,
                "active_threshold": args.active_threshold,
                "hist_bins": args.hist_bins,
                "hist_range": args.hist_range,
                "duplicate_sample_size": args.duplicate_sample_size,
            },
            "tags": args.tags or None,
            "name": args.wandb_run_name,
        }
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        if args.wandb_mode in {"online", "offline"}:
            init_kwargs["mode"] = args.wandb_mode
        wandb_run = wandb.init(**{k: v for k, v in init_kwargs.items() if v is not None})

    hist_range = tuple(args.hist_range)
    hist_edges = np.linspace(hist_range[0], hist_range[1], args.hist_bins + 1)
    x_hist = np.zeros(args.hist_bins, dtype=np.int64)
    y_hist = np.zeros(args.hist_bins, dtype=np.int64)
    x_outside_low = 0
    x_outside_high = 0
    y_outside_low = 0
    y_outside_high = 0

    seq_length: Optional[int] = None
    channel_count: Optional[int] = None

    total_sequences = 0
    total_timesteps = 0

    x_sum = 0.0
    x_sumsq = 0.0
    x_min = math.inf
    x_max = -math.inf

    y_sum = 0.0
    y_sumsq = 0.0
    y_min = math.inf
    y_max = -math.inf

    x_zero_count = 0
    y_zero_count = 0
    xy_zero_count = 0

    press_active_count = 0
    press_total_count = 0

    file_sequence_counts: Dict[int, int] = defaultdict(int)
    file_zero_counts: Dict[int, int] = defaultdict(int)
    file_timestep_counts: Dict[int, int] = defaultdict(int)
    file_active_counts: Dict[int, int] = defaultdict(int)

    duplicate_info: Dict[str, Dict[str, object]] = {}
    hashed_sequences = 0

    for batch_data, base_indices in loader:
        batch_np = batch_data.numpy()
        base_indices_list = base_indices.tolist()

        if seq_length is None:
            seq_length = batch_np.shape[1]
            channel_count = batch_np.shape[2]

        x_vals = batch_np[:, :, 0]
        y_vals = batch_np[:, :, 1]

        batch_timesteps = x_vals.size
        total_timesteps += batch_timesteps
        total_sequences += batch_np.shape[0]

        x_sum += float(x_vals.sum())
        x_sumsq += float(np.square(x_vals).sum())
        x_min = min(x_min, float(x_vals.min()))
        x_max = max(x_max, float(x_vals.max()))

        y_sum += float(y_vals.sum())
        y_sumsq += float(np.square(y_vals).sum())
        y_min = min(y_min, float(y_vals.min()))
        y_max = max(y_max, float(y_vals.max()))

        zero_mask_x = np.abs(x_vals) <= args.zero_eps
        zero_mask_y = np.abs(y_vals) <= args.zero_eps
        x_zero_count += int(zero_mask_x.sum())
        y_zero_count += int(zero_mask_y.sum())
        xy_zero_count += int(np.logical_and(zero_mask_x, zero_mask_y).sum())

        x_outside_low += int((x_vals < hist_range[0]).sum())
        x_outside_high += int((x_vals > hist_range[1]).sum())
        y_outside_low += int((y_vals < hist_range[0]).sum())
        y_outside_high += int((y_vals > hist_range[1]).sum())

        x_hist += np.histogram(x_vals, bins=args.hist_bins, range=hist_range)[0]
        y_hist += np.histogram(y_vals, bins=args.hist_bins, range=hist_range)[0]

        if channel_count and channel_count > 2:
            press_vals = batch_np[:, :, 2]
            press_total_count += press_vals.size
            press_active_count += int((press_vals > args.active_threshold).sum())

        for local_idx, base_idx in enumerate(base_indices_list):
            file_id, _ = dataset.segment_index[base_idx]
            file_sequence_counts[file_id] += 1
            file_timestep_counts[file_id] += batch_np.shape[1]
            file_zero_counts[file_id] += int(
                np.logical_and(
                    np.abs(batch_np[local_idx, :, 0]) <= args.zero_eps,
                    np.abs(batch_np[local_idx, :, 1]) <= args.zero_eps,
                ).sum()
            )
            if channel_count and channel_count > 2:
                file_active_counts[file_id] += int((batch_np[local_idx, :, 2] > args.active_threshold).sum())

            if args.duplicate_sample_size and hashed_sequences < args.duplicate_sample_size:
                digest = hashlib.sha1(batch_np[local_idx].tobytes()).hexdigest()
                entry = duplicate_info.setdefault(digest, {"count": 0, "examples": []})
                entry["count"] = int(entry["count"]) + 1
                examples: List[int] = entry["examples"]  # type: ignore[assignment]
                if len(examples) < 3:
                    examples.append(base_idx)
                hashed_sequences += 1

    if total_sequences == 0 or total_timesteps == 0:
        raise RuntimeError("No data processed. Check sampling configuration or dataset contents.")

    x_mean = x_sum / total_timesteps
    x_variance = max(x_sumsq / total_timesteps - x_mean ** 2, 0.0)
    x_std = math.sqrt(x_variance)

    y_mean = y_sum / total_timesteps
    y_variance = max(y_sumsq / total_timesteps - y_mean ** 2, 0.0)
    y_std = math.sqrt(y_variance)

    x_zero_pct = x_zero_count / total_timesteps
    y_zero_pct = y_zero_count / total_timesteps
    xy_zero_pct = xy_zero_count / total_timesteps

    press_active_pct = None
    if press_total_count:
        press_active_pct = press_active_count / press_total_count

    duplicate_rows = [
        (hash_key, info)
        for hash_key, info in duplicate_info.items()
        if int(info["count"]) > 1
    ]
    duplicate_rows.sort(key=lambda item: int(item[1]["count"]), reverse=True)
    top_duplicates = duplicate_rows[: args.duplicate_top_k]

    per_file_rows: List[Tuple[str, int, float, Optional[float]]] = []
    for file_id, seq_count in file_sequence_counts.items():
        file_path = dataset.files[file_id].name
        total_steps = file_timestep_counts[file_id]
        zero_ratio = file_zero_counts[file_id] / total_steps if total_steps else 0.0
        active_ratio = None
        if press_total_count:
            active_ratio = file_active_counts[file_id] / total_steps if total_steps else 0.0
        per_file_rows.append((file_path, seq_count, zero_ratio, active_ratio))

    per_file_rows.sort(key=lambda item: item[1], reverse=True)
    per_file_rows = per_file_rows[: args.file_table_limit]

    summary_payload = {
        "total_sequences": total_sequences,
        "total_timesteps": total_timesteps,
        "sequence_length": seq_length,
        "channels": channel_count,
        "x_mean": x_mean,
        "x_std": x_std,
        "x_min": x_min,
        "x_max": x_max,
        "x_zero_pct": x_zero_pct,
        "y_mean": y_mean,
        "y_std": y_std,
        "y_min": y_min,
        "y_max": y_max,
        "y_zero_pct": y_zero_pct,
        "xy_zero_pct": xy_zero_pct,
        "x_outside_low": x_outside_low,
        "x_outside_high": x_outside_high,
        "y_outside_low": y_outside_low,
        "y_outside_high": y_outside_high,
        "hashed_sequences": hashed_sequences,
        "duplicate_hashes": len(duplicate_info),
        "duplicate_collisions": sum(int(info["count"]) - 1 for info in duplicate_info.values() if int(info["count"]) > 1),
        "duplicate_sample_unique": sum(1 for info in duplicate_info.values() if int(info["count"]) == 1),
    }
    if press_active_pct is not None:
        summary_payload["press_active_pct"] = press_active_pct

    print("QA summary")
    for key in [
        "total_sequences",
        "total_timesteps",
        "sequence_length",
        "channels",
        "x_mean",
        "x_std",
        "x_min",
        "x_max",
        "x_zero_pct",
        "y_mean",
        "y_std",
        "y_min",
        "y_max",
        "y_zero_pct",
        "xy_zero_pct",
        "press_active_pct",
    ]:
        if key in summary_payload:
            print(f"  {key}: {summary_payload[key]}")

    if wandb_run:
        wandb_log = dict(summary_payload)
        wandb_log["x_histogram"] = wandb.Histogram(np_histogram=(x_hist.astype(np.float64), hist_edges))
        wandb_log["y_histogram"] = wandb.Histogram(np_histogram=(y_hist.astype(np.float64), hist_edges))

        if per_file_rows:
            file_table = wandb.Table(columns=["file", "sequences", "xy_zero_ratio", "press_active_ratio"])
            for file_name, seq_count, zero_ratio, active_ratio in per_file_rows:
                file_table.add_data(
                    file_name,
                    seq_count,
                    zero_ratio,
                    float(active_ratio) if active_ratio is not None else float("nan"),
                )
            wandb_log["per_file_summary"] = file_table

        if top_duplicates:
            dup_table = wandb.Table(columns=["hash", "count", "examples"])
            for hash_key, info in top_duplicates:
                example_strings = [describe_segment(dataset, idx) for idx in info["examples"]]  # type: ignore[index]
                dup_table.add_data(hash_key, int(info["count"]), "; ".join(example_strings))
            wandb_log["duplicate_examples"] = dup_table

        wandb.log(wandb_log)
        wandb_run.summary.update(summary_payload)
        wandb.finish()
    else:
        if per_file_rows:
            print("\nTop files by sequences scanned:")
            for file_name, seq_count, zero_ratio, active_ratio in per_file_rows:
                active_str = f", active_ratio={active_ratio:.3f}" if active_ratio is not None else ""
                print(f"  {file_name}: sequences={seq_count}, zero_ratio={zero_ratio:.3f}{active_str}")
        if top_duplicates:
            print("\nPotential duplicates (sample-limited):")
            for hash_key, info in top_duplicates:
                example_strings = [describe_segment(dataset, idx) for idx in info["examples"]]  # type: ignore[index]
                print(f"  {hash_key}: count={info['count']} -> {example_strings}")

    close_dataset_handles(dataset)


if __name__ == "__main__":
    main()
