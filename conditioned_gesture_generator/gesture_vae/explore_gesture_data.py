import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import h5py
import plotly.graph_objects as go
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from conditioned_gesture_generator.train_cnn_gesture_classifier_no_int import H5ActionSequenceDataset


class XYOnlyGestureDataset(Dataset):
    """Expose only the first two gesture channels (x, y)."""

    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset
        self.sequence_length = getattr(base_dataset, "sequence_length", None)
        self.num_channels = getattr(base_dataset, "num_channels", None)

        if self.sequence_length is None or self.num_channels is None:
            sample = base_dataset[0]
            self.sequence_length = sample.shape[-2]
            self.num_channels = sample.shape[-1]

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sequence = self.base_dataset[idx]
        if sequence.shape[-1] > 2:
            sequence = sequence[..., :2]
        return sequence


def _resolve_h5_files(processed_data_dir: Path, file_whitelist: Optional[List[str]]) -> List[Path]:
    if file_whitelist:
        candidates = [processed_data_dir / fname for fname in file_whitelist]
    else:
        candidates = sorted(processed_data_dir.glob('*.h5'))

    existing = []
    missing = []
    for candidate in candidates:
        if candidate.exists():
            existing.append(candidate)
        else:
            missing.append(candidate)

    if missing:
        print(f"Warning: Skipping missing files: {[p.name for p in missing]}")

    if not existing:
        raise FileNotFoundError(f"No HDF5 files found in {processed_data_dir}")

    return existing


def _dataset_has_embeddings(processed_data_dir: Path, file_whitelist: Optional[List[str]]) -> bool:
    files = _resolve_h5_files(processed_data_dir, file_whitelist)
    for file_path in files:
        with h5py.File(file_path, 'r') as f:
            if 'embeddings' in f:
                return True
            if 'actions' in f and 'embeddings' not in f:
                return False
    return False


class H5ActionsOnlySequenceDataset(Dataset):
    """Load individual action segments from HDF5 files lacking embeddings."""

    def __init__(self, processed_data_dir: Path, file_whitelist: Optional[List[str]] = None):
        self.data_dir = processed_data_dir
        self.h5_files = _resolve_h5_files(processed_data_dir, file_whitelist)
        self._file_handles: Dict[int, Dict[str, h5py.File]] = {}
        self.segment_index: List[Tuple[str, Optional[int], int]] = []
        self.sequence_length: Optional[int] = None
        self.num_channels: Optional[int] = None

        for file_path in self.h5_files:
            with h5py.File(file_path, 'r') as f:
                if 'actions' not in f:
                    print(f"Warning: HDF5 file {file_path} missing 'actions' dataset; skipping")
                    continue
                actions_ds = f['actions']
                if actions_ds.ndim == 4:
                    num_traj, num_segments, seq_len, channels = actions_ds.shape
                    if self.sequence_length is None:
                        self.sequence_length = seq_len
                        self.num_channels = channels
                    elif seq_len != self.sequence_length or channels != self.num_channels:
                        raise ValueError(
                            "Inconsistent action shape detected across files: "
                            f"expected ({self.sequence_length}, {self.num_channels}) "
                            f"but found ({seq_len}, {channels}) in {file_path}"
                        )
                    for traj_idx in range(num_traj):
                        for seg_idx in range(num_segments):
                            self.segment_index.append((str(file_path), traj_idx, seg_idx))
                elif actions_ds.ndim == 3:
                    num_segments, seq_len, channels = actions_ds.shape
                    if self.sequence_length is None:
                        self.sequence_length = seq_len
                        self.num_channels = channels
                    elif seq_len != self.sequence_length or channels != self.num_channels:
                        raise ValueError(
                            "Inconsistent action shape detected across files: "
                            f"expected ({self.sequence_length}, {self.num_channels}) "
                            f"but found ({seq_len}, {channels}) in {file_path}"
                        )
                    for seg_idx in range(num_segments):
                        self.segment_index.append((str(file_path), None, seg_idx))
                else:
                    raise ValueError(
                        f"Expected 3D or 4D 'actions' dataset in {file_path}, found shape {actions_ds.shape}"
                    )

        if not self.segment_index:
            raise ValueError("No action segments found in the provided HDF5 data")

    def __len__(self) -> int:
        return len(self.segment_index)

    def _get_file_handle(self, file_path: str) -> h5py.File:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        if worker_id not in self._file_handles:
            self._file_handles[worker_id] = {}
        if file_path not in self._file_handles[worker_id]:
            self._file_handles[worker_id][file_path] = h5py.File(file_path, 'r')
        return self._file_handles[worker_id][file_path]

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_path, traj_idx, seg_idx = self.segment_index[idx]
        h5_file = self._get_file_handle(file_path)
        actions_ds = h5_file['actions']
        if traj_idx is None:
            action_np = actions_ds[seg_idx]
        else:
            action_np = actions_ds[traj_idx, seg_idx]
        return torch.from_numpy(np.array(action_np, copy=True)).float()


def load_manifest(path: Path) -> Dict[str, List[str]]:
    with path.open("r") as f:
        manifest = json.load(f)
    if "splits" not in manifest:
        raise KeyError(f"Manifest at {path} missing required 'splits' key")
    return manifest["splits"]


def compute_value_bins(num_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float64)


def compute_time_bins(sequence_length: int) -> np.ndarray:
    return np.arange(sequence_length + 1, dtype=np.float64)


def analyze_split(
    dataset: Dataset,
    split_name: str,
    num_bins: int,
    batch_size: int,
    num_workers: int,
    max_batches: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    sequence_length = getattr(dataset, "sequence_length", None)
    if sequence_length is None:
        raise ValueError("Dataset must expose sequence_length metadata")

    value_edges = compute_value_bins(num_bins)
    time_edges = compute_time_bins(sequence_length)
    bin_centers = (value_edges[:-1] + value_edges[1:]) / 2.0

    value_counts_x = np.zeros(num_bins, dtype=np.int64)
    value_counts_y = np.zeros(num_bins, dtype=np.int64)
    time_value_counts_x = np.zeros((sequence_length, num_bins), dtype=np.int64)
    time_value_counts_y = np.zeros((sequence_length, num_bins), dtype=np.int64)

    total_count = 0
    total_sequences = 0
    all_zero_sequences = 0
    x_zero_sequences = 0
    y_zero_sequences = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_sq_x = 0.0
    sum_sq_y = 0.0
    time_sum_x = np.zeros(sequence_length, dtype=np.float64)
    time_sum_y = np.zeros(sequence_length, dtype=np.float64)

    progress = tqdm(loader, desc=f"Processing split '{split_name}'")

    for batch_idx, batch in enumerate(progress):
        batch_np = batch.cpu().numpy()
        if batch_np.shape[-1] < 2:
            raise ValueError("Expected at least two gesture channels (x, y)")

        x_vals = batch_np[..., 0]
        y_vals = batch_np[..., 1]

        flat_x = x_vals.reshape(-1)
        flat_y = y_vals.reshape(-1)

        hist_x, _ = np.histogram(flat_x, bins=value_edges)
        hist_y, _ = np.histogram(flat_y, bins=value_edges)
        value_counts_x += hist_x
        value_counts_y += hist_y

        batch_size_actual = x_vals.shape[0]
        total_sequences += batch_size_actual
        time_indices = np.tile(np.arange(sequence_length, dtype=np.float64), (batch_size_actual, 1))

        hist2d_x, _, _ = np.histogram2d(
            time_indices.reshape(-1),
            flat_x,
            bins=[time_edges, value_edges],
        )
        hist2d_y, _, _ = np.histogram2d(
            time_indices.reshape(-1),
            flat_y,
            bins=[time_edges, value_edges],
        )
        time_value_counts_x += hist2d_x.astype(np.int64)
        time_value_counts_y += hist2d_y.astype(np.int64)

        total_count += flat_x.size
        sum_x += flat_x.sum()
        sum_y += flat_y.sum()
        sum_sq_x += np.square(flat_x).sum()
        sum_sq_y += np.square(flat_y).sum()
        time_sum_x += x_vals.sum(axis=0)
        time_sum_y += y_vals.sum(axis=0)

        seq_all_zero = np.all((x_vals == 0) & (y_vals == 0), axis=1)
        seq_x_zero = np.all(x_vals == 0, axis=1)
        seq_y_zero = np.all(y_vals == 0, axis=1)

        all_zero_sequences += int(seq_all_zero.sum())
        x_zero_sequences += int(seq_x_zero.sum())
        y_zero_sequences += int(seq_y_zero.sum())

        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break

    if total_count == 0 or total_sequences == 0:
        raise RuntimeError(f"Split '{split_name}' produced no samples")

    per_timestep_count = total_count / sequence_length

    return {
        "value_edges": value_edges,
        "bin_centers": bin_centers,
        "value_counts_x": value_counts_x,
        "value_counts_y": value_counts_y,
        "time_edges": time_edges,
        "time_value_counts_x": time_value_counts_x,
        "time_value_counts_y": time_value_counts_y,
        "mean_x": sum_x / total_count,
        "mean_y": sum_y / total_count,
        "std_x": np.sqrt(sum_sq_x / total_count - (sum_x / total_count) ** 2),
        "std_y": np.sqrt(sum_sq_y / total_count - (sum_y / total_count) ** 2),
        "time_mean_x": time_sum_x / per_timestep_count,
        "time_mean_y": time_sum_y / per_timestep_count,
        "samples_processed": total_count,
        "sequences_processed": total_sequences,
        "sequence_all_zero_count": all_zero_sequences,
        "sequence_x_zero_count": x_zero_sequences,
        "sequence_y_zero_count": y_zero_sequences,
        "sequence_all_zero_fraction": all_zero_sequences / total_sequences,
        "sequence_x_zero_fraction": x_zero_sequences / total_sequences,
        "sequence_y_zero_fraction": y_zero_sequences / total_sequences,
    }


def _prepare_counts_for_plot(counts: np.ndarray, log_scale: bool) -> np.ndarray:
    if not log_scale:
        return counts
    counts_float = counts.astype(np.float64)
    counts_float[counts_float <= 0] = np.nan
    return counts_float


def _drop_zero_bin(bin_centers: np.ndarray, counts: np.ndarray, value_edges: np.ndarray) -> np.ndarray:
    if bin_centers.size == 0:
        return bin_centers, counts
    zero_mask = ~np.isclose(value_edges[:-1], 0.0)
    if zero_mask.all():
        return bin_centers, counts
    if not zero_mask.any():
        return np.array([]), np.array([])
    return bin_centers[zero_mask], counts[zero_mask]


def _remove_zero_bin_from_heatmap(value_edges: np.ndarray, counts: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
    if counts.size == 0:
        return None
    zero_mask = ~np.isclose(value_edges[:-1], 0.0)
    if zero_mask.all():
        return {
            "value_edges": value_edges,
            "counts": counts,
        }
    if not zero_mask.any():
        return None
    kept_indices = np.where(zero_mask)[0]
    trimmed_counts = counts[:, kept_indices]
    edge_indices = np.unique(np.concatenate([kept_indices, kept_indices + 1]))
    trimmed_edges = value_edges[edge_indices]
    if trimmed_counts.size == 0 or trimmed_edges.size <= 1:
        return None
    return {
        "value_edges": trimmed_edges,
        "counts": trimmed_counts,
    }


def create_value_distribution_figure(
    split_name: str,
    bin_centers: np.ndarray,
    counts_x: np.ndarray,
    counts_y: np.ndarray,
    *,
    log_scale: bool,
    suffix: str = "",
) -> go.Figure:
    plot_counts_x = _prepare_counts_for_plot(counts_x, log_scale)
    plot_counts_y = _prepare_counts_for_plot(counts_y, log_scale)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=plot_counts_x,
            mode="lines",
            name="x value frequency",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=plot_counts_y,
            mode="lines",
            name="y value frequency",
        )
    )
    title_suffix = f" ({suffix})" if suffix else ""
    fig.update_layout(
        title=f"Value Frequency Distribution — {split_name}{title_suffix}",
        xaxis_title="Value (normalized)",
        yaxis_title="Count",
        hovermode="x unified",
    )
    if log_scale:
        fig.update_yaxes(type="log")
    return fig


def create_time_value_heatmap(
    split_name: str,
    axis_name: str,
    time_edges: np.ndarray,
    value_edges: np.ndarray,
    counts: np.ndarray,
    *,
    suffix: str = "",
) -> go.Figure:
    time_centers = (time_edges[:-1] + time_edges[1:]) / 2.0
    value_centers = (value_edges[:-1] + value_edges[1:]) / 2.0

    fig = go.Figure(
        data=go.Heatmap(
            z=counts,
            x=value_centers,
            y=time_centers,
            colorbar=dict(title="Count"),
        )
    )
    title_suffix = f" ({suffix})" if suffix else ""
    fig.update_layout(
        title=f"Time vs {axis_name.upper()} Value Distribution — {split_name}{title_suffix}",
        xaxis_title=f"{axis_name.upper()} value (normalized)",
        yaxis_title="Timestep",
    )
    return fig


def log_to_wandb(
    split_name: str,
    value_fig: go.Figure,
    heatmap_x: go.Figure,
    heatmap_y: go.Figure,
    stats: Dict[str, np.ndarray],
    *,
    value_fig_no_zero: Optional[go.Figure] = None,
    heatmap_y_no_zero: Optional[go.Figure] = None,
    step: Optional[int] = None,
) -> None:
    import wandb

    payload = {
        f"{split_name}/value_distribution": wandb.Plotly(value_fig),
        f"{split_name}/time_vs_x": wandb.Plotly(heatmap_x),
        f"{split_name}/time_vs_y": wandb.Plotly(heatmap_y),
        f"{split_name}/mean_x": float(stats["mean_x"]),
        f"{split_name}/mean_y": float(stats["mean_y"]),
        f"{split_name}/std_x": float(stats["std_x"]),
        f"{split_name}/std_y": float(stats["std_y"]),
        f"{split_name}/sequences_processed": int(stats["sequences_processed"]),
        f"{split_name}/seq_all_zero_frac": float(stats["sequence_all_zero_fraction"]),
        f"{split_name}/seq_x_zero_frac": float(stats["sequence_x_zero_fraction"]),
        f"{split_name}/seq_y_zero_frac": float(stats["sequence_y_zero_fraction"]),
        f"{split_name}/seq_all_zero_count": int(stats["sequence_all_zero_count"]),
        f"{split_name}/seq_x_zero_count": int(stats["sequence_x_zero_count"]),
        f"{split_name}/seq_y_zero_count": int(stats["sequence_y_zero_count"]),
    }
    if value_fig_no_zero is not None:
        payload[f"{split_name}/value_distribution_no_zero"] = wandb.Plotly(value_fig_no_zero)
    if heatmap_y_no_zero is not None:
        payload[f"{split_name}/time_vs_y_no_zero"] = wandb.Plotly(heatmap_y_no_zero)
    if "time_mean_x" in stats and "time_mean_y" in stats:
        payload[f"{split_name}/time_mean_x"] = stats["time_mean_x"].tolist()
        payload[f"{split_name}/time_mean_y"] = stats["time_mean_y"].tolist()
    wandb.log(payload, step=step)


def save_figures(
    output_dir: Path,
    split_name: str,
    value_fig: go.Figure,
    heatmap_x: go.Figure,
    heatmap_y: go.Figure,
    *,
    value_fig_no_zero: Optional[go.Figure] = None,
    heatmap_y_no_zero: Optional[go.Figure] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    value_fig.write_html(output_dir / f"{split_name}_value_distribution.html")
    heatmap_x.write_html(output_dir / f"{split_name}_time_vs_x.html")
    heatmap_y.write_html(output_dir / f"{split_name}_time_vs_y.html")
    if value_fig_no_zero is not None:
        value_fig_no_zero.write_html(output_dir / f"{split_name}_value_distribution_no_zero.html")
    if heatmap_y_no_zero is not None:
        heatmap_y_no_zero.write_html(output_dir / f"{split_name}_time_vs_y_no_zero.html")


def build_dataset(
    processed_data_dir: Path,
    manifest_path: Optional[Path],
    split_name: Optional[str],
    file_whitelist: Optional[List[str]] = None,
) -> XYOnlyGestureDataset:
    has_embeddings = _dataset_has_embeddings(processed_data_dir, file_whitelist)
    if has_embeddings:
        dataset = H5ActionSequenceDataset(
            processed_data_dir=str(processed_data_dir),
            manifest_path=str(manifest_path) if manifest_path else None,
            split=split_name,
        )
    else:
        dataset = H5ActionsOnlySequenceDataset(
            processed_data_dir=processed_data_dir,
            file_whitelist=file_whitelist,
        )
    return XYOnlyGestureDataset(dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore gesture value distributions")
    parser.add_argument("--processed_data_dir", type=str, required=True, help="Directory with processed HDF5 trajectories")
    parser.add_argument("--manifest", type=str, default=None, help="Manifest JSON describing dataset splits")
    parser.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of manifest split names to analyze (defaults to all splits)",
    )
    parser.add_argument("--bins", type=int, default=6000, help="Number of bins for value histograms (per dimension)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for data loading")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers")
    parser.add_argument("--run_name", type=str, default=None, help="Optional wandb run name")
    parser.add_argument("--entity", type=str, default=None, help="Optional wandb entity (team) name")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional directory to export figures as HTML")
    parser.add_argument("--max_batches", type=int, default=None, help="Optional limit on processed batches (for debugging)")
    parser.add_argument("--log_scale", dest="log_scale", action="store_true", help="Plot frequency counts on a log scale")
    parser.add_argument("--no_log_scale", dest="log_scale", action="store_false", help="Disable log scaling for frequency plots")
    parser.set_defaults(log_scale=True)
    return parser.parse_args()


def main():
    args = parse_args()

    processed_data_dir = Path(args.processed_data_dir)
    if not processed_data_dir.exists():
        raise FileNotFoundError(f"Processed data directory {processed_data_dir} does not exist")

    manifest_path = Path(args.manifest) if args.manifest else None
    if manifest_path and not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file {manifest_path} does not exist")

    if args.bins <= 0:
        raise ValueError("--bins must be a positive integer")

    if args.disable_wandb:
        wandb_run = None
    else:
        import wandb

        wandb_run = wandb.init(
            project="gesture_exploration",
            name=args.run_name,
            entity=args.entity,
            config={
                "processed_data_dir": str(processed_data_dir),
                "manifest": str(manifest_path) if manifest_path else None,
                "bins": args.bins,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "max_batches": args.max_batches,
                "log_scale": args.log_scale,
            },
        )

    output_dir = Path(args.output_dir) if args.output_dir else None

    splits_dict: Dict[str, List[str]] = {}
    if manifest_path:
        splits_dict = load_manifest(manifest_path)
        if args.splits:
            missing = [s for s in args.splits if s not in splits_dict]
            if missing:
                raise ValueError(f"Requested splits not found in manifest: {missing}")
            selected_splits = args.splits
        else:
            selected_splits = list(splits_dict.keys())
    else:
        if args.splits:
            raise ValueError("--splits provided without a manifest")
        selected_splits = ["all"]

    summaries: Dict[str, Dict[str, np.ndarray]] = {}

    for split_name in selected_splits:
        file_whitelist = splits_dict[split_name] if manifest_path else None
        dataset = build_dataset(
            processed_data_dir,
            manifest_path,
            split_name if manifest_path else None,
            file_whitelist,
        )

        analysis = analyze_split(
            dataset=dataset,
            split_name=split_name,
            num_bins=args.bins,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_batches=args.max_batches,
        )

        value_fig = create_value_distribution_figure(
            split_name=split_name,
            bin_centers=analysis["bin_centers"],
            counts_x=analysis["value_counts_x"],
            counts_y=analysis["value_counts_y"],
            log_scale=args.log_scale,
        )

        bin_centers_no_zero, counts_x_no_zero = _drop_zero_bin(
            analysis["bin_centers"], analysis["value_counts_x"], analysis["value_edges"]
        )
        _, counts_y_no_zero = _drop_zero_bin(
            analysis["bin_centers"], analysis["value_counts_y"], analysis["value_edges"]
        )
        if bin_centers_no_zero.size > 0:
            value_fig_no_zero = create_value_distribution_figure(
                split_name=split_name,
                bin_centers=bin_centers_no_zero,
                counts_x=counts_x_no_zero,
                counts_y=counts_y_no_zero,
                log_scale=args.log_scale,
                suffix="no zero bin",
            )
        else:
            value_fig_no_zero = go.Figure()
            value_fig_no_zero.update_layout(
                title=f"Value Frequency Distribution — {split_name} (no zero bin)",
                xaxis_title="Value (normalized)",
                yaxis_title="Count",
                annotations=[
                    dict(
                        text="All samples fall in the zero bin",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                    )
                ],
            )
            if args.log_scale:
                value_fig_no_zero.update_yaxes(type="log")

        heatmap_x = create_time_value_heatmap(
            split_name=split_name,
            axis_name="x",
            time_edges=analysis["time_edges"],
            value_edges=analysis["value_edges"],
            counts=analysis["time_value_counts_x"],
        )
        heatmap_y = create_time_value_heatmap(
            split_name=split_name,
            axis_name="y",
            time_edges=analysis["time_edges"],
            value_edges=analysis["value_edges"],
            counts=analysis["time_value_counts_y"],
        )

        heatmap_y_no_zero = None
        trimmed_heatmap = _remove_zero_bin_from_heatmap(analysis["value_edges"], analysis["time_value_counts_y"])
        if trimmed_heatmap is not None:
            heatmap_y_no_zero = create_time_value_heatmap(
                split_name=split_name,
                axis_name="y",
                time_edges=analysis["time_edges"],
                value_edges=trimmed_heatmap["value_edges"],
                counts=trimmed_heatmap["counts"],
                suffix="no zero bin",
            )
        else:
            placeholder = go.Figure()
            placeholder.update_layout(
                title=f"Time vs Y Value Distribution — {split_name} (no zero bin)",
                xaxis_title="Y value (normalized)",
                yaxis_title="Timestep",
                annotations=[
                    dict(
                        text="Unable to create zero-excluded heatmap",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                    )
                ],
            )
            heatmap_y_no_zero = placeholder

        print(f"Split '{split_name}':")
        print(f"  Samples processed: {analysis['samples_processed']:,}")
        print(f"  Mean (x, y): ({analysis['mean_x']:.4f}, {analysis['mean_y']:.4f})")
        print(f"  Std  (x, y): ({analysis['std_x']:.4f}, {analysis['std_y']:.4f})")
        print(
            "  Gestures: total {total:,} | all-zero {all_zero:,} ({all_pct:.2f}%) | x-zero {x_zero:,} ({x_pct:.2f}%) | y-zero {y_zero:,} ({y_pct:.2f}%)"
            .format(
                total=analysis['sequences_processed'],
                all_zero=analysis['sequence_all_zero_count'],
                all_pct=analysis['sequence_all_zero_fraction'] * 100.0,
                x_zero=analysis['sequence_x_zero_count'],
                x_pct=analysis['sequence_x_zero_fraction'] * 100.0,
                y_zero=analysis['sequence_y_zero_count'],
                y_pct=analysis['sequence_y_zero_fraction'] * 100.0,
            )
        )

        if output_dir is not None:
            save_figures(
                output_dir,
                split_name,
                value_fig,
                heatmap_x,
                heatmap_y,
                value_fig_no_zero=value_fig_no_zero,
                heatmap_y_no_zero=heatmap_y_no_zero,
            )

        if not args.disable_wandb and wandb_run is not None:
            log_to_wandb(
                split_name=split_name,
                value_fig=value_fig,
                heatmap_x=heatmap_x,
                heatmap_y=heatmap_y,
                stats=analysis,
                value_fig_no_zero=value_fig_no_zero,
                heatmap_y_no_zero=heatmap_y_no_zero,
            )

        summaries[split_name] = analysis

    if wandb_run is not None:
        wandb_run.finish()

    return summaries


if __name__ == "__main__":
    main()
