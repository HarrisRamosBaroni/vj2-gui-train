"""Population-style gesture data augmentation for HDF5 gesture-only datasets."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable
    class _DummyTqdm:
        def __init__(self, iterable=None, total=None, desc=None, unit=None):
            self._iterable = iterable if iterable is not None else range(total or 0)

        def __iter__(self):
            return iter(self._iterable)

        def update(self, n: int = 1) -> None:  # pragma: no cover - no-op
            return None

        def close(self) -> None:  # pragma: no cover - no-op
            return None

    def tqdm(iterable=None, total=None, desc=None, unit=None):  # type: ignore
        return _DummyTqdm(iterable=iterable, total=total, desc=desc, unit=unit)


AugmentationFn = Callable[[torch.Tensor, np.random.Generator], torch.Tensor]


class NoActiveSegments(RuntimeError):
    """Raised when a source file has no non-zero gesture segments."""


# ---------------------------------------------------------------------------
# Augmentation primitives
# ---------------------------------------------------------------------------


def temporal_shift(batch: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Shift sequences forward/backward with clipped magnitude to retain signal."""
    seq_len = batch.shape[1]
    if seq_len == 0:
        return batch.clone()

    max_shift = max(1, seq_len // 8)  # keep at most ~12.5% padding
    shifts = rng.integers(-max_shift, max_shift + 1, size=batch.shape[0])

    shifted = batch.clone()
    for i, shift in enumerate(shifts):
        if shift == 0:
            continue
        if shift > 0:
            shifted[i, shift:, :] = batch[i, :-shift, :]
            shifted[i, :shift, :] = 0.0
        else:
            shift_abs = -shift
            shifted[i, :-shift_abs, :] = batch[i, shift_abs:, :]
            shifted[i, -shift_abs:, :] = 0.0
    return shifted


def axis_offsets(batch: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Add independent XY offsets sampled from a moderated range."""
    offsets = torch.from_numpy(
        rng.uniform(-0.3, 0.3, size=(batch.shape[0], 1, 2))
    ).to(device=batch.device, dtype=batch.dtype)
    return batch + offsets


def tail_mask(batch: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Trim part of the active tail while ensuring gestures remain non-empty."""
    masked = batch.clone()
    seq_len = batch.shape[1]
    if seq_len == 0:
        return masked

    max_mask = max(1, seq_len // 3)  # trim at most one third
    eps = 1e-8
    for i in range(batch.shape[0]):
        seq = batch[i]
        active = torch.any(torch.abs(seq[:, :2]) > eps, dim=1)
        if not torch.any(active):  # already filtered elsewhere, but guard anyway
            continue
        last_active_idx = torch.nonzero(active, as_tuple=False)[-1].item()
        available = seq_len - last_active_idx - 1
        if available <= 0:
            continue
        mask_len = int(rng.integers(0, min(max_mask, available) + 1))
        if mask_len > 0:
            masked[i, last_active_idx + 1:last_active_idx + 1 + mask_len, :] = 0.0
    return masked


def reverse_sequence(batch: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Reverse temporal order of gestures."""
    del rng  # signature compatibility
    return torch.flip(batch, dims=(1,))


AUGMENTATIONS: Dict[str, AugmentationFn] = {
    "temporal_shift": temporal_shift,
    "axis_offsets": axis_offsets,
    "tail_mask": tail_mask,
    "reverse": reverse_sequence,
}


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _sanitize_attribute_value(value):
    """Convert attribute values into h5py-friendly scalars/arrays."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "U":
            return value.astype("S")
        if value.dtype.kind == "O":
            return np.array([_sanitize_attribute_value(v) for v in value], dtype="S")
        return value
    if isinstance(value, np.generic):
        if value.dtype.kind == "U":
            return np.bytes_(value)
        return value.item()
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, str) for v in value):
            return np.array(value, dtype="S")
        return np.array([_sanitize_attribute_value(v) for v in value])
    if isinstance(value, str):
        return value
    return value


def _assign_attrs(attr_obj, attrs: Dict) -> None:
    for key, value in attrs.items():
        try:
            attr_obj[key] = _sanitize_attribute_value(value)
        except TypeError:
            attr_obj[key] = str(value)


def _flatten_actions(actions: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Convert actions dataset to [segments, T, C]."""
    if actions.ndim == 4:
        segments, seq_len, channels = actions.shape[0] * actions.shape[1], actions.shape[2], actions.shape[3]
        actions_flat = actions.reshape(segments, seq_len, channels)
    elif actions.ndim == 3:
        segments, seq_len, channels = actions.shape
        actions_flat = actions
    else:
        raise ValueError(f"Unsupported 'actions' shape {actions.shape}; expected 3D or 4D")
    return actions_flat, seq_len, channels


def _normalize_chunks(shape: Tuple[int, ...], chunks) -> Optional[Tuple[int, ...]]:
    """Adapt original chunk configuration to the current dataset shape."""
    if chunks is None:
        return None

    chunk_list = list(chunks)
    ndim = len(shape)

    while len(chunk_list) > ndim:
        # Merge leading dimensions until ranks match (handles grids like (1,7,250,3)).
        chunk_list[1] = max(1, chunk_list[0] * chunk_list[1])
        chunk_list.pop(0)

    if len(chunk_list) != ndim:
        return None

    normalized = tuple(min(max(1, int(c)), s) for c, s in zip(chunk_list, shape))
    return normalized


def read_actions_payload(path: Path) -> Dict:
    """Load actions and metadata, returning only non-zero gesture segments."""
    with h5py.File(path, "r") as src:
        if "actions" not in src:
            raise KeyError(f"File {path} missing required 'actions' dataset")
        actions_ds = src["actions"]
        raw_actions = np.array(actions_ds[...], dtype=np.float32)
        segments_flat, seq_len, channels = _flatten_actions(raw_actions)

        xy = segments_flat[:, :, :2]
        active_mask = ~np.all(xy == 0.0, axis=(1, 2))
        filtered_actions = segments_flat[active_mask]
        removed_segments = int(active_mask.size - active_mask.sum())

        payload = {
            "segments": filtered_actions,
            "removed_segments": removed_segments,
            "total_segments": int(active_mask.size),
            "sequence_length": seq_len,
            "num_channels": channels,
            "original_dtype": actions_ds.dtype,
            "dataset_attrs": {k: v for k, v in actions_ds.attrs.items()},
            "file_attrs": {k: v for k, v in src.attrs.items()},
            "compression": actions_ds.compression,
            "compression_opts": actions_ds.compression_opts,
            "shuffle": actions_ds.shuffle,
            "fletcher32": actions_ds.fletcher32,
            "chunks": actions_ds.chunks,
        }
    return payload


def write_actions_file(
    path: Path,
    segments: np.ndarray,
    original_dtype,
    *,
    file_attrs: Optional[Dict] = None,
    dataset_attrs: Optional[Dict] = None,
    compression=None,
    compression_opts=None,
    shuffle=None,
    fletcher32=None,
    chunks=None,
) -> None:
    """Persist gesture segments as a 3D [N, T, C] dataset."""
    ensure_parent(path)
    create_kwargs = {}
    if compression is not None:
        create_kwargs["compression"] = compression
        if compression_opts is not None:
            create_kwargs["compression_opts"] = compression_opts
    if chunks is not None:
        normalized_chunks = _normalize_chunks(segments.shape, chunks)
        if normalized_chunks is not None:
            create_kwargs["chunks"] = normalized_chunks
    if shuffle is not None:
        create_kwargs["shuffle"] = shuffle
    if fletcher32 is not None:
        create_kwargs["fletcher32"] = fletcher32

    with h5py.File(path, "w") as dst:
        ds = dst.create_dataset(
            "actions",
            data=segments.astype(original_dtype, copy=False),
            **create_kwargs,
        )
        if dataset_attrs:
            _assign_attrs(ds.attrs, dataset_attrs)
        if file_attrs:
            _assign_attrs(dst.attrs, file_attrs)


def copy_actions_only(source_path: Path, destination_path: Path) -> Tuple[int, int]:
    """Create an actions-only copy, returning (kept_segments, removed_segments)."""
    payload = read_actions_payload(source_path)
    segments = payload["segments"]

    if segments.size == 0:
        raise NoActiveSegments(f"{source_path} contains no active gesture segments")

    file_attrs = dict(payload["file_attrs"])
    file_attrs.update(
        {
            "source_file": source_path.name,
            "augmentation": "original",
            "contains_embeddings": False,
            "total_segments": payload["total_segments"],
            "removed_segments": payload["removed_segments"],
        }
    )

    dataset_attrs = dict(payload["dataset_attrs"])
    dataset_attrs.update(
        {
            "contains_embeddings": False,
            "sequence_length": payload["sequence_length"],
            "num_channels": payload["num_channels"],
            "total_segments": segments.shape[0],
            "removed_segments": payload["removed_segments"],
        }
    )

    write_actions_file(
        destination_path,
        segments,
        payload["original_dtype"],
        file_attrs=file_attrs,
        dataset_attrs=dataset_attrs,
        compression=payload["compression"],
        compression_opts=payload["compression_opts"],
        shuffle=payload["shuffle"],
        fletcher32=payload["fletcher32"],
        chunks=payload["chunks"],
    )

    return segments.shape[0], payload["removed_segments"]


def _ensure_nonzero(original_xy: torch.Tensor, augmented_xy: torch.Tensor) -> torch.Tensor:
    """Fallback to original gesture if augmentation produced an all-zero result."""
    zero_mask = torch.all(torch.abs(augmented_xy) < 1e-8, dim=(1, 2))
    if zero_mask.any():
        augmented_xy[zero_mask] = original_xy[zero_mask]
    return augmented_xy


def apply_augmentation_to_file(
    parent_path: Path,
    destination_path: Path,
    augmentation_name: str,
    augmentation_fn: AugmentationFn,
    rng: np.random.Generator,
) -> None:
    """Apply augmentation to an actions-only file and persist the result."""
    payload = read_actions_payload(parent_path)
    segments = payload["segments"]
    if segments.size == 0:
        raise NoActiveSegments(f"{parent_path} contains no active gesture segments")

    seq_len = payload["sequence_length"]
    channels = payload["num_channels"]
    if channels < 2:
        raise ValueError(f"Augmentation requires at least two action channels, found {channels} in {parent_path}")

    tensor = torch.from_numpy(segments.copy())
    original_xy = tensor[:, :, :2].clone()
    augmented_xy = augmentation_fn(original_xy, rng)
    augmented_xy.clamp_(0.0, 1.0)
    augmented_xy = _ensure_nonzero(original_xy, augmented_xy)
    tensor[:, :, :2] = augmented_xy

    dataset_attrs = dict(payload["dataset_attrs"])
    dataset_attrs.update(
        {
            "augmentation": augmentation_name,
            "contains_embeddings": False,
            "total_segments": tensor.shape[0],
            "removed_segments": 0,
        }
    )

    file_attrs = dict(payload["file_attrs"])
    file_attrs.update(
        {
            "augmentation": augmentation_name,
            "source_file": parent_path.name,
            "contains_embeddings": False,
            "total_segments": tensor.shape[0],
            "removed_segments": 0,
        }
    )

    write_actions_file(
        destination_path,
        tensor.numpy(),
        payload["original_dtype"],
        file_attrs=file_attrs,
        dataset_attrs=dataset_attrs,
        compression=payload["compression"],
        compression_opts=payload["compression_opts"],
        shuffle=payload["shuffle"],
        fletcher32=payload["fletcher32"],
        chunks=payload["chunks"],
    )


# ---------------------------------------------------------------------------
# Manifest + population management
# ---------------------------------------------------------------------------


def generate_augmented_filename(relative_path: Path, counter: int) -> Path:
    new_stem = f"{relative_path.stem}_aug{counter:04d}"
    return relative_path.with_name(new_stem + relative_path.suffix)


def load_manifest(manifest_path: Path) -> Dict:
    with manifest_path.open("r") as f:
        manifest = json.load(f)
    if "splits" not in manifest:
        raise ValueError(f"Manifest at {manifest_path} missing required 'splits' key")
    return manifest


def persist_manifest(manifest: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(manifest, f, indent=2)


def copy_original_files(
    data_dir: Path,
    output_dir: Path,
    manifest: Dict,
    max_workers: Optional[int] = None,
) -> Tuple[Dict[str, Path], Dict[str, List[str]], List[Tuple[str, str]], Dict[str, int]]:
    """Copy every manifest-listed file into the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    split_membership: Dict[str, List[str]] = defaultdict(list)
    for split_name, files in manifest["splits"].items():
        for rel_path in files:
            split_membership[rel_path].append(split_name)

    unique_files = sorted(split_membership.keys())
    copied_paths: Dict[str, Path] = {}
    skipped_files: List[Tuple[str, str]] = []
    stats = {"kept_segments": 0, "removed_segments": 0}

    def _copy_one(rel_path: str) -> Tuple[str, Path, int, int]:
        src_path = data_dir / rel_path
        dst_path = output_dir / rel_path
        ensure_parent(dst_path)
        if not src_path.exists():
            raise FileNotFoundError(f"Referenced file '{rel_path}' not found in {data_dir}")
        kept, removed = copy_actions_only(src_path, dst_path)
        return rel_path, dst_path, kept, removed

    worker_count = max_workers if max_workers and max_workers > 0 else min(32, (os.cpu_count() or 1))

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(_copy_one, rel_path): rel_path for rel_path in unique_files}
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(unique_files),
            desc="Copying originals",
            unit="file",
        ):
            rel_path = futures[future]
            try:
                _, dst_path, kept, removed = future.result()
            except Exception as exc:  # pragma: no cover - handled gracefully at runtime
                skipped_files.append((rel_path, str(exc)))
                split_membership.pop(rel_path, None)
                continue
            copied_paths[rel_path] = dst_path
            stats["kept_segments"] += kept
            stats["removed_segments"] += removed

    return copied_paths, split_membership, skipped_files, stats


def augment_dataset(
    output_dir: Path,
    copied_paths: Dict[str, Path],
    split_membership: Dict[str, List[str]],
    manifest_splits: Dict[str, List[str]],
    num_augmentations: int,
    rng: np.random.Generator,
) -> Tuple[List[str], int]:
    """Perform population-style augmentation and update manifest splits."""
    if num_augmentations <= 0:
        return [], 0

    new_files: List[str] = []
    global_counter = 1
    zero_fallbacks = 0

    total_new = num_augmentations * len(copied_paths)
    progress = tqdm(total=total_new, desc="Augmentations", unit="file") if total_new else None

    for rel_path in copied_paths.keys():
        population = [rel_path]
        for _ in range(num_augmentations):
            parent_rel = rng.choice(population)
            parent_path = output_dir / parent_rel

            aug_name = rng.choice(list(AUGMENTATIONS.keys()))
            aug_fn = AUGMENTATIONS[aug_name]

            new_rel_path = generate_augmented_filename(Path(rel_path), global_counter)
            new_path = output_dir / new_rel_path
            ensure_parent(new_path)

            try:
                apply_augmentation_to_file(parent_path, new_path, aug_name, aug_fn, rng)
            except NoActiveSegments:
                zero_fallbacks += 1
                continue

            new_rel_str = str(new_rel_path)
            population.append(new_rel_str)
            new_files.append(new_rel_str)

            for split_name in split_membership.get(rel_path, []):
                manifest_splits.setdefault(split_name, []).append(new_rel_str)

            global_counter += 1
            if progress is not None:
                progress.update(1)

    if progress is not None:
        progress.close()

    return new_files, zero_fallbacks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate augmented gesture datasets from HDF5 manifests.")
    parser.add_argument("--data-path", required=True, help="Directory containing source HDF5 files referenced by the manifest")
    parser.add_argument("--manifest", required=True, help="Path to the source manifest JSON")
    parser.add_argument("--output-dir", required=True, help="Directory where augmented dataset and manifest will be written")
    parser.add_argument("-N", "--num-augmentations", type=int, default=0, help="Number of augmentation rounds per original file")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    parser.add_argument("--output-manifest", default=None, help="Optional path for the augmented manifest (defaults to output-dir / manifest name)")
    parser.add_argument(
        "--copy-workers",
        type=int,
        default=None,
        help="Number of parallel workers for copying originals (defaults to min(32, CPU count))",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_path)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_manifest_path = Path(args.output_manifest) if args.output_manifest else output_dir / manifest_path.name

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file {manifest_path} does not exist")

    manifest = load_manifest(manifest_path)
    updated_manifest = copy.deepcopy(manifest)
    updated_manifest["splits"] = {k: list(v) for k, v in manifest["splits"].items()}

    rng = np.random.default_rng(args.seed)

    copied_paths, split_membership, skipped_files, copy_stats = copy_original_files(
        data_dir=data_dir,
        output_dir=output_dir,
        manifest=manifest,
        max_workers=args.copy_workers,
    )

    if skipped_files:
        skipped_set = {rel_path for rel_path, _ in skipped_files}
        for split_name, files in updated_manifest["splits"].items():
            updated_manifest["splits"][split_name] = [
                rel_path for rel_path in files if rel_path not in skipped_set
            ]

        print("Skipped files (missing or empty after filtering):")
        for rel_path, reason in skipped_files:
            print(f"  {rel_path}: {reason}")

    new_files, zero_fallbacks = augment_dataset(
        output_dir=output_dir,
        copied_paths=copied_paths,
        split_membership=split_membership,
        manifest_splits=updated_manifest["splits"],
        num_augmentations=max(0, args.num_augmentations),
        rng=rng,
    )

    updated_manifest.setdefault("augmentation_info", {})
    updated_manifest["augmentation_info"].update(
        {
            "num_augmentations_per_file": max(0, args.num_augmentations),
            "total_new_files": len(new_files),
            "seed": args.seed,
            "augmentations": list(AUGMENTATIONS.keys()),
            "contains_embeddings": False,
            "copied_segments": copy_stats["kept_segments"],
            "removed_zero_segments": copy_stats["removed_segments"],
            "augmentation_zero_fallbacks": zero_fallbacks,
        }
    )

    persist_manifest(updated_manifest, output_manifest_path)

    summary_lines = [
        f"Copied {len(copied_paths)} original files to {output_dir}",
        f"Total kept segments: {copy_stats['kept_segments']}",
        f"Removed zero segments: {copy_stats['removed_segments']}",
        f"Generated {len(new_files)} augmented files",
        f"Augmented manifest written to {output_manifest_path}",
    ]
    if skipped_files:
        summary_lines.insert(1, f"Skipped {len(skipped_files)} files (see above)")
    if zero_fallbacks:
        summary_lines.insert(4, f"Skipped {zero_fallbacks} augmentations with empty results")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
