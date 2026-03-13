"""Batch orchestrator for unified manual/auto midline pipeline."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np

from .analysis import run_steady_state_kinematics, save_results, save_analysis_ready_csv
from .naming_config import NamingConfig
from .auto_midline import AutoMidlineResult, extract_auto_midline
from .image_io import load_image_folder, load_image_stack, load_sldy_stack, natural_sort_key
from .manual_input import (
    coords_xy_to_rowcol,
    resample_polyline_xy,
    review_auto_midline_overlay,
    trace_midline_on_image,
)

RunMode = Literal["manual", "auto"]

# Extended kind to include sldy
DatasetKind = Literal["folder_sequence", "single_stack", "sldy"]


@dataclass
class DatasetSpec:
    dataset_id: str
    label: str
    kind: Literal["folder_sequence", "single_stack", "sldy"]
    source_path: Path
    files: list[Path]


@dataclass
class PipelineResult:
    dataset: DatasetSpec
    output_dir: Path
    init_mode_used: Literal["manual", "auto_approved", "auto_denied_manual", "auto_failed_manual"]
    status: Literal["ok", "failed"]
    message: str = ""


def discover_sldy_datasets(root_folder: str | Path) -> list[DatasetSpec]:
    """Discover SlideBook .sldy datasets under root_folder."""
    root = Path(root_folder)
    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")

    datasets: list[DatasetSpec] = []
    for sldy_path in sorted(root.rglob("*.sldy"), key=lambda p: natural_sort_key(str(p))):
        rel = sldy_path.relative_to(root)
        dataset_id = _sanitize_id(str(rel.with_suffix("")))
        datasets.append(
            DatasetSpec(
                dataset_id=dataset_id,
                label=str(rel),
                kind="sldy",
                source_path=sldy_path,
                files=[sldy_path],
            )
        )
    return datasets


def discover_tiff_datasets(root_folder: str | Path) -> list[DatasetSpec]:
    """Discover folder-of-folders TIFF datasets.

    Rules:
    - Any directory with 2+ TIFF files is treated as one folder sequence dataset.
    - Remaining TIFF files are treated as single-stack datasets.
    """
    root = Path(root_folder)
    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")

    all_tiffs = sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".tif", ".tiff")],
        key=lambda p: natural_sort_key(str(p.relative_to(root))),
    )
    by_dir: dict[Path, list[Path]] = {}
    for tif_path in all_tiffs:
        by_dir.setdefault(tif_path.parent, []).append(tif_path)

    datasets: list[DatasetSpec] = []
    consumed: set[Path] = set()
    for directory, files in sorted(by_dir.items(), key=lambda kv: str(kv[0])):
        files_sorted = sorted(files, key=lambda p: natural_sort_key(p.name))
        if len(files_sorted) >= 2:
            rel = directory.relative_to(root)
            dataset_id = _sanitize_id(str(rel)) or _sanitize_id(directory.name)
            datasets.append(
                DatasetSpec(
                    dataset_id=dataset_id,
                    label=str(rel),
                    kind="folder_sequence",
                    source_path=directory,
                    files=files_sorted,
                )
            )
            consumed.update(files_sorted)

    for tif_path in all_tiffs:
        if tif_path in consumed:
            continue
        rel = tif_path.relative_to(root)
        dataset_id = _sanitize_id(str(rel.with_suffix("")))
        datasets.append(
            DatasetSpec(
                dataset_id=dataset_id,
                label=str(rel),
                kind="single_stack",
                source_path=tif_path,
                files=[tif_path],
            )
        )

    return sorted(datasets, key=lambda ds: ds.dataset_id)


def load_dataset_stack(dataset: DatasetSpec) -> np.ndarray:
    if dataset.kind == "folder_sequence":
        return load_image_folder(dataset.source_path)
    if dataset.kind == "sldy":
        return load_sldy_stack(dataset.source_path)
    return load_image_stack(dataset.source_path)


def run_dataset(
    dataset: DatasetSpec,
    output_root: str | Path,
    mode: RunMode = "manual",
    checkpoint_path: Optional[str] = None,
    spacing_px: float = 15.0,
    disk_radius: int = 28,
    threshold: float = 10.0,
    time_interval: float = 1.0,
    time_unit: str = "frame",
    auto_review_callback: Optional[Callable[[np.ndarray, AutoMidlineResult], bool]] = None,
    manual_trace_callback: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    naming_config: Optional[NamingConfig] = None,
    tracking_frame_callback: Optional[Callable[[int, int, np.ndarray, np.ndarray], None]] = None,
    training_data_dir: Optional[str | Path] = None,
) -> PipelineResult:
    out_root = Path(output_root)
    out_dir = out_root / dataset.dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        images = load_dataset_stack(dataset)
        first_frame = images[0]
        initial_points: np.ndarray
        init_mode_used: PipelineResult.__annotations__["init_mode_used"] = "manual"

        collected_coords_xy: Optional[list] = None
        if mode == "auto":
            auto_result = extract_auto_midline(first_frame, checkpoint_path=checkpoint_path)
            reviewer = auto_review_callback or _default_auto_review
            approved = reviewer(first_frame, auto_result)
            if approved:
                initial_points = auto_coords_to_initial_points(auto_result.midline_coords_xy, spacing_px=spacing_px)
                init_mode_used = "auto_approved"
                collected_coords_xy = [list(p) for p in auto_result.midline_coords_xy]
            else:
                tracer = manual_trace_callback or _default_manual_trace
                traced_xy = tracer(first_frame, spacing_px)
                initial_points = manual_coords_to_initial_points(traced_xy, spacing_px=spacing_px)
                init_mode_used = "auto_denied_manual"
                collected_coords_xy = traced_xy.tolist() if hasattr(traced_xy, "tolist") else [list(p) for p in traced_xy]
            _write_auto_metadata(out_dir, auto_result, approved=approved)
        else:
            tracer = manual_trace_callback or _default_manual_trace
            traced_xy = tracer(first_frame, spacing_px)
            initial_points = manual_coords_to_initial_points(traced_xy, spacing_px=spacing_px)
            init_mode_used = "manual"
            collected_coords_xy = traced_xy.tolist() if hasattr(traced_xy, "tolist") else [list(p) for p in traced_xy]

        if training_data_dir is not None and collected_coords_xy is not None:
            _save_training_sample(training_data_dir, dataset.dataset_id, first_frame, collected_coords_xy, init_mode_used)

        if progress_callback:
            progress_callback(f"[{dataset.dataset_id}] running tracking and kinematics")

        profile = run_steady_state_kinematics(
            images=images,
            initial_points=initial_points,
            disk_radius=disk_radius,
            threshold=threshold,
            time_interval=time_interval,
            time_unit=time_unit,
            frame_viz_callback=tracking_frame_callback,
        )
        save_results(profile, out_dir)
        _write_run_metadata(dataset, out_dir, init_mode_used)

        if naming_config is not None and naming_config.enabled:
            folder_name = dataset.source_path.stem
            csv_stem = naming_config.derive_name(folder_name)
            if csv_stem is not None:
                dest = Path(output_root) / naming_config.output_subdir / f"{csv_stem}.csv"
                save_analysis_ready_csv(profile, dest)
                if progress_callback:
                    progress_callback(f"[{dataset.dataset_id}] analysis-ready: {dest.name}")
            else:
                if progress_callback:
                    progress_callback(
                        f"[{dataset.dataset_id}] WARNING: not enough tokens in '{folder_name}' for naming template"
                    )

        return PipelineResult(dataset=dataset, output_dir=out_dir, init_mode_used=init_mode_used, status="ok")
    except Exception as exc:
        return PipelineResult(
            dataset=dataset,
            output_dir=out_dir,
            init_mode_used="auto_failed_manual" if mode == "auto" else "manual",
            status="failed",
            message=str(exc),
        )


def run_batch(
    root_folder: str | Path,
    output_root: str | Path,
    mode: RunMode = "manual",
    checkpoint_path: Optional[str] = None,
    spacing_px: float = 15.0,
    disk_radius: int = 28,
    threshold: float = 10.0,
    time_interval: float = 1.0,
    time_unit: str = "frame",
    auto_review_callback: Optional[Callable[[np.ndarray, AutoMidlineResult], bool]] = None,
    manual_trace_callback: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    naming_config: Optional[NamingConfig] = None,
    tracking_frame_callback: Optional[Callable[[int, int, np.ndarray, np.ndarray], None]] = None,
    training_data_dir: Optional[str | Path] = None,
) -> list[PipelineResult]:
    datasets = discover_tiff_datasets(root_folder) + discover_sldy_datasets(root_folder)
    datasets = sorted(datasets, key=lambda ds: ds.dataset_id)
    if not datasets:
        raise FileNotFoundError("No TIFF or .sldy datasets found under selected root folder.")

    if progress_callback:
        progress_callback(f"Found {len(datasets)} datasets.")

    results: list[PipelineResult] = []
    for idx, dataset in enumerate(datasets, start=1):
        if progress_callback:
            progress_callback(f"Processing {idx}/{len(datasets)}: {dataset.label}")
        result = run_dataset(
            dataset=dataset,
            output_root=output_root,
            mode=mode,
            checkpoint_path=checkpoint_path,
            spacing_px=spacing_px,
            disk_radius=disk_radius,
            threshold=threshold,
            time_interval=time_interval,
            time_unit=time_unit,
            auto_review_callback=auto_review_callback,
            manual_trace_callback=manual_trace_callback,
            progress_callback=progress_callback,
            naming_config=naming_config,
            tracking_frame_callback=tracking_frame_callback,
            training_data_dir=training_data_dir,
        )
        results.append(result)
    return results


def auto_coords_to_initial_points(midline_coords_xy: list[tuple[float, float]], spacing_px: float = 15.0) -> np.ndarray:
    if len(midline_coords_xy) < 2:
        raise ValueError("Auto midline has fewer than 2 points.")
    coords = np.array(midline_coords_xy, dtype=float)
    # Ensure tip->base by placing right-most point first.
    if coords[0, 0] < coords[-1, 0]:
        coords = coords[::-1]
    sampled = resample_polyline_xy(coords, spacing_px=spacing_px)
    return coords_xy_to_rowcol(sampled)


def manual_coords_to_initial_points(trace_xy: np.ndarray, spacing_px: float = 15.0) -> np.ndarray:
    coords = np.asarray(trace_xy, dtype=float)
    if coords.shape[0] < 2:
        raise ValueError("Manual trace must contain at least 2 points.")
    sampled = resample_polyline_xy(coords, spacing_px=spacing_px)
    return coords_xy_to_rowcol(sampled)


def _default_auto_review(first_frame: np.ndarray, auto_result: AutoMidlineResult) -> bool:
    return review_auto_midline_overlay(
        first_frame,
        auto_result.midline_coords_xy,
        auto_result.qc_point_xy,
        title=f"Auto midline ({auto_result.method}, conf={auto_result.confidence:.2f})",
    )


def _default_manual_trace(first_frame: np.ndarray, spacing_px: float) -> np.ndarray:
    return trace_midline_on_image(first_frame, spacing_px=spacing_px)


def _write_auto_metadata(output_dir: Path, auto_result: AutoMidlineResult, approved: bool) -> None:
    payload = {
        "method": auto_result.method,
        "confidence": auto_result.confidence,
        "approved": approved,
        "qc_point_xy": {"x": auto_result.qc_point_xy[0], "y": auto_result.qc_point_xy[1]},
        "midline_num_points": len(auto_result.midline_coords_xy),
        "notes": auto_result.notes,
    }
    with open(output_dir / "auto_midline_review.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _write_run_metadata(dataset: DatasetSpec, output_dir: Path, init_mode_used: str) -> None:
    payload = {
        "dataset_id": dataset.dataset_id,
        "dataset_label": dataset.label,
        "dataset_kind": dataset.kind,
        "source_path": str(dataset.source_path),
        "input_files": [str(p) for p in dataset.files],
        "initialization_mode": init_mode_used,
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _sanitize_id(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw).strip("_")
    return cleaned or "dataset"


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float64)
    lo, hi = a.min(), a.max()
    if hi <= 1.0:
        a = a * 255.0
    elif hi > 255.0:
        a = (a - lo) / (hi - lo) * 255.0 if hi > lo else a * 0.0
    return np.clip(a, 0, 255).astype(np.uint8)


def _save_training_sample(
    training_data_dir: str | Path,
    dataset_id: str,
    image: np.ndarray,
    coords_xy: list,
    mode: str,
) -> None:
    try:
        ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        sample_dir = Path(training_data_dir) / f"{dataset_id}_{ts_ms}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        try:
            import cv2
            img_u8 = _normalize_to_uint8(image)
            cv2.imwrite(str(sample_dir / "image.png"), img_u8)
        except ImportError:
            # cv2 not available — fall back to a simple PGM write
            img_u8 = _normalize_to_uint8(image)
            h, w = img_u8.shape[:2]
            pgm_path = sample_dir / "image.png"
            with open(pgm_path, "wb") as fh:
                fh.write(f"P5\n{w} {h}\n255\n".encode())
                fh.write(img_u8.tobytes())

        meta = {
            "coords_xy": coords_xy,
            "mode": mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(sample_dir / "coords.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
    except Exception:
        pass  # Never block the pipeline
