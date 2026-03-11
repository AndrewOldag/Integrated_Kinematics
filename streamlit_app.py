"""Streamlit front-end for the integrated root kinematics pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from integrated_pipeline.analysis import run_steady_state_kinematics, save_results
from integrated_pipeline.auto_midline import AutoMidlineResult, extract_auto_midline
from integrated_pipeline.data_loader import discover_output_profiles
from integrated_pipeline.pipeline_runner import (
    DatasetSpec,
    auto_coords_to_initial_points,
    discover_tiff_datasets,
    load_dataset_stack,
    manual_coords_to_initial_points,
    run_batch,
)


def parse_points_text(points_text: str) -> np.ndarray:
    points: list[tuple[float, float]] = []
    for line in points_text.strip().splitlines():
        raw = line.strip()
        if not raw:
            continue
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid line '{raw}'. Use format: x,y")
        x_val = float(parts[0])
        y_val = float(parts[1])
        points.append((x_val, y_val))
    if len(points) < 2:
        raise ValueError("Provide at least 2 points.")
    return np.array(points, dtype=float)


def ensure_output_dir(base: str, dataset_id: str) -> Path:
    out_dir = Path(base) / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_run_metadata(dataset: DatasetSpec, output_dir: Path, init_mode_used: str) -> None:
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


def write_auto_metadata(output_dir: Path, auto_result: AutoMidlineResult, approved: bool) -> None:
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


st.set_page_config(page_title="All-in-One Root Kinematics", page_icon="🌱", layout="wide")
st.title("All-in-One Root Kinematics")
st.caption("Modern Streamlit interface for manual/auto midline + tracking + analysis.")

if "datasets" not in st.session_state:
    st.session_state.datasets = []
if "auto_cache" not in st.session_state:
    st.session_state.auto_cache = {}
if "frame_cache" not in st.session_state:
    st.session_state.frame_cache = {}


def _first_frame_for_dataset(dataset: DatasetSpec) -> np.ndarray:
    cached = st.session_state.frame_cache.get(dataset.dataset_id)
    if cached is not None:
        return cached
    images = load_dataset_stack(dataset)
    first = images[0]
    st.session_state.frame_cache[dataset.dataset_id] = first
    return first


def _to_rgb_u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 3:
        rgb = arr
    else:
        gray = arr.astype(np.float64)
        if gray.max() <= 1.0:
            gray = gray * 255.0
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=2)
    return rgb.astype(np.uint8)


def _extract_polyline_from_canvas_json(json_data: dict | None) -> np.ndarray | None:
    if not json_data:
        return None
    objects = json_data.get("objects", [])
    points: list[tuple[float, float]] = []
    for obj in objects:
        if obj.get("type") != "path":
            continue
        left = float(obj.get("left", 0.0))
        top = float(obj.get("top", 0.0))
        for seg in obj.get("path", []):
            if not isinstance(seg, list) or len(seg) < 3:
                continue
            cmd = str(seg[0]).upper()
            if cmd in {"M", "L", "Q", "C"}:
                x_val = float(seg[-2]) + left
                y_val = float(seg[-1]) + top
                points.append((x_val, y_val))
    if len(points) < 2:
        return None
    return np.array(points, dtype=float)


def render_manual_canvas(dataset: DatasetSpec, frame: np.ndarray, height_px: int = 520) -> np.ndarray | None:
    st.caption("Draw one continuous path from tip -> base. Use clear to redraw.")
    rgb = _to_rgb_u8(frame)
    h, w = rgb.shape[:2]
    if w == 0 or h == 0:
        return None
    width_px = max(1, int(round((height_px / h) * w)))
    pil_bg = Image.fromarray(rgb)
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=3,
        stroke_color="#00FF66",
        background_image=pil_bg,
        update_streamlit=True,
        height=height_px,
        width=width_px,
        drawing_mode="freedraw",
        key=f"manual_canvas_{dataset.dataset_id}",
    )
    return _extract_polyline_from_canvas_json(canvas_result.json_data)

with st.sidebar:
    st.header("Pipeline Settings")
    input_root = st.text_input("TIFF root folder", value=r"D:\nolan_lab\kinematics_data")
    output_root = st.text_input("Output folder", value=r"D:\nolan_lab\all_in_one\integrated_outputs")
    checkpoint_path = st.text_input("Auto checkpoint (.pth, optional)", value="")
    mode = st.selectbox("Initialization mode", ["auto", "manual"], index=0)
    spacing_px = st.number_input("Point spacing (px)", min_value=1.0, max_value=500.0, value=15.0, step=1.0)
    disk_radius = st.number_input("Disk radius", min_value=1, max_value=500, value=28, step=1)
    threshold = st.number_input("Threshold", min_value=0.0, max_value=500.0, value=10.0, step=1.0)
    time_interval = st.number_input("Time interval", min_value=0.001, max_value=1_000_000.0, value=1.0, step=0.1)
    time_unit = st.text_input("Time unit", value="frame")

    st.divider()
    st.subheader("Batch Auto Shortcut")
    auto_approve_all = st.checkbox("Auto-approve all datasets in batch mode", value=True)
    run_batch_btn = st.button("Run Full Batch")

col_left, col_right = st.columns([1.2, 1.0])

with col_left:
    if st.button("Discover datasets"):
        try:
            discovered = discover_tiff_datasets(input_root)
            st.session_state.datasets = discovered
            st.success(f"Found {len(discovered)} dataset(s).")
        except Exception as exc:
            st.error(str(exc))

    datasets: list[DatasetSpec] = st.session_state.datasets
    if datasets:
        selected_label = st.selectbox(
            "Choose dataset for interactive run",
            options=[f"{d.dataset_id}  |  {d.label}" for d in datasets],
        )
        selected_idx = [f"{d.dataset_id}  |  {d.label}" for d in datasets].index(selected_label)
        selected_dataset = datasets[selected_idx]
        st.write(f"**Kind:** `{selected_dataset.kind}`")
        st.write(f"**Source:** `{selected_dataset.source_path}`")
        with st.expander("Input files"):
            for f in selected_dataset.files:
                st.code(str(f), language="text")

        if st.button("Load first frame for drawing/review"):
            try:
                _ = _first_frame_for_dataset(selected_dataset)
                st.success("First frame loaded.")
            except Exception as exc:
                st.error(f"Failed to load frame: {exc}")

        if mode == "auto":
            auto_key = selected_dataset.dataset_id
            if st.button("Run auto midline on first frame"):
                with st.spinner("Running auto midline..."):
                    try:
                        first_frame = _first_frame_for_dataset(selected_dataset)
                        auto_result = extract_auto_midline(
                            first_frame, checkpoint_path=checkpoint_path.strip() or None
                        )
                        st.session_state.auto_cache[auto_key] = {
                            "frame": first_frame,
                            "result": auto_result,
                        }
                        st.success("Auto midline generated.")
                    except Exception as exc:
                        st.error(f"Auto midline failed: {exc}")

            cached = st.session_state.auto_cache.get(auto_key)
            if cached is not None:
                auto_result: AutoMidlineResult = cached["result"]
                st.image(
                    auto_result.overlay_image,
                    caption=f"Auto overlay ({auto_result.method}, conf={auto_result.confidence:.2f})",
                    use_container_width=True,
                )
                auto_decision = st.radio("Auto midline decision", ["Approve", "Deny"], horizontal=True)
            else:
                auto_decision = "Deny"

            first_frame_for_manual = None
            try:
                first_frame_for_manual = _first_frame_for_dataset(selected_dataset)
            except Exception:
                first_frame_for_manual = None

            canvas_points = None
            if first_frame_for_manual is not None:
                canvas_points = render_manual_canvas(selected_dataset, first_frame_for_manual)
                if canvas_points is not None:
                    st.success(f"Captured {len(canvas_points)} drawn points.")

            manual_points = st.text_area(
                "Optional manual fallback points (x,y per line; tip->base). "
                "If left empty, canvas drawing is used.",
                value="",
                height=160,
            )

            if st.button("Run selected dataset"):
                with st.spinner("Running tracking + analysis..."):
                    try:
                        images = load_dataset_stack(selected_dataset)
                        init_mode_used = "auto_approved"
                        if cached is not None and auto_decision == "Approve":
                            auto_result = cached["result"]
                            initial_points = auto_coords_to_initial_points(
                                auto_result.midline_coords_xy, spacing_px=spacing_px
                            )
                            approved = True
                        else:
                            if manual_points.strip():
                                manual_xy = parse_points_text(manual_points)
                            elif canvas_points is not None:
                                manual_xy = canvas_points
                            else:
                                raise ValueError("Provide manual fallback points using canvas or text.")
                            initial_points = manual_coords_to_initial_points(
                                manual_xy, spacing_px=spacing_px
                            )
                            init_mode_used = "auto_denied_manual"
                            approved = False
                            auto_result = cached["result"] if cached is not None else None

                        profile = run_steady_state_kinematics(
                            images=images,
                            initial_points=initial_points,
                            disk_radius=int(disk_radius),
                            threshold=float(threshold),
                            time_interval=float(time_interval),
                            time_unit=time_unit.strip() or "frame",
                        )
                        out_dir = ensure_output_dir(output_root, selected_dataset.dataset_id)
                        save_results(profile, out_dir)
                        write_run_metadata(selected_dataset, out_dir, init_mode_used)
                        if auto_result is not None:
                            write_auto_metadata(out_dir, auto_result, approved=approved)
                        st.success(f"Completed. Outputs saved to {out_dir}")
                    except Exception as exc:
                        st.error(f"Run failed: {exc}")
        else:
            first_frame_for_manual = None
            try:
                first_frame_for_manual = _first_frame_for_dataset(selected_dataset)
            except Exception:
                first_frame_for_manual = None

            canvas_points = None
            if first_frame_for_manual is not None:
                canvas_points = render_manual_canvas(selected_dataset, first_frame_for_manual)
                if canvas_points is not None:
                    st.success(f"Captured {len(canvas_points)} drawn points.")

            manual_points = st.text_area(
                "Optional manual points (x,y per line, tip->base). "
                "If left empty, canvas drawing is used.",
                value="",
                height=200,
            )
            if st.button("Run selected dataset"):
                with st.spinner("Running tracking + analysis..."):
                    try:
                        if manual_points.strip():
                            manual_xy = parse_points_text(manual_points)
                        elif canvas_points is not None:
                            manual_xy = canvas_points
                        else:
                            raise ValueError("Provide manual points using canvas or text.")
                        initial_points = manual_coords_to_initial_points(manual_xy, spacing_px=spacing_px)
                        images = load_dataset_stack(selected_dataset)
                        profile = run_steady_state_kinematics(
                            images=images,
                            initial_points=initial_points,
                            disk_radius=int(disk_radius),
                            threshold=float(threshold),
                            time_interval=float(time_interval),
                            time_unit=time_unit.strip() or "frame",
                        )
                        out_dir = ensure_output_dir(output_root, selected_dataset.dataset_id)
                        save_results(profile, out_dir)
                        write_run_metadata(selected_dataset, out_dir, "manual")
                        st.success(f"Completed. Outputs saved to {out_dir}")
                    except Exception as exc:
                        st.error(f"Run failed: {exc}")
    else:
        st.info("Click **Discover datasets** after setting the TIFF root folder.")

with col_right:
    st.subheader("Output Summary")
    if st.button("Refresh output summary"):
        pass
    profiles = discover_output_profiles(output_root)
    if not profiles:
        st.write("No output profiles found yet.")
    else:
        for profile in profiles:
            peak = f"{profile.regr_peak:.5g}" if profile.regr_peak is not None else "n/a"
            peak_loc = f"{profile.regr_peak_location:.5g}" if profile.regr_peak_location is not None else "n/a"
            st.markdown(
                f"**{profile.dataset_id}**  \n"
                f"mode: `{profile.initialization_mode}`  \n"
                f"points: `{len(profile.l_domain)}`  \n"
                f"REGR peak: `{peak}` at `{peak_loc}`  \n"
                f"path: `{profile.output_dir}`"
            )
            st.divider()

if run_batch_btn:
    if mode != "auto":
        st.warning("Full batch shortcut currently supports auto mode only.")
    elif not auto_approve_all:
        st.warning("Enable 'Auto-approve all datasets' for batch mode, or run datasets interactively.")
    else:
        with st.spinner("Running full auto batch..."):
            try:
                results = run_batch(
                    root_folder=input_root,
                    output_root=output_root,
                    mode="auto",
                    checkpoint_path=checkpoint_path.strip() or None,
                    spacing_px=float(spacing_px),
                    disk_radius=int(disk_radius),
                    threshold=float(threshold),
                    time_interval=float(time_interval),
                    time_unit=time_unit.strip() or "frame",
                    auto_review_callback=(lambda _img, _res: bool(auto_approve_all)),
                )
                ok = sum(1 for r in results if r.status == "ok")
                st.success(f"Batch finished. OK: {ok}, Failed: {len(results) - ok}")
            except Exception as exc:
                st.error(f"Batch run failed: {exc}")
