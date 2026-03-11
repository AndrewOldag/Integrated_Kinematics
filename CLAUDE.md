# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Desktop UI (PySide6)
python main.py

# Streamlit UI (browser-based, supports canvas drawing)
streamlit run streamlit_app.py

# Smoke test (non-interactive, uses synthetic data)
python tests/smoke_test.py
```

Install dependencies: `pip install -r requirements.txt`

## Architecture

All logic lives in the `integrated_pipeline/` package. `main.py` and `streamlit_app.py` are thin launchers.

### Data flow

1. **Discovery** — `pipeline_runner.discover_tiff_datasets()` scans a root folder for TIFF datasets. Directories with 2+ TIFFs become `folder_sequence` datasets; lone TIFFs become `single_stack` datasets.
2. **Midline initialization** — for each dataset's first frame, get ordered (x, y) points along the root midline, tip → base, using either:
   - **Auto** (`auto_midline.extract_auto_midline`): classical Otsu threshold + per-column medians, optionally upgraded with a PyTorch checkpoint loaded from the legacy module at `D:/nolan_lab/root_midline_extraction/predict.py`. User approves or denies the result.
   - **Manual** (`manual_input.trace_midline_on_image`): Matplotlib click-trace (desktop) or `streamlit-drawable-canvas` freedraw (Streamlit).
3. **Coordinate conversion** — `manual_input.resample_polyline_xy` resamples to fixed `spacing_px`, then `coords_xy_to_rowcol` converts (x, y) → (row, col) for the tracker.
4. **Tracking** — `tracking.track_points` propagates points frame-to-frame using iterative affine patch matching on disk-shaped neighborhoods (`domain.create_disk_domain`). Points whose column is below `threshold` are frozen (root base).
5. **Kinematics analysis** — `analysis.steady_state_analysis` computes velocity and strain profiles by fitting a logistic model (`fitting.fit_logistic`) to raw displacement data. `analysis.compute_growth_zone_metrics` adds Evans, percent, and absolute growth zone widths.
6. **Output** — `analysis.save_results` writes CSVs and a `REGR.png` plot to `<output_root>/<dataset_id>/`.

### Module responsibilities

| Module | Role |
|---|---|
| `pipeline_runner.py` | `DatasetSpec`, `PipelineResult`, `run_dataset`, `run_batch`; orchestrates the full flow with callback hooks for UI interaction |
| `app.py` | PySide6 `MainWindow`; uses `QThread` + `threading.Event` to bridge blocking worker callbacks to the GUI |
| `streamlit_app.py` | Streamlit UI; single-dataset interactive mode and full batch shortcut |
| `analysis.py` | `run_steady_state_kinematics`, `steady_state_analysis`, `compute_growth_zone_metrics`, `save_results` |
| `tracking.py` | Affine Lucas-Kanade patch tracking via `scipy.ndimage.map_coordinates` |
| `auto_midline.py` | Classical and (optional) deep-learning midline extraction; `AutoMidlineResult` dataclass |
| `manual_input.py` | Matplotlib interactive tracing, `resample_polyline_xy`, `coords_xy_to_rowcol` |
| `fitting.py` | Logistic model fit for the velocity profile |
| `image_io.py` | TIFF stack/folder loading with natural-sort ordering |
| `data_loader.py` | `discover_output_profiles` — reads completed output directories for the summary panel |
| `domain.py` | Disk sample-point domain for patch tracking |
| `plotting.py` | Steady-state profile plots |

### UI interaction pattern (PySide6)

`PipelineWorker` runs `run_batch` in a `QThread`. When the pipeline needs user input (auto review or manual trace), the worker signals the main thread via `request_auto_review` / `request_manual_trace`, then blocks on a `threading.Event`. The main thread shows a dialog, then calls `worker.set_auto_response` / `worker.set_manual_response` to unblock.

### Coordinate conventions

- Midline traces are collected as `(x, y)` (column-first, image-space).
- The tracker stores and operates on `(row, col)` (row-first). Conversion: `coords_xy_to_rowcol` swaps axes.
- Tip is right-most (largest x); `auto_coords_to_initial_points` enforces this by reversing if needed.

### Output contract

Each processed dataset writes to `<output_root>/<dataset_id>/`:
- `rawData.csv` — raw (distance, velocity) scatter
- `fitData.csv` — fitted (l, velocity, strain) profiles
- `xyCoord.csv` — tracked point coordinates per frame
- `REGR_peak_location.csv`, `REGR_peak_value.csv`
- `EVANS_zone.csv`, `PERCENT_zone.csv`, `ABS_zone.csv` (+ `_width` scalars)
- `run_metadata.json` — dataset info and `initialization_mode`
- `auto_midline_review.json` — auto method, confidence, approval (auto mode only)
- `REGR.png` — velocity/strain plot
