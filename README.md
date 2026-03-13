# All-in-One Root Kinematics

A PySide6 desktop app that combines root midline initialization, affine patch tracking, and steady-state kinematics analysis into a single pipeline.

## Features

- **Auto midline detection** — classical Otsu threshold + per-column medians, with optional deep-learning upgrade via a PyTorch checkpoint
- **Manual midline tracing** — click-trace on the first frame in a PySide6 dialog
- **Affine patch tracking** — Lucas-Kanade iterative tracking propagates midline points frame-to-frame
- **Kinematics analysis** — logistic velocity profile fit, REGR/strain profiles, Evans/percent/absolute growth zone metrics
- **Analysis viewer** — second tab for browsing and comparing completed run outputs
- **SlideBook support** — `.sldy` files are discovered and processed alongside TIFFs (requires `slidebook-python`)

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Input formats

| Format | How it's detected |
|---|---|
| Folder of TIFFs (2+ files) | Treated as a frame sequence |
| Single multi-page TIFF | Treated as an image stack |
| `.sldy` file | Loaded via `slidebook-python` |

Point the pipeline at a root folder — all datasets underneath it are discovered and processed in batch.

## Output

Each processed dataset writes to `<output_root>/<dataset_id>/`:

| File | Contents |
|---|---|
| `rawData.csv` | Raw (distance, velocity) scatter |
| `fitData.csv` | Fitted (l, velocity, strain) profiles |
| `xyCoord.csv` | Tracked point coordinates per frame |
| `REGR_peak_location.csv` | Location of peak strain rate |
| `REGR_peak_value.csv` | Peak strain rate value |
| `EVANS_zone.csv`, `PERCENT_zone.csv`, `ABS_zone.csv` | Growth zone masks (+ `_width` scalars) |
| `REGR.png` | Velocity and strain rate plot |
| `run_metadata.json` | Dataset info and initialization mode |
| `auto_midline_review.json` | Auto method, confidence, approval (auto mode only) |

## Smoke test

```bash
python tests/smoke_test.py
```

Runs the full pipeline on synthetic data non-interactively and exits with `Smoke tests passed.`
