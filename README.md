# All-in-one integrated pipeline

This project contains an isolated implementation that combines:

- manual root midline tracing and tracking,
- automatic first-frame midline extraction with user approval/deny,
- fallback to manual tracing when auto is denied,
- steady-state kinematics analysis and CSV outputs.

Legacy projects remain untouched.

## Run the UI

```bash
python main.py
```

## Run the Streamlit UI

```bash
streamlit run streamlit_app.py
```

Manual tracing in Streamlit supports drawing directly on the first frame via
an in-browser canvas (tip -> base).

## Output contract

Each processed dataset writes a folder containing:

- `rawData.csv`
- `fitData.csv`
- `xyCoord.csv`
- `REGR_peak_location.csv`
- `REGR_peak_value.csv`
- `EVANS_zone.csv`, `PERCENT_zone.csv`, `ABS_zone.csv`
- `run_metadata.json`
- `auto_midline_review.json` (auto mode only)

## Smoke test

```bash
python tests/smoke_test.py
```
