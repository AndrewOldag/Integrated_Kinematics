"""Smoke tests for manual/auto integrated flow without interactive UI."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrated_pipeline.pipeline_runner import run_batch


def _make_synthetic_stack(path: Path, frames: int = 8, h: int = 140, w: int = 220) -> None:
    stack = np.ones((frames, h, w), dtype=np.float32)
    for t in range(frames):
        # Root-like diagonal band drifting right over time.
        for x in range(20, 180):
            y = int(30 + 0.28 * x + 0.4 * t)
            y0 = max(0, y - 2)
            y1 = min(h, y + 3)
            x0 = max(0, x - 1)
            x1 = min(w, x + 2)
            stack[t, y0:y1, x0:x1] = 0.1
    tifffile.imwrite(str(path), (stack * 65535).astype(np.uint16))


def _noninteractive_manual(first_frame: np.ndarray, spacing_px: float) -> np.ndarray:
    h, w = first_frame.shape[:2]
    xs = np.linspace(25, w - 30, 22)
    ys = 30 + 0.28 * xs
    return np.column_stack([xs, ys])


def run_smoke() -> None:
    root_tmp = Path(tempfile.mkdtemp(prefix="aio_smoke_"))
    try:
        input_root = root_tmp / "input"
        output_root = root_tmp / "output"
        seq_dir = input_root / "nested" / "dataset_A"
        seq_dir.mkdir(parents=True, exist_ok=True)
        _make_synthetic_stack(seq_dir / "sample_stack.tiff")

        # 1) Manual-only path
        manual_out = output_root / "manual"
        res_manual = run_batch(
            root_folder=input_root,
            output_root=manual_out,
            mode="manual",
            manual_trace_callback=_noninteractive_manual,
        )
        assert res_manual and all(r.status == "ok" for r in res_manual), "Manual smoke test failed."
        assert (manual_out / res_manual[0].dataset.dataset_id / "fitData.csv").exists()

        # 2) Auto approved path
        auto_ok_out = output_root / "auto_ok"
        res_auto_ok = run_batch(
            root_folder=input_root,
            output_root=auto_ok_out,
            mode="auto",
            auto_review_callback=lambda _img, _res: True,
            manual_trace_callback=_noninteractive_manual,
        )
        assert res_auto_ok and all(r.status == "ok" for r in res_auto_ok), "Auto approved smoke test failed."
        assert res_auto_ok[0].init_mode_used == "auto_approved"
        assert (auto_ok_out / res_auto_ok[0].dataset.dataset_id / "auto_midline_review.json").exists()

        # 3) Auto denied -> manual fallback path
        auto_deny_out = output_root / "auto_deny"
        res_auto_deny = run_batch(
            root_folder=input_root,
            output_root=auto_deny_out,
            mode="auto",
            auto_review_callback=lambda _img, _res: False,
            manual_trace_callback=_noninteractive_manual,
        )
        assert res_auto_deny and all(r.status == "ok" for r in res_auto_deny), "Auto deny smoke test failed."
        assert res_auto_deny[0].init_mode_used == "auto_denied_manual"
        assert (auto_deny_out / res_auto_deny[0].dataset.dataset_id / "xyCoord.csv").exists()

        print("Smoke tests passed.")
        print(f"Temporary artifacts: {root_tmp}")
    finally:
        shutil.rmtree(root_tmp, ignore_errors=True)


if __name__ == "__main__":
    run_smoke()
