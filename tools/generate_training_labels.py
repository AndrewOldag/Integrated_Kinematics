"""Generate training labels from collected midline trace samples.

Usage:
    python tools/generate_training_labels.py \
        --samples-dir ./training_samples \
        --dataset-dir D:/nolan_lab/root_midline_extraction/dataset/dataset \
        --val-fraction 0.15
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import scipy.io


def _make_midline_mask(image: np.ndarray, coords_xy: list[list[float]]) -> np.ndarray:
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([[int(round(x)), int(round(y))] for x, y in coords_xy], dtype=np.int32)
    cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=3)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel)
    return mask


def _make_qc_heatmap(image: np.ndarray, tip_xy: list[float], sigma: float = 15.0) -> np.ndarray:
    h, w = image.shape[:2]
    cx, cy = float(tip_xy[0]), float(tip_xy[1])
    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    gx = np.exp(-0.5 * ((xs - cx) / sigma) ** 2)
    gy = np.exp(-0.5 * ((ys - cy) / sigma) ** 2)
    heatmap = np.outer(gy, gx)
    heatmap = (heatmap / heatmap.max() * 65535).astype(np.uint16)
    return heatmap


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training labels from collected midline samples.")
    parser.add_argument("--samples-dir", required=True, help="Root directory of collected training samples")
    parser.add_argument("--dataset-dir", required=True, help="Target dataset directory for the model trainer")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Fraction of data to use for validation")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    dataset_dir = Path(args.dataset_dir)

    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "midline_masks"
    heatmaps_dir = dataset_dir / "qc_heatmaps"
    for d in [images_dir, masks_dir, heatmaps_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Find all samples
    sample_jsons = sorted(samples_dir.glob("*/coords.json"))
    if not sample_jsons:
        print(f"No samples found under {samples_dir}")
        return

    existing_stems = {p.stem for p in images_dir.glob("*.png")}
    new_count = 0

    for coords_path in sample_jsons:
        stem = coords_path.parent.name
        if stem in existing_stems:
            continue

        image_path = coords_path.parent / "image.png"
        if not image_path.exists():
            print(f"  SKIP {stem}: image.png not found")
            continue

        with open(coords_path, encoding="utf-8") as fh:
            meta = json.load(fh)

        coords_xy: list[list[float]] = meta["coords_xy"]
        if len(coords_xy) < 2:
            print(f"  SKIP {stem}: fewer than 2 coords")
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"  SKIP {stem}: could not read image")
            continue

        mask = _make_midline_mask(image, coords_xy)
        heatmap = _make_qc_heatmap(image, coords_xy[0])

        shutil.copy(image_path, images_dir / f"{stem}.png")
        cv2.imwrite(str(masks_dir / f"{stem}.png"), mask)
        cv2.imwrite(str(heatmaps_dir / f"{stem}.png"), heatmap)

        new_count += 1

    # Collect all stems in images/ and write meta.mat
    all_stems = sorted(p.stem for p in images_dir.glob("*.png"))
    total = len(all_stems)

    rng = random.Random(42)
    shuffled = list(all_stems)
    rng.shuffle(shuffled)

    n_val = max(1, round(total * args.val_fraction)) if total > 1 else 0
    val_stems = set(shuffled[:n_val])
    train_stems = [s for s in all_stems if s not in val_stems]
    val_stems_sorted = [s for s in all_stems if s in val_stems]

    train_indices = [all_stems.index(s) + 1 for s in train_stems]
    val_indices = [all_stems.index(s) + 1 for s in val_stems_sorted]

    scipy.io.savemat(
        str(dataset_dir / "meta.mat"),
        {"train_indices": np.array(train_indices, dtype=np.int32),
         "val_indices": np.array(val_indices, dtype=np.int32)},
    )

    print(f"New samples added: {new_count}")
    print(f"Total dataset size: {total}")
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")


if __name__ == "__main__":
    main()
