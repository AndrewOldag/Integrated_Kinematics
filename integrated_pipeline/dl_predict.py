"""
Inference pipeline for root midline extraction and QC localization.

Full pipeline:
    raw image -> classical root crop -> resize -> model -> scale back
    -> midline coordinates + QC point in original image space
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from skimage.morphology import skeletonize

from . import dl_config as config
from . import dl_preprocessing as preprocessing
from .dl_model import DualHeadUNet, build_model


def load_image_grayscale(path: str) -> Optional[np.ndarray]:
    """
    Load an image as grayscale uint8. Handles TIFF and other formats
    by falling back to Pillow if OpenCV fails.

    Returns:
        Grayscale uint8 array (H, W), or None if loading fails.
    """
    # Try OpenCV first
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        # Pillow fallback (handles more TIFF variants)
        try:
            pil_img = PILImage.open(path)
            gray = np.array(pil_img.convert("L"))
        except Exception:
            return None

    # Ensure uint8 (TIFF may be uint16)
    if gray.dtype != np.uint8:
        gray = (gray.astype(np.float32) / gray.max() * 255).astype(np.uint8)

    return gray


@torch.no_grad()
def predict_single(
    model: DualHeadUNet,
    image: np.ndarray,
    device: torch.device,
    target_size: Tuple[int, int] = config.IMAGE_SIZE,
) -> Dict:
    """
    Run inference on a single image.

    Full pipeline: crop -> resize -> predict -> map back to original coords.

    Args:
        model: Trained model in eval mode.
        image: Raw microscope image (H, W), grayscale uint8.
        device: Computation device.
        target_size: Network input size (H, W).

    Returns:
        Dict with:
            - 'midline_mask': Full-resolution binary midline mask (H, W)
            - 'midline_coords': Ordered (x, y) coordinates of the midline
            - 'qc_point': (x, y) of the QC location in original coords
            - 'qc_heatmap': Full-resolution QC heatmap (H, W)
            - 'bbox': (x, y, w, h) crop bounding box
            - 'midline_mask_raw': Network-resolution midline prediction
    """
    original_size = image.shape[:2]  # (H, W)

    # Step 1: Classical root region detection and crop
    bbox = preprocessing.find_root_bbox(image)
    bx, by, bw, bh = bbox

    cropped = image[by : by + bh, bx : bx + bw]
    crop_size = cropped.shape[:2]  # (H, W) of crop before resize

    # Step 2: Resize to network input size
    resized = preprocessing.resize_crop(cropped, target_size, cv2.INTER_LINEAR)

    # Step 3: Prepare tensor
    img_float = resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_float).unsqueeze(0).unsqueeze(0).to(device)

    # Step 4: Model forward pass
    model.eval()
    predictions = model(img_tensor)

    midline_raw = torch.sigmoid(predictions["midline"]).cpu().numpy()[0, 0]
    qc_raw = torch.sigmoid(predictions["qc"]).cpu().numpy()[0, 0]

    # Step 5: Map predictions back to original image coordinates
    midline_full = preprocessing.map_predictions_to_original(
        midline_raw, bbox, original_size, cv2.INTER_LINEAR
    )
    qc_full = preprocessing.map_predictions_to_original(
        qc_raw, bbox, original_size, cv2.INTER_LINEAR
    )

    # Step 6: Post-process midline -> skeleton -> ordered coordinates
    midline_binary = (midline_full > 0.5).astype(np.uint8)
    midline_skeleton = skeletonize(midline_binary).astype(np.uint8)
    midline_coords = extract_ordered_midline(midline_skeleton)

    # Step 7: Derive QC as the rightmost point of the midline (root tip)
    if midline_coords:
        qc_point = max(midline_coords, key=lambda pt: pt[0])
    else:
        # Fallback to heatmap peak if midline is empty
        qc_point = preprocessing.extract_qc_point(qc_raw, bbox, crop_size)

    return {
        "midline_mask": midline_binary,
        "midline_skeleton": midline_skeleton,
        "midline_coords": midline_coords,
        "qc_point": qc_point,
        "bbox": bbox,
        "midline_mask_raw": midline_raw,
    }


def extract_ordered_midline(
    skeleton: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Extract ordered (x, y) coordinates from a skeletonized midline mask.

    The coordinates are ordered from one end of the midline to the other
    by sorting along the principal axis of the point cloud.

    Args:
        skeleton: Binary skeleton image (H, W), uint8 with values 0 or 1.

    Returns:
        List of (x, y) coordinates ordered along the midline.
    """
    ys, xs = np.where(skeleton > 0)
    if len(xs) == 0:
        return []

    points = np.column_stack([xs, ys])  # (N, 2) as (x, y)

    if len(points) < 2:
        return [(int(xs[0]), int(ys[0]))]

    # Find principal axis using PCA to determine ordering direction
    centroid = points.mean(axis=0)
    centered = points - centroid

    # SVD to find principal direction
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    principal = vt[0]  # First principal component direction

    # Project points onto principal axis and sort
    projections = centered @ principal
    sort_idx = np.argsort(projections)

    ordered = points[sort_idx].tolist()
    return [(int(x), int(y)) for x, y in ordered]


def predict_directory(
    model: DualHeadUNet,
    image_dir: Path,
    device: torch.device,
    output_dir: Optional[Path] = None,
) -> List[Dict]:
    """
    Run inference on all images in a directory.

    Args:
        model: Trained model.
        image_dir: Directory containing images.
        device: Computation device.
        output_dir: Directory to save results (optional).

    Returns:
        List of result dicts from predict_single().
    """
    image_dir = Path(image_dir)
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    image_files = sorted([
        f for f in image_dir.iterdir()
        if f.suffix.lower() in extensions
    ])

    if not image_files:
        print(f"No images found in {image_dir}")
        return []

    print(f"Found {len(image_files)} images in {image_dir}")
    all_results = []

    for img_path in image_files:
        print(f"  Processing: {img_path.name}")

        image = load_image_grayscale(str(img_path))
        if image is None:
            print(f"    WARNING: Failed to read {img_path}, skipping.")
            continue

        result = predict_single(model, image, device)
        all_results.append(result)

    return all_results
