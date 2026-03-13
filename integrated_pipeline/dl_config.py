"""
Inference-only configuration for the bundled root midline extraction model.

Training-time path constants are set to None — only inference constants
(IMAGE_SIZE, ENCODER_NAME, etc.) are used at runtime.
"""

# =============================================================================
# Paths (training-only — not used at inference)
# =============================================================================

DATASET_ROOT = None
IMAGES_DIR = None
MIDLINE_MASKS_DIR = None
QC_HEATMAPS_DIR = None
PREVIEW_DIR = None
META_MAT_PATH = None
OUTPUT_DIR = None
CHECKPOINT_DIR = None
LOG_DIR = None
PREDICTIONS_DIR = None

# Supported raw image extensions (checked in order when resolving image paths)
IMAGE_EXTENSIONS = [".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp"]

# =============================================================================
# Pre-processing: Classical Root Cropping
# =============================================================================

# Gaussian blur kernel size for root detection (must be odd)
CROP_BLUR_KERNEL = 51

# Padding around detected root bounding box (fraction of bbox size)
CROP_PADDING_FRACTION = 0.15

# Minimum padding in pixels on each side
CROP_PADDING_MIN_PX = 100

# Morphological kernel sizes for closing and opening
CROP_MORPH_CLOSE_KERNEL = 51
CROP_MORPH_OPEN_KERNEL = 21

# =============================================================================
# Model
# =============================================================================

# Network input size (after cropping and resizing)
IMAGE_SIZE = (512, 512)  # (H, W)

# Encoder backbone (from segmentation-models-pytorch)
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = None  # Not needed at inference — checkpoint weights are loaded directly

# Number of input channels (grayscale = 1)
IN_CHANNELS = 1

# =============================================================================
# Loss weights (used by CombinedLoss in model definition)
# =============================================================================

LAMBDA_QC = 1.0

# =============================================================================
# Miscellaneous
# =============================================================================

SEED = 42
