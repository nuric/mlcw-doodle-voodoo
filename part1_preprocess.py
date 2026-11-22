"""Part 1 Preprocessing: Rasterize Quick, Draw! sketches into train/val/test splits."""

import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from binary_file_parser import unpack_drawings
from helpers import center_and_scale_strokes, rasterize_drawing
from part1_common import (
    DATA_DIR,
    IMAGE_SIZE,
    PART1_OUTPUT_DIR,
    RANDOM_SEED,
    set_seed,
)

# =============================================================================
# Global Configuration (Edit these to experiment!)
# =============================================================================
# Pick a subset of categories, initially 10 categories. These categories are the names of the files in the `data/` directory (e.g., `data/apple.bin`).
# ### BEGIN SOLUTION ###
pass
# ### END SOLUTION ###

# Maximum number of strokes to render per drawing (None = render all strokes).
MAX_STROKES = 5

# Dataset size & filtering
MAX_SAMPLES_PER_CATEGORY = 1500
RECOGNIZED_ONLY = True

# Split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# =============================================================================
# Category Loading & Rasterization
# =============================================================================
def load_and_rasterize_category(
    category,
    label_idx,
    data_dir=DATA_DIR,
    max_samples=MAX_SAMPLES_PER_CATEGORY,
    max_strokes=MAX_STROKES,
    recognized_only=RECOGNIZED_ONLY,
    image_size=IMAGE_SIZE,
):
    """Load drawings for one category, truncate strokes, and rasterize to [0, 1]."""
    bin_path = data_dir / f"{category}.bin"
    images = []

    for drawing in unpack_drawings(bin_path):
        if recognized_only and not drawing["recognized"]:
            continue
        if not drawing["image"]:
            continue

        strokes = drawing["image"]
        if max_strokes is not None:
            strokes = strokes[:max_strokes]
        drawing = {"image": center_and_scale_strokes(strokes)}
        images.append(rasterize_drawing(drawing, size=image_size))

        if len(images) >= max_samples:
            break

    X_cat = np.asarray(images, dtype=np.float32) / 255.0
    y_cat = np.full(len(images), label_idx, dtype=np.int64)
    return X_cat, y_cat


# =============================================================================
# Visualization Helper
# =============================================================================
def save_sample_grid(X, y, categories, output_dir=PART1_OUTPUT_DIR):
    """Save a grid showing one sample rasterized image per category."""
    n_cats = len(categories)
    cols = min(5, n_cats)
    rows = (n_cats + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, cat in enumerate(categories):
        match = np.where(y == idx)[0]
        ax = axes[idx]
        if len(match) > 0:
            img = X[match[0]].reshape(IMAGE_SIZE, IMAGE_SIZE)
            ax.imshow(img, cmap="gray_r")
        ax.set_title(cat, fontsize=9)
        ax.axis("off")

    for idx in range(n_cats, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(f"Sample Rasterized Sketches (max_strokes={MAX_STROKES})", fontsize=11)
    fig.tight_layout()
    save_path = output_dir / "sample_grid.pdf"
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved sample grid to {save_path}")


# =============================================================================
# Main Preprocessing Pipeline
# =============================================================================
def main():
    set_seed(RANDOM_SEED)
    assert abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    PART1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Categories ({len(CATEGORIES)}): {CATEGORIES}")
    print(
        f"Settings: max_strokes={MAX_STROKES}, max_samples/cat={MAX_SAMPLES_PER_CATEGORY}, "
        f"recognized_only={RECOGNIZED_ONLY}, image_size={IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    # Load and rasterize each category
    X_parts, y_parts = [], []
    for label_idx, category in enumerate(CATEGORIES):
        X_cat, y_cat = load_and_rasterize_category(category, label_idx)
        X_parts.append(X_cat)
        y_parts.append(y_cat)
        print(f"  [{label_idx}] {category:<15s} -> {len(X_cat)} samples")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    X_train, X_val, X_test = None, None, None
    # Use the already imported train_test_split to create train/val/test splits with VAL_RATIO and TEST_RATIO.
    # ### BEGIN SOLUTION ###
    pass
    # ### END SOLUTION ###

    # Save splits and metadata
    np.savez_compressed(PART1_OUTPUT_DIR / "train.npz", X=X_train, y=y_train)
    np.savez_compressed(PART1_OUTPUT_DIR / "val.npz", X=X_val, y=y_val)
    np.savez_compressed(PART1_OUTPUT_DIR / "test.npz", X=X_test, y=y_test)

    metadata = {
        "categories": CATEGORIES,
        "image_size": IMAGE_SIZE,
        "max_strokes": MAX_STROKES,
    }
    with open(PART1_OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    save_sample_grid(X_train, y_train, CATEGORIES)
    print(
        f"\nSaved preprocessed splits to '{PART1_OUTPUT_DIR}/': "
        f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )


if __name__ == "__main__":
    main()
