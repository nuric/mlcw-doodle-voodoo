"""Part 2 Step 1 — Preprocessing: Load, center, and resample QuickDraw strokes."""

import json

import numpy as np

from binary_file_parser import unpack_drawings
from helpers import center_and_scale_strokes
from part2_common import (
    DATA_DIR,
    MAX_CONTEXT_LEN,
    PART2_OUTPUT_DIR,
    POINTS_PER_STROKE,
    RANDOM_SEED,
    resample_stroke,
    set_seed,
    style_canvas,
)
import matplotlib.pyplot as plt

# =============================================================================
# Global Configuration (Edit these to experiment!)
# =============================================================================
# Pick a subset of categories (this time around 5).
# These categories are the names of the files in the `data/` directory (e.g., `data/face.bin`).
pass

MAX_DRAWINGS_PER_CATEGORY = 8000
MIN_STROKES = 2
RECOGNIZED_ONLY = True

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# =============================================================================
# Drawing Loading
# =============================================================================
def _preprocess_single_drawing(raw_strokes):
    """Center, scale, truncate, and resample a drawing's strokes to (L, P, 2)."""
    centered = center_and_scale_strokes(raw_strokes)
    truncated = centered[:MAX_CONTEXT_LEN]
    resampled = np.stack(
        [resample_stroke(x, y, num_points=POINTS_PER_STROKE) for x, y in truncated],
        axis=0,
    )  # (L, P, 2)
    return centered, truncated, resampled


def load_drawings_for_category(
    category,
    max_drawings=MAX_DRAWINGS_PER_CATEGORY,
    min_strokes=MIN_STROKES,
    recognized_only=RECOGNIZED_ONLY,
    data_dir=DATA_DIR,
):
    """Load drawings for one category, center on canvas, and resample strokes to (P, 2)."""
    bin_path = data_dir / f"{category}.bin"
    drawings_norm = []

    for drawing in unpack_drawings(bin_path):
        if recognized_only and not drawing["recognized"]:
            continue
        if len(drawing["image"]) < min_strokes:
            continue

        _, _, strokes = _preprocess_single_drawing(drawing["image"])
        drawings_norm.append(strokes)
        if max_drawings is not None and len(drawings_norm) >= max_drawings:
            break

    return drawings_norm


def _pack_drawing_list(drawings):
    """Pack a list of (L_i, P, 2) arrays into flat (sum_L, P, 2) + offsets for .npz storage."""
    lengths = [len(d) for d in drawings]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    strokes = np.concatenate(drawings, axis=0).astype(np.float32)
    return strokes, offsets


# =============================================================================
# Visualization Helper
# =============================================================================
def save_preprocessing_plot(
    categories,
    output_dir=PART2_OUTPUT_DIR,
    data_dir=DATA_DIR,
    min_strokes=MIN_STROKES,
    recognized_only=RECOGNIZED_ONLY,
):
    """Save a step-by-step grid showing raw, centered/truncated, and resampled strokes."""
    samples = []
    for category in categories:
        bin_path = data_dir / f"{category}.bin"
        for drawing in unpack_drawings(bin_path):
            if recognized_only and not drawing["recognized"]:
                continue
            if len(drawing["image"]) < min_strokes:
                continue
            samples.append((category, drawing["image"]))
            break

    if not samples:
        return

    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9.5, 2.7 * n_rows), squeeze=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    col_titles = [
        "1. Raw Sketch\n(canvas [0, 255], variable pts)",
        f"2. Centered, Scaled & Truncated\n(canvas [0, 255], max {MAX_CONTEXT_LEN} strokes)",
        f"3. Resampled & Canonicalized\n(normalized [0, 1], P={POINTS_PER_STROKE} pts, ■=start)",
    ]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=9.5, fontweight="bold", pad=8)

    for row_idx, (category, raw_strokes) in enumerate(samples):
        centered, truncated, resampled = _preprocess_single_drawing(raw_strokes)
        ax_raw, ax_ctr, ax_res = axes[row_idx]

        # 1. Raw strokes
        total_raw_pts = sum(len(x) for x, _ in raw_strokes)
        for s_idx, (x, y) in enumerate(raw_strokes):
            color = colors[s_idx % len(colors)]
            ax_raw.plot(x, y, color=color, marker="o", markersize=2.5, linewidth=1.6, alpha=0.85)
        ax_raw.set_ylabel(category, fontsize=10, fontweight="bold", rotation=90, labelpad=8)
        style_canvas(ax_raw, 255.0, f"{len(raw_strokes)} strokes, {total_raw_pts} raw pts")

        # 2. Centered, scaled & truncated strokes (faintly show any dropped strokes)
        for s_idx, (x, y) in enumerate(centered):
            if s_idx < len(truncated):
                color = colors[s_idx % len(colors)]
                ax_ctr.plot(x, y, color=color, marker="o", markersize=2.5, linewidth=1.6, alpha=0.85)
            else:
                ax_ctr.plot(x, y, color="#cbd5e1", linestyle=":", linewidth=1.2)
        style_canvas(ax_ctr, 255.0, f"kept {len(truncated)}/{len(centered)} strokes")

        # 3. Resampled & canonicalized strokes in [0, 1]
        for s_idx, stroke in enumerate(resampled):
            color = colors[s_idx % len(colors)]
            ax_res.plot(
                stroke[:, 0],
                stroke[:, 1],
                color=color,
                marker="o",
                markersize=3.0,
                linewidth=1.6,
                alpha=0.85,
            )
            ax_res.plot(
                stroke[0, 0],
                stroke[0, 1],
                marker="s",
                markersize=4.5,
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.6,
                zorder=5,
            )
        style_canvas(ax_res, 1.0, f"shape ({len(resampled)}, {POINTS_PER_STROKE}, 2)")

    fig.suptitle(
        "Part 2 Preprocessing Pipeline: Raw -> Centered & Truncated -> Resampled Strokes",
        fontsize=11.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_path = output_dir / "preprocessing_steps.pdf"
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved preprocessing visualization to {save_path}")


# =============================================================================
# Main Preprocessing Pipeline
# =============================================================================
def main():
    set_seed(RANDOM_SEED)
    assert abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    PART2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Categories ({len(CATEGORIES)}): {CATEGORIES}")
    print(
        f"Settings: max_drawings/cat={MAX_DRAWINGS_PER_CATEGORY}, min_strokes={MIN_STROKES}, "
        f"max_context_len={MAX_CONTEXT_LEN}, points/stroke={POINTS_PER_STROKE}"
    )

    all_drawings = []
    all_labels = []
    for label_idx, category in enumerate(CATEGORIES):
        cat_drawings = load_drawings_for_category(category)
        all_drawings.extend(cat_drawings)
        all_labels.extend([label_idx] * len(cat_drawings))
        print(f"  [{label_idx}] {category:<15s} -> {len(cat_drawings)} drawings")

    rng = np.random.default_rng(RANDOM_SEED)
    labels_arr = np.asarray(all_labels)
    train_idx, val_idx, test_idx = [], [], []
    for label in np.unique(labels_arr):
        cls_idx = np.where(labels_arr == label)[0]
        rng.shuffle(cls_idx)
        n_cls = len(cls_idx)
        n_train = int(round(n_cls * TRAIN_RATIO))
        n_val = int(round(n_cls * VAL_RATIO))
        train_idx.append(cls_idx[:n_train])
        val_idx.append(cls_idx[n_train : n_train + n_val])
        test_idx.append(cls_idx[n_train + n_val :])

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    splits = {
        "train": [all_drawings[i] for i in train_idx],
        "val": [all_drawings[i] for i in val_idx],
        "test": [all_drawings[i] for i in test_idx],
    }

    npz_payload = {}
    for split_name, split_drawings in splits.items():
        strokes, offsets = _pack_drawing_list(split_drawings)
        npz_payload[f"{split_name}_strokes"] = strokes
        npz_payload[f"{split_name}_offsets"] = offsets
        print(f"  {split_name:<5s}: {len(split_drawings):5d} drawings ({len(strokes):6d} strokes)")

    np.savez_compressed(PART2_OUTPUT_DIR / "continuous_splits.npz", **npz_payload)

    metadata = {
        "categories": CATEGORIES,
        "max_context_len": MAX_CONTEXT_LEN,
        "points_per_stroke": POINTS_PER_STROKE,
    }
    with open(PART2_OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    save_preprocessing_plot(CATEGORIES)
    print(f"\nSaved continuous stroke splits to '{PART2_OUTPUT_DIR}/continuous_splits.npz'")
    print("Next step: Run `python3 part2_kmeans.py` to learn stroke tokens.")


if __name__ == "__main__":
    main()
