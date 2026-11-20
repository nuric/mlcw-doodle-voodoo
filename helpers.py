"""Matplotlib visualization and shared ML utilities for Quick, Draw!"""
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw


def _format_drawing_info(drawing):
    """Format metadata from a binary_file_parser drawing dict into a concise title."""
    country = drawing.get('country_code', b'')
    if isinstance(country, bytes):
        country = country.decode('ascii', errors='replace')
    recognized = bool(drawing.get('recognized', False))
    n_strokes = len(drawing.get('image', ()))
    ts = drawing.get('timestamp')
    ts_str = (
        datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        if ts is not None
        else 'N/A'
    )
    key_id = drawing.get('key_id', 'N/A')
    return (
        f"Strokes: {n_strokes} | Country: {country} | Recognized: {recognized}\n"
        f"key_id: {key_id} | {ts_str}"
    )


_DEFAULT_PLOT_KWARGS = {'linewidth': 3, 'marker': 'o', 'markersize': 3}


def _setup_axes(ax, drawing, title=None):
    """Initialize or configure matplotlib axes for a 256x256 Quick, Draw! canvas."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    ax.set_xlim(0, 255)
    ax.set_ylim(255, 0)
    ax.set_aspect('equal')
    ax.axis('off')

    if title is None:
        title = _format_drawing_info(drawing)
    if title:
        ax.set_title(title, fontsize=10)

    return fig, ax


def plot_drawing(drawing, ax=None, title=None, **plot_kwargs):
    """Plot a drawing dict from binary_file_parser as a static matplotlib figure.

    Returns:
        Tuple of (fig, ax).
    """
    fig, ax = _setup_axes(ax, drawing, title=title)
    kwargs = {**_DEFAULT_PLOT_KWARGS, **plot_kwargs}

    for x, y in drawing['image']:
        ax.plot(x, y, **kwargs)

    return fig, ax


def animate_drawing(drawing, ax=None, interval_ms=40, **plot_kwargs):
    """Animates a Quick, Draw! sketch point-by-point."""
    fig, ax = _setup_axes(ax, drawing, title="")
    kwargs = {**_DEFAULT_PLOT_KWARGS, **plot_kwargs}

    # Pre-create a Line2D artist for each stroke so colors stay distinct
    strokes = drawing["image"]
    lines = [ax.plot([], [], **kwargs)[0] for _ in strokes]

    # Build a flat list of animation frames: (stroke_idx, point_idx)
    frames = []
    for s_idx, (x, _) in enumerate(strokes):
        for p_idx in range(1, len(x) + 1):
            frames.append((s_idx, p_idx))

    # Add a short pause at the end (15 extra frames) so students see the finished sketch
    if frames:
        frames.extend([frames[-1]] * 15)

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        s_idx, p_idx = frame
        # Draw all completed previous strokes in full
        for prev_s in range(s_idx):
            px, py = strokes[prev_s]
            lines[prev_s].set_data(px, py)
        # Draw the active stroke up to point p_idx
        cx, cy = strokes[s_idx]
        lines[s_idx].set_data(cx[:p_idx], cy[:p_idx])
        return lines

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        interval=interval_ms,
        blit=True,
    )
    return fig, anim


def center_and_scale_strokes(raw_strokes, pad_ratio=0.00):
    """Center all strokes in [0, 255] and uniformly scale their bounds with a margin."""
    points_x = [np.asarray(x, dtype=np.float32) for x, _ in raw_strokes if len(x)]
    points_y = [np.asarray(y, dtype=np.float32) for _, y in raw_strokes if len(y)]
    if not points_x or not points_y:
        return raw_strokes

    all_x = np.concatenate(points_x)
    all_y = np.concatenate(points_y)
    min_x, max_x = float(all_x.min()), float(all_x.max())
    min_y, max_y = float(all_y.min()), float(all_y.max())
    span = max(max_x - min_x, max_y - min_y, 1.0)
    scale = 255.0 * (1.0 - 2.0 * pad_ratio) / span
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5

    return [
        (
            (np.asarray(x, dtype=np.float32) - center_x) * scale + 127.5,
            (np.asarray(y, dtype=np.float32) - center_y) * scale + 127.5,
        )
        for x, y in raw_strokes
    ]


def rasterize_drawing(drawing, size=28):
    """Rasterize a drawing dict into a 1D uint8 array (0-255) of length size*size."""
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    scale = (size - 1) / 255.0

    for x, y in drawing["image"]:
        pts = [(xi * scale, yi * scale) for xi, yi in zip(x, y)]
        if len(pts) > 1:
            draw.line(pts, fill=255, width=1)
        elif pts:
            draw.point(pts[0], fill=255)

    return np.asarray(img, dtype=np.uint8).ravel()


# =============================================================================
# Shared Training & Reproducibility Utilities
# =============================================================================
def set_seed(seed=42):
    """Set random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Return the best available PyTorch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_training_history(
    history,
    model_name,
    output_dir,
    loss_ylabel="Loss",
    acc_ylabel="Accuracy",
    acc_scale=1.0,
    acc_label="Acc",
    loss_title=None,
    acc_title=None,
):
    """Save train/validation loss and accuracy curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_title(loss_title or f"{model_name} Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(loss_ylabel)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a * acc_scale for a in history["train_acc"]], label=f"Train {acc_label}")
    ax2.plot(epochs, [a * acc_scale for a in history["val_acc"]], label=f"Val {acc_label}")
    ax2.set_title(acc_title or f"{model_name} Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(acc_ylabel)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = output_dir / f"{model_name.lower().replace(' ', '_')}_history.pdf"
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {save_path}")


if __name__ == '__main__':
    import matplotlib
    from binary_file_parser import unpack_drawing

    sample_file = 'data/cat.bin'
    with open(sample_file, 'rb') as f:
        sample_drawing = unpack_drawing(f)

    fig, _ = plot_drawing(sample_drawing)
    _anim_fig, anim = animate_drawing(sample_drawing)

    scaled_drawing = {
        **sample_drawing,
        'image': center_and_scale_strokes(sample_drawing['image']),
    }

    raster_fig, (raw_ax, scaled_ax) = plt.subplots(1, 2, figsize=(8, 4))
    raw_ax.imshow(rasterize_drawing(sample_drawing).reshape(28, 28), cmap='gray_r')
    raw_ax.set_title('Raw Rasterized (28x28)', fontsize=10)
    raw_ax.axis('off')

    scaled_ax.imshow(rasterize_drawing(scaled_drawing).reshape(28, 28), cmap='gray_r')
    scaled_ax.set_title('Centered & Scaled (28x28)', fontsize=10)
    scaled_ax.axis('off')

    if matplotlib.get_backend().lower() == 'agg':
        fig.savefig('sample_plot.pdf', bbox_inches='tight')
        anim.save('sample_animation.gif', writer='pillow', fps=30)
        raster_fig.savefig('sample_rasterized.pdf', bbox_inches='tight')
        print(
            'Non-interactive backend detected; saved sample_plot.pdf, '
            'sample_animation.gif, and sample_rasterized.pdf'
        )
    else:
        plt.show()
