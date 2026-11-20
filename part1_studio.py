"""Part 1 Interactive Classification Studio: Draw strokes and compare all 4 models live."""

import json
import pickle
import random
import struct
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from binary_file_parser import unpack_drawings
from helpers import center_and_scale_strokes, rasterize_drawing
from part1_cnn import SketchCNN
from part1_common import (
    DATA_DIR,
    IMAGE_SIZE,
    PART1_OUTPUT_DIR,
    get_device,
)
from part1_mlp import SketchMLP


# =============================================================================
# Binary Export Helper
# =============================================================================
def pack_drawing(drawing, file_handle):
    """Serialize a drawing dict into the Quick, Draw! binary format."""
    country = drawing.get("country_code", b"XX")
    if isinstance(country, str):
        country = country.encode("ascii", errors="replace")
    country = country[:2].ljust(2, b"X")

    strokes = drawing["image"]
    file_handle.write(
        struct.pack(
            "<Q2sbIH",
            int(drawing.get("key_id", 0)),
            country,
            int(bool(drawing.get("recognized", 1))),
            int(drawing.get("timestamp", time.time())),
            len(strokes),
        )
    )
    for x, y in strokes:
        n_points = len(x)
        fmt = f"{n_points}B"
        file_handle.write(struct.pack("H", n_points))
        file_handle.write(struct.pack(fmt, *(max(0, min(255, int(round(v)))) for v in x)))
        file_handle.write(struct.pack(fmt, *(max(0, min(255, int(round(v)))) for v in y)))


# =============================================================================
# Model Loading & Live Prediction (All 4 Part 1 Models)
# =============================================================================
def load_part1_models(output_dir=PART1_OUTPUT_DIR, device=None):
    """Load metadata and all 4 trained Part 1 classifiers (k-NN, LogReg, MLP, CNN)."""
    if device is None:
        device = get_device()

    meta_path = output_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at '{meta_path}'. Run `part1_preprocess.py` first."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    required = ["knn_model.pkl", "logreg_model.pkl", "mlp_model.pt", "cnn_model.pt"]
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing trained model checkpoints in '{output_dir}': {missing}. "
            "Run part1_knn.py, part1_logreg.py, part1_mlp.py, and part1_cnn.py first."
        )

    with open(output_dir / "knn_model.pkl", "rb") as f:
        knn_model = pickle.load(f)
    knn_model.n_jobs = 1  # Avoid thread dispatch overhead for single-sample live inference

    with open(output_dir / "logreg_model.pkl", "rb") as f:
        logreg_model = pickle.load(f)

    mlp_model = torch.load(output_dir / "mlp_model.pt", map_location=device, weights_only=False).to(device)
    mlp_model.eval()

    cnn_model = torch.load(output_dir / "cnn_model.pt", map_location=device, weights_only=False).to(device)
    cnn_model.eval()

    models = {
        "k-NN": knn_model,
        "LogReg": logreg_model,
        "MLP": mlp_model,
        "CNN": cnn_model,
    }
    return models, metadata, device


@torch.no_grad()
def part1_predict(strokes, models, metadata, device):
    """Predict sketch class and measure inference time for all 4 models after a stroke."""
    categories = metadata["categories"]
    img_size = metadata.get("image_size", IMAGE_SIZE)

    norm_strokes = center_and_scale_strokes(strokes)
    raster = rasterize_drawing({"image": norm_strokes}, size=img_size)
    x_np = (raster.astype(np.float32) / 255.0).reshape(1, -1)
    x_tensor = torch.from_numpy(x_np).to(device)

    results = []
    for name, model in models.items():
        t0 = time.perf_counter()
        if isinstance(model, torch.nn.Module):
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        else:
            probs = model.predict_proba(x_np)[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        pred_idx = int(np.argmax(probs))
        results.append(
            {
                "model": name,
                "label": categories[pred_idx],
                "prob": float(probs[pred_idx]),
                "time_ms": elapsed_ms,
            }
        )

    print(f"\n--- Predictions after {len(strokes)} stroke(s) ---")
    for r in results:
        print(
            f"  {r['model']:<8s} -> {r['label']:<12s} "
            f"(prob={r['prob']:5.1%}, time={r['time_ms']:.2f} ms)"
        )
    return results


# =============================================================================
# Interactive Studio (Canvas + Live Model Comparison Scoreboard)
# =============================================================================
class Part1Studio:
    """Interactive 256x256 canvas showing predictions and latency of all 4 models."""

    def __init__(
        self,
        models,
        metadata,
        device,
        output_path="my_drawing.bin",
        country_code=b"XX",
        min_dist=12,
    ):
        self.models = models
        self.metadata = metadata
        self.device = device
        self.output_path = output_path
        self.country_code = country_code
        self.min_dist_sq = min_dist * min_dist

        self.strokes = []
        self.lines = []
        self._drawing = False
        self._cur_x = []
        self._cur_y = []
        self._cur_line = None

        plt.rcParams["keymap.save"] = []
        self.fig, (self.ax, self.ax_raster, self.ax_pred) = plt.subplots(
            1, 3, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.1, 0.7, 1.0]}
        )
        self.ax.set_xlim(0, 255)
        self.ax.set_ylim(255, 0)
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        img_size = self.metadata.get("image_size", IMAGE_SIZE)
        self._raster_im = self.ax_raster.imshow(
            np.zeros((img_size, img_size), dtype=np.uint8),
            cmap="gray_r",
            vmin=0,
            vmax=255,
        )
        self.ax_raster.set_title(f"Rasterized ({img_size}x{img_size})", fontsize=10)
        self.ax_raster.set_xticks([])
        self.ax_raster.set_yticks([])

        self._update_scoreboard([])
        self._update_title("Draw a stroke to see live predictions from all 4 models!")

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.tight_layout()

    def _update_title(self, status=""):
        cats = ", ".join(self.metadata["categories"])
        base = (
            f"Strokes: {len(self.strokes)} | [u]ndo  [c]lear  [s]ave  [q]uit\n"
            f"Classes: {cats}"
        )
        self.ax.set_title(f"{base}\n{status}" if status else base, fontsize=8.5)
        self.fig.canvas.draw_idle()

    def _update_scoreboard(self, results):
        self.ax_pred.clear()
        self.ax_pred.set_xlim(0, 100)
        self.ax_pred.set_ylim(-0.6, 3.6)
        self.ax_pred.invert_yaxis()
        self.ax_pred.set_xlabel("Prediction (%)")
        self.ax_pred.grid(axis="x", alpha=0.3)

        model_names = list(self.models.keys())
        self.ax_pred.set_yticks(range(len(model_names)))
        self.ax_pred.set_yticklabels(model_names, fontsize=10, fontweight="bold")

        if not results:
            self.ax_pred.set_title("Model Predictions (Draw on canvas)", fontsize=10)
            self.ax_pred.text(
                50, 1.5, "Waiting for strokes...", ha="center", va="center", color="#64748b"
            )
            self.fig.canvas.draw_idle()
            return

        colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]
        probs = [r["prob"] * 100.0 for r in results]
        bars = self.ax_pred.barh(range(len(results)), probs, color=colors, height=0.52, alpha=0.85)

        for idx, (bar, r) in enumerate(zip(bars, results)):
            txt = f" {r['label']} ({r['prob']:.0%}) — {r['time_ms']:.2f} ms"
            x_pos = min(bar.get_width() + 2.0, 52.0)
            self.ax_pred.text(
                x_pos,
                idx,
                txt,
                va="center",
                ha="left",
                fontsize=9.5,
                fontweight="bold",
                color="#0f172a",
            )

        self.ax_pred.set_title(
            f"Predictions & Latency after {len(self.strokes)} Stroke(s)", fontsize=10
        )
        self.fig.canvas.draw_idle()

    def _update_raster(self):
        img_size = self.metadata.get("image_size", IMAGE_SIZE)
        if self.strokes:
            norm_strokes = center_and_scale_strokes(self.strokes)
            raster = rasterize_drawing({"image": norm_strokes}, size=img_size)
            self._raster_im.set_data(raster.reshape(img_size, img_size))
        else:
            self._raster_im.set_data(np.zeros((img_size, img_size), dtype=np.uint8))

    def _refresh_predictions(self, status=""):
        self._update_raster()
        if self.strokes:
            results = part1_predict(self.strokes, self.models, self.metadata, self.device)
            self._update_scoreboard(results)
        else:
            self._update_scoreboard([])
        self._update_title(status)

    def _clamp(self, val):
        return max(0, min(255, int(round(val))))

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        self._drawing = True
        self._cur_x = [self._clamp(event.xdata)]
        self._cur_y = [self._clamp(event.ydata)]
        (self._cur_line,) = self.ax.plot(
            self._cur_x, self._cur_y, color="#1e293b", linewidth=3, marker="o", markersize=3
        )
        self.fig.canvas.draw_idle()

    def _on_move(self, event):
        if not self._drawing or event.inaxes != self.ax:
            return
        x, y = self._clamp(event.xdata), self._clamp(event.ydata)
        dx = x - self._cur_x[-1]
        dy = y - self._cur_y[-1]
        if dx * dx + dy * dy < self.min_dist_sq:
            return
        self._cur_x.append(x)
        self._cur_y.append(y)
        self._cur_line.set_data(self._cur_x, self._cur_y)
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if not self._drawing or event.button != 1:
            return
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            x, y = self._clamp(event.xdata), self._clamp(event.ydata)
            if x != self._cur_x[-1] or y != self._cur_y[-1]:
                self._cur_x.append(x)
                self._cur_y.append(y)
                self._cur_line.set_data(self._cur_x, self._cur_y)
        self._drawing = False
        self.strokes.append((tuple(self._cur_x), tuple(self._cur_y)))
        self.lines.append(self._cur_line)
        self._cur_line = None
        self._refresh_predictions(f"Updated predictions after stroke #{len(self.strokes)}")

    def _on_key(self, event):
        if event.key in ("u", "z"):
            if self.strokes:
                self.strokes.pop()
                self.lines.pop().remove()
                self._refresh_predictions("Undid last stroke")
        elif event.key == "c":
            for line in self.lines:
                line.remove()
            self.strokes.clear()
            self.lines.clear()
            self._refresh_predictions("Cleared canvas")
        elif event.key == "s":
            if not self.strokes:
                self._update_title("Nothing to save!")
                return
            drawing = self.to_drawing()
            with open(self.output_path, "wb") as f:
                pack_drawing(drawing, f)
            self._update_title(f"Saved to {self.output_path}")
            print(f"Saved {len(self.strokes)} strokes to {self.output_path}")
        elif event.key == "q":
            plt.close(self.fig)

    def to_drawing(self):
        return {
            "key_id": random.getrandbits(64),
            "country_code": self.country_code,
            "recognized": 1,
            "timestamp": int(time.time()),
            "image": center_and_scale_strokes(self.strokes),
        }


# =============================================================================
# Headless Demo Fallback (Stroke-by-Stroke Prediction & Timing Visualization)
# =============================================================================
def run_headless_demo(models, metadata, device, max_steps=5):
    """Simulate drawing a sketch stroke-by-stroke and plot all 4 models' predictions."""
    category = "cat" if "cat" in metadata["categories"] else metadata["categories"][0]
    sample_strokes = None
    for d in unpack_drawings(DATA_DIR / f"{category}.bin"):
        if d["recognized"] and len(d["image"]) >= max_steps:
            sample_strokes = d["image"][:max_steps]
            break

    steps = len(sample_strokes)
    fig, axes = plt.subplots(2, steps, figsize=(3.2 * steps, 5.5))

    for s in range(1, steps + 1):
        partial = sample_strokes[:s]
        results = part1_predict(partial, models, metadata, device)

        ax_draw = axes[0, s - 1]
        for idx, (x, y) in enumerate(partial):
            color = "#dc2626" if idx == s - 1 else "#1e293b"
            ax_draw.plot(x, y, color=color, linewidth=2.5)
        ax_draw.set_xlim(0, 255)
        ax_draw.set_ylim(255, 0)
        ax_draw.set_aspect("equal")
        ax_draw.set_xticks([])
        ax_draw.set_yticks([])
        ax_draw.set_title(f"After Stroke {s}/{steps}", fontsize=9.5)

        ax_bar = axes[1, s - 1]
        names = [r["model"] for r in results]
        probs = [r["prob"] * 100.0 for r in results]
        bar_colors = [
            "#16a34a" if r["label"] == category else "#94a3b8" for r in results
        ]
        bars = ax_bar.barh(range(len(results)), probs, color=bar_colors, height=0.55)
        ax_bar.set_xlim(0, 105)
        ax_bar.invert_yaxis()
        ax_bar.set_yticks(range(len(results)))
        ax_bar.set_yticklabels(names if s == 1 else [], fontsize=8.5)
        ax_bar.set_xlabel("Pred (%)", fontsize=8)

        for idx, (bar, r) in enumerate(zip(bars, results)):
            ax_bar.text(
                min(bar.get_width() + 2, 40),
                idx,
                f"{r['label']} ({r['time_ms']:.1f}ms)",
                va="center",
                fontsize=7.5,
                fontweight="bold",
            )

    fig.suptitle(
        f"Part 1 Live Studio Demo — Ground Truth: '{category}' (Green Bar = Correct)",
        fontsize=11,
    )
    fig.tight_layout()
    save_path = PART1_OUTPUT_DIR / "studio_predictions.pdf"
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nSaved stroke-by-stroke prediction demo to {save_path}")


def main():
    models, metadata, device = load_part1_models()

    if matplotlib.get_backend().lower() == "agg":
        run_headless_demo(models, metadata, device)
    else:
        _studio = Part1Studio(models, metadata, device)
        plt.show()


if __name__ == "__main__":
    main()
