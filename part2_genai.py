"""Part 2 Step 4 — Interactive Tab-Completion Studio using K-Means Stroke Tokens."""

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from binary_file_parser import unpack_drawings
from helpers import center_and_scale_strokes
from part2_common import (
    DATA_DIR,
    PAD_TOKEN,
    PART2_OUTPUT_DIR,
    decode_token_to_stroke,
    encode_strokes_to_tokens,
    get_device,
    resample_stroke,
    stroke_norm_to_raw,
    style_canvas,
)
from part2_rnn import NextStrokeRNN  # Needed for torch.load unpickling

# =============================================================================
# Configuration
# =============================================================================
MAX_TOTAL_STROKES = 12
TEMPERATURE = 0.5         # 0.0 for greedy argmax; >0 for top-k temperature sampling
TOP_K = 5


# =============================================================================
# Model Loading & Single-Step Token Prediction
# =============================================================================
def load_rnn_generator(output_dir=PART2_OUTPUT_DIR, device=None):
    """Load metadata, K-Means centroids, and the trained Part 2 RNN checkpoint."""
    if device is None:
        device = get_device()

    meta_path = output_dir / "metadata.json"
    cent_path = output_dir / "centroids.npy"
    rnn_ckpt = output_dir / "rnn_model.pt"

    if not meta_path.exists() or not cent_path.exists():
        raise FileNotFoundError(
            f"Missing files in '{output_dir}'. "
            "Run `python3 part2_preprocess.py` and `python3 part2_kmeans.py` first."
        )
    if not rnn_ckpt.exists():
        raise FileNotFoundError(
            f"RNN checkpoint not found at '{rnn_ckpt}'. Run `python3 part2_rnn.py` first."
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    centroids = np.load(cent_path).astype(np.float32)
    model = torch.load(rnn_ckpt, map_location=device, weights_only=False).to(device)
    model.eval()
    return model, centroids, metadata, device


@torch.no_grad()
def predict_next_stroke(
    strokes_raw,
    model,
    centroids,
    metadata,
    device,
    temperature=TEMPERATURE,
):
    """Predict the next stroke (x_tuple, y_tuple) and whether EOS was reached."""
    max_len = metadata["max_context_len"]
    n_pts = metadata["points_per_stroke"]
    eos_token = metadata["eos_token"]

    recent = strokes_raw[-max_len:]
    strokes_norm = np.stack(
        [resample_stroke(x, y, num_points=n_pts) for x, y in recent], axis=0
    )
    token_ids = encode_strokes_to_tokens(strokes_norm, centroids)  # (k,)

    inp = torch.from_numpy(token_ids).long().unsqueeze(0).to(device)  # (1, k)
    logits = model(inp)[0, -1].clone()  # (vocab_size,)

    # Mask out PAD and already-used stroke tokens so we do not repeat strokes
    logits[PAD_TOKEN] = -1e9
    for used_tok in token_ids:
        logits[int(used_tok)] = -1e9

    is_eos = int(logits.argmax().item()) == eos_token
    logits[eos_token] = -1e9  # Mask EOS so we can still decode the best stroke if requested

    if temperature <= 0.0:
        next_token = int(logits.argmax().item())
    else:
        top_vals, top_idx = torch.topk(logits, k=min(TOP_K, len(centroids)))
        probs = torch.softmax(top_vals / temperature, dim=-1)
        choice = int(torch.multinomial(probs, num_samples=1).item())
        next_token = int(top_idx[choice].item())

    pred_norm = decode_token_to_stroke(next_token, centroids)
    return stroke_norm_to_raw(pred_norm), next_token, is_eos


# =============================================================================
# Simplified Interactive Tab-Completion Studio
# =============================================================================
class NextStrokeStudio:
    """Interactive 256x256 canvas: draw strokes with mouse and press [Tab] to complete."""

    def __init__(self, model, centroids, metadata, device, min_dist=8):
        self.model = model
        self.centroids = centroids
        self.metadata = metadata
        self.device = device
        self.min_dist_sq = min_dist * min_dist

        self.strokes = []
        self.lines = []
        self._drawing = False
        self._cur_x = []
        self._cur_y = []
        self._cur_line = None

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        style_canvas(self.ax)

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._update_title("Draw 1-2 strokes, then press [Tab] to autocomplete!")

    def _update_title(self, status=""):
        cats = ", ".join(self.metadata["categories"])
        header = (
            f"Categories: {cats} | Strokes: {len(self.strokes)}/{MAX_TOTAL_STROKES}\n"
            f"[Tab/n] next stroke   [u]ndo   [c]lear   [q]uit"
        )
        self.ax.set_title(f"{header}\n{status}" if status else header, fontsize=9)
        self.fig.canvas.draw_idle()

    def _clamp(self, val):
        return max(0, min(255, int(round(val))))

    def _autocomplete_stroke(self):
        if not self.strokes:
            self._update_title("Draw at least 1 stroke first!")
            return
        if len(self.strokes) >= MAX_TOTAL_STROKES:
            self._update_title(f"Reached max {MAX_TOTAL_STROKES} strokes.")
            return

        (x, y), token_id, is_eos = predict_next_stroke(
            self.strokes, self.model, self.centroids, self.metadata, self.device
        )
        (line,) = self.ax.plot(x, y, color="#dc2626", linewidth=2.8, linestyle="--")
        self.strokes.append((x, y))
        self.lines.append(line)

        eos_note = " (Model signals <EOS>: doodle complete!)" if is_eos else ""
        self._update_title(f"Predicted Token #{token_id}{eos_note}")

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        self._drawing = True
        self._cur_x = [self._clamp(event.xdata)]
        self._cur_y = [self._clamp(event.ydata)]
        (self._cur_line,) = self.ax.plot(self._cur_x, self._cur_y, color="#1e293b", linewidth=2.8)
        self.fig.canvas.draw_idle()

    def _on_move(self, event):
        if not self._drawing or event.inaxes != self.ax:
            return
        x, y = self._clamp(event.xdata), self._clamp(event.ydata)
        if (x - self._cur_x[-1]) ** 2 + (y - self._cur_y[-1]) ** 2 < self.min_dist_sq:
            return
        self._cur_x.append(x)
        self._cur_y.append(y)
        self._cur_line.set_data(self._cur_x, self._cur_y)
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if not self._drawing or event.button != 1:
            return
        self._drawing = False
        self.strokes.append((tuple(self._cur_x), tuple(self._cur_y)))
        self.lines.append(self._cur_line)
        self._cur_line = None
        self._update_title("Press [Tab] to predict the next stroke")

    def _on_key(self, event):
        if event.key in ("tab", "n"):
            self._autocomplete_stroke()
        elif event.key in ("u", "z") and self.strokes:
            self.strokes.pop()
            self.lines.pop().remove()
            self._update_title("Undid last stroke")
        elif event.key == "c":
            for line in self.lines:
                line.remove()
            self.strokes.clear()
            self.lines.clear()
            self._update_title("Cleared canvas")
        elif event.key == "q":
            plt.close(self.fig)


# =============================================================================
# Headless Demo Fallback (Saves Single-Row Autoregressive Rollout)
# =============================================================================
def run_headless_demo(model, centroids, metadata, device, seed_strokes=2, max_gen_strokes=6):
    """Autoregressively complete one sample drawing across a single row and save figure."""
    category = metadata["categories"][0]
    sample_strokes = None
    for d in unpack_drawings(DATA_DIR / f"{category}.bin"):
        if d["recognized"] and len(d["image"]) >= seed_strokes + 2:
            sample_strokes = center_and_scale_strokes(d["image"])
            break

    seed = list(sample_strokes[:seed_strokes])
    generated = list(seed)
    for _ in range(max_gen_strokes):
        next_stroke, _, is_eos = predict_next_stroke(
            generated, model, centroids, metadata, device, temperature=0.0
        )
        generated.append(next_stroke)
        if is_eos:
            break

    n_gen = len(generated) - seed_strokes
    panels = [
        (seed, f"{category}: Seed ({seed_strokes} strokes)"),
        (generated, f"{category}: RNN Tab-Complete (+{n_gen})"),
        (sample_strokes, f"{category}: Ground Truth ({len(sample_strokes)} strokes)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5), layout="constrained")
    for col_idx, (strokes, title) in enumerate(panels):
        ax = axes[col_idx]
        for s_idx, (x, y) in enumerate(strokes):
            if col_idx == 1 and s_idx >= seed_strokes:
                ax.plot(x, y, color="#dc2626", linestyle="--", linewidth=2.2)
            elif col_idx == 2:
                ax.plot(x, y, color="#16a34a", linestyle="-", linewidth=2.0)
            else:
                ax.plot(x, y, color="#1e293b", linestyle="-", linewidth=2.2)
        style_canvas(ax)
        ax.set_title(title, fontsize=8.5, pad=4)

    fig.suptitle(
        "Part 2 K-Means + RNN Autoregressive Completion (Black=Seed, Red=Generated)",
        fontsize=9.5,
    )
    save_path = PART2_OUTPUT_DIR / "sample_generation.pdf"
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.08, dpi=150)
    plt.close(fig)
    print(f"Saved autoregressive tab-completion demo to {save_path}")


def main():
    model, centroids, metadata, device = load_rnn_generator()

    if matplotlib.get_backend().lower() == "agg":
        run_headless_demo(model, centroids, metadata, device)
    else:
        _studio = NextStrokeStudio(model, centroids, metadata, device)
        plt.show()


if __name__ == "__main__":
    main()
