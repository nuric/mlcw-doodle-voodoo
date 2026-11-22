"""Part 2 Step 2 — K-Means Stroke Tokenizer: Cluster strokes into discrete tokens."""

import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin

from part2_common import (
    MAX_CONTEXT_LEN,
    PAD_TOKEN,
    PART2_OUTPUT_DIR,
    RANDOM_SEED,
    encode_strokes_to_tokens,
    get_vocab_info,
    load_continuous_splits,
    set_seed,
    style_canvas,
)

# =============================================================================
# K-Means Configuration
# =============================================================================
# These should be sufficient, so you don't have to play around too much.
KMEANS_BATCH_SIZE = 2048
KMEANS_MAX_ITER = 3000

# Please experiment as usual by checking your own hyperparameters
# ### BEGIN SOLUTION ###
pass
# ### END SOLUTION ###


# =============================================================================
# Clustering & Tokenization Functions
# =============================================================================
def fit_stroke_kmeans(train_drawings, num_clusters, seed=RANDOM_SEED):
    """Fit MiniBatchKMeans on all training strokes and return (medoids, inertia)."""
    all_train_strokes = np.concatenate(train_drawings, axis=0)  # (M, P, 2)
    n_strokes, n_pts, _ = all_train_strokes.shape
    flat_strokes = all_train_strokes.reshape(n_strokes, n_pts * 2)

    print(f"Fitting K-Means (K={num_clusters}) on {n_strokes} training strokes...")
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        batch_size=KMEANS_BATCH_SIZE,
        max_iter=KMEANS_MAX_ITER,
        random_state=seed,
    )
    kmeans.fit(flat_strokes)

    # Snap each cluster center to its nearest real training stroke (medoid)
    medoid_idx = pairwise_distances_argmin(kmeans.cluster_centers_, flat_strokes)
    medoids = all_train_strokes[medoid_idx].astype(np.float32)
    return medoids, float(kmeans.inertia_)


def tokenize_drawings(drawings, centroids, max_len=MAX_CONTEXT_LEN):
    """Convert variable-length drawings into padded (x, y) token ID matrices.

    For a drawing with stroke tokens [t_0, t_1, ..., t_{L-1}]:
      - x[i, :L] = [t_0, t_1, ..., t_{L-1}]
      - y[i, :L] = [t_1, ..., t_{L-1}, EOS_TOKEN]
      - Padded positions are filled with PAD_TOKEN (0).
    """
    _, eos_token, _ = get_vocab_info(len(centroids))
    n = len(drawings)
    x = np.full((n, max_len), PAD_TOKEN, dtype=np.int64)
    y = np.full((n, max_len), PAD_TOKEN, dtype=np.int64)

    for i, strokes in enumerate(drawings):
        tokens = encode_strokes_to_tokens(strokes[:max_len], centroids)
        seq_len = len(tokens)
        x[i, :seq_len] = tokens
        if seq_len > 1:
            y[i, : seq_len - 1] = tokens[1:]
        y[i, seq_len - 1] = eos_token

    return x, y


def plot_kmeans_vocabulary(centroids, output_dir=PART2_OUTPUT_DIR, max_display=16):
    """Visualize learned K-Means stroke tokens on 256x256 mini-canvases."""
    n_show = min(len(centroids), max_display)
    cols = 8
    rows = (n_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 1.5 * rows))
    for i, ax in enumerate(axes.ravel()):
        if i >= n_show:
            ax.axis("off")
            continue

        stroke = centroids[i] * 255.0
        ax.plot(stroke[:, 0], stroke[:, 1], color="#2563eb", linewidth=2.2)
        ax.scatter(stroke[0, 0], stroke[0, 1], color="#16a34a", s=14, zorder=3)
        style_canvas(ax)
        ax.set_title(f"Token {i + 1}", fontsize=8)

    fig.suptitle(
        f"K-Means Stroke Vocabulary (First {n_show}/{len(centroids)} Tokens, Green Dot = Start)",
        fontsize=11,
    )
    fig.tight_layout()
    save_path = output_dir / "kmeans_tokens.pdf"
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved K-Means token vocabulary plot to {save_path}")


# =============================================================================
# Main K-Means Pipeline
# =============================================================================
def main():
    set_seed(RANDOM_SEED)
    train_drawings, val_drawings, test_drawings, metadata = load_continuous_splits()
    max_len = metadata["max_context_len"]

    centroids = None
    # Use `fit_stroke_kmeans` to perform k-means clustering and assign `centroids`.
    # ### BEGIN SOLUTION ###
    pass
    # ### END SOLUTION ###

    num_clusters = len(centroids)
    np.save(PART2_OUTPUT_DIR / "centroids.npy", centroids)
    plot_kmeans_vocabulary(centroids)

    train_x, train_y = tokenize_drawings(train_drawings, centroids, max_len=max_len)
    val_x, val_y = tokenize_drawings(val_drawings, centroids, max_len=max_len)
    test_x, test_y = tokenize_drawings(test_drawings, centroids, max_len=max_len)

    # Compute reconstruction error (quantization RMSE in pixels) on validation strokes
    val_strokes = np.concatenate(val_drawings, axis=0)
    val_tokens = encode_strokes_to_tokens(val_strokes, centroids)
    recon_strokes = centroids[val_tokens - 1]
    quant_l2_px = float(
        np.mean(np.sqrt(((val_strokes - recon_strokes) ** 2).sum(axis=-1))) * 255.0
    )
    print(f"Validation K-Means Quantization Error: {quant_l2_px:.2f} px mean point L2")

    np.savez_compressed(
        PART2_OUTPUT_DIR / "tokenized_splits.npz",
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
    )

    pad_token, eos_token, vocab_size = get_vocab_info(num_clusters)
    metadata.update(
        {
            "num_clusters": num_clusters,
            "pad_token": pad_token,
            "eos_token": eos_token,
            "vocab_size": vocab_size,
            "quantization_l2_px": round(quant_l2_px, 2),
        }
    )
    with open(PART2_OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"Saved tokenized splits (vocab_size={vocab_size}: PAD={pad_token}, "
        f"Strokes=1..{num_clusters}, EOS={eos_token}) to '{PART2_OUTPUT_DIR}/tokenized_splits.npz'"
    )
    print("Next step: Run `python3 part2_rnn.py` to train the autoregressive RNN.")


if __name__ == "__main__":
    main()
