"""Shared utilities for Part 2 (K-Means Stroke Tokenization & Autoregressive RNN)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import pairwise_distances_argmin
from torch.utils.data import DataLoader, TensorDataset

from helpers import get_device, set_seed

# =============================================================================
# Configuration & Token Constants
# =============================================================================
# You do NOT need to edit these values!
DATA_DIR = Path("data")
PART2_OUTPUT_DIR = Path("part2_outputs")
RANDOM_SEED = 42

MAX_CONTEXT_LEN = 12      # Maximum number of strokes in an autoregressive sequence
POINTS_PER_STROKE = 16    # P: fixed number of resampled (x, y) points per stroke

PAD_TOKEN = 0             # Padding token ID (ignored in CrossEntropyLoss)
# Stroke cluster IDs are 1 .. K
# EOS_TOKEN is K + 1


def get_vocab_info(num_clusters):
    """Return special token IDs and total vocabulary size for K clusters."""
    eos_token = num_clusters + 1
    vocab_size = num_clusters + 2  # PAD (0) + K clusters (1..K) + EOS (K+1)
    return PAD_TOKEN, eos_token, vocab_size


# =============================================================================
# Stroke Resampling & Canonicalization
# =============================================================================
def _canonicalize_stroke_direction(pts):
    """Orient stroke points consistently so K-Means averages do not cancel out.

    - Closed loops: start at the top-most point and wind clockwise.
    - Open curves: start at the top-left-most endpoint.
    """
    diffs = np.diff(pts, axis=0)
    arc_len = float(np.sqrt((diffs ** 2).sum(axis=1)).sum())
    end_dist = float(np.linalg.norm(pts[0] - pts[-1]))

    if arc_len > 1e-4 and end_dist < 0.25 * arc_len:
        # Closed loop: shift start to top-most point
        top_idx = int(np.argmin(pts[:, 1]))
        pts = np.roll(pts, -top_idx, axis=0)
        # Ensure clockwise winding in screen coordinates (y down)
        x, y = pts[:, 0], pts[:, 1]
        signed_area = 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
        if signed_area < 0.0:
            pts = pts[::-1].copy()
    else:
        # Open stroke: orient from top-left to bottom-right
        if (pts[0, 0] + pts[0, 1]) > (pts[-1, 0] + pts[-1, 1]):
            pts = pts[::-1].copy()

    return pts


def resample_stroke(x, y, num_points=POINTS_PER_STROKE):
    """Resample a raw (x, y) stroke to `num_points` evenly spaced points in [0, 1]."""
    pts = np.stack([np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)], axis=1)
    pts = np.clip(pts / 255.0, 0.0, 1.0)

    if len(pts) == 0:
        return np.zeros((num_points, 2), dtype=np.float32)
    if len(pts) == 1:
        return np.repeat(pts, num_points, axis=0)

    diffs = np.diff(pts, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]

    if total_len < 1e-6:
        return np.repeat(pts[:1], num_points, axis=0)

    target_dists = np.linspace(0.0, total_len, num_points, dtype=np.float32)
    resampled_x = np.interp(target_dists, cum_len, pts[:, 0])
    resampled_y = np.interp(target_dists, cum_len, pts[:, 1])
    resampled = np.stack([resampled_x, resampled_y], axis=1).astype(np.float32)
    return _canonicalize_stroke_direction(resampled)


def stroke_norm_to_raw(stroke_norm):
    """Convert a normalized (P, 2) stroke in [0, 1] to integer (x_tuple, y_tuple) in [0, 255]."""
    coords = np.clip(np.round(stroke_norm * 255.0), 0, 255).astype(int)
    return tuple(coords[:, 0].tolist()), tuple(coords[:, 1].tolist())


# =============================================================================
# Token Encoding & Decoding Helpers
# =============================================================================
def encode_strokes_to_tokens(strokes_norm, centroids):
    """Map an array of normalized strokes (N, P, 2) to token IDs in 1..K."""
    flat_strokes = np.asarray(strokes_norm, dtype=np.float32).reshape(len(strokes_norm), -1)
    flat_centroids = np.asarray(centroids, dtype=np.float32).reshape(len(centroids), -1)
    cluster_idx = pairwise_distances_argmin(flat_strokes, flat_centroids)
    return (cluster_idx + 1).astype(np.int64)  # +1 because 0 is PAD_TOKEN


def style_canvas(ax, limit=255.0, subtitle=None):
    """Configure a square QuickDraw canvas axis with subtle bounds and an optional caption."""
    ax.set_xlim(0, limit)
    ax.set_ylim(limit, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(0.8)
    if subtitle is not None:
        ax.set_xlabel(subtitle, fontsize=8, color="#334155", labelpad=4)


def decode_token_to_stroke(token_id, centroids):
    """Convert a stroke token ID (1..K) back to its (P, 2) centroid in [0, 1]."""
    cluster_idx = int(token_id) - 1
    if cluster_idx < 0 or cluster_idx >= len(centroids):
        raise ValueError(f"Token ID {token_id} is not a valid stroke cluster (1..{len(centroids)}).")
    return centroids[cluster_idx]


# =============================================================================
# Dataset Loading Helpers
# =============================================================================
def load_continuous_splits(output_dir=PART2_OUTPUT_DIR):
    """Load preprocessed continuous stroke lists for K-Means clustering."""
    output_dir = Path(output_dir)
    npz_path = output_dir / "continuous_splits.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Continuous splits not found at '{npz_path}'. Run `python3 part2_preprocess.py` first."
        )

    data = np.load(npz_path)
    splits = {}
    for split in ("train", "val", "test"):
        strokes_all = data[f"{split}_strokes"]
        offsets = data[f"{split}_offsets"]
        splits[split] = [
            strokes_all[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)
        ]

    with open(output_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return splits["train"], splits["val"], splits["test"], metadata


def load_tokenized_splits(output_dir=PART2_OUTPUT_DIR):
    """Load K-Means tokenized splits (x, y), centroids, and metadata for RNN training."""
    output_dir = Path(output_dir)
    tok_path = output_dir / "tokenized_splits.npz"
    cent_path = output_dir / "centroids.npy"
    if not tok_path.exists() or not cent_path.exists():
        raise FileNotFoundError(
            f"Tokenized data not found in '{output_dir}'. "
            "Run `python3 part2_preprocess.py` and `python3 part2_kmeans.py` first."
        )

    data = np.load(tok_path)
    splits = {}
    for name in ("train", "val", "test"):
        splits[name] = {
            "x": data[f"{name}_x"].astype(np.int64),
            "y": data[f"{name}_y"].astype(np.int64),
        }

    centroids = np.load(cent_path).astype(np.float32)
    with open(output_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return splits["train"], splits["val"], splits["test"], centroids, metadata


def make_dataloaders(train_split, val_split, test_split, batch_size):
    """Wrap train, validation, and test token split dicts into PyTorch DataLoaders."""
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_split["x"]), torch.from_numpy(train_split["y"])),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_split["x"]), torch.from_numpy(val_split["y"])),
        batch_size=batch_size,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_split["x"]), torch.from_numpy(test_split["y"])),
        batch_size=batch_size,
    )
    return train_loader, val_loader, test_loader


# =============================================================================
# Autoregressive Token Training & Evaluation Helpers
# =============================================================================
def _run_token_epoch(model, loader, criterion, optimizer, device):
    """Run one training or evaluation epoch for next-token classification."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.set_grad_enabled(is_train):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(x_batch)  # (B, T, V)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y_batch.reshape(-1))

            if is_train:
                loss.backward()
                optimizer.step()

            valid_mask = y_batch != PAD_TOKEN
            n_valid = int(valid_mask.sum().item())
            total_loss += loss.item() * n_valid
            preds = logits.argmax(dim=-1)
            total_correct += int((preds[valid_mask] == y_batch[valid_mask]).sum().item())
            total_tokens += n_valid

    avg_loss = total_loss / max(total_tokens, 1)
    acc = total_correct / max(total_tokens, 1)
    return avg_loss, acc


def train_token_predictor(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    weight_decay,
    model_name,
    device,
):
    """Train an autoregressive token-level RNN and return training/validation history."""
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\nTraining {model_name} on {device} for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_token_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = _run_token_epoch(model, val_loader, criterion, None, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"  Epoch {epoch:2d}/{epochs} | "
            f"train_loss={train_loss:.4f} (acc={train_acc:.1%}) | "
            f"val_loss={val_loss:.4f} (acc={val_acc:.1%})"
        )

    return history


@torch.no_grad()
def evaluate_token_predictor(
    model,
    test_loader,
    test_split,
    centroids,
    eos_token,
    device,
    model_name,
    output_dir=PART2_OUTPUT_DIR,
):
    """Evaluate token-level accuracy on test split and visualize decoded stroke predictions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    all_logits = []
    for x_batch, _ in test_loader:
        all_logits.append(model(x_batch.to(device)).cpu())
    logits = torch.cat(all_logits, dim=0)  # (N, T, V)

    x_test = test_split["x"]
    y_true = torch.from_numpy(test_split["y"])
    valid_mask = y_true != PAD_TOKEN

    test_loss = float(criterion(logits.reshape(-1, logits.size(-1)), y_true.reshape(-1)).item())
    valid_logits = logits[valid_mask]
    valid_targets = y_true[valid_mask]

    top1_acc = float((valid_logits.argmax(dim=-1) == valid_targets).float().mean().item())
    top5_preds = valid_logits.topk(k=min(5, valid_logits.size(-1)), dim=-1).indices
    top5_acc = float((top5_preds == valid_targets.unsqueeze(-1)).any(dim=-1).float().mean().item())

    print(f"\n=== {model_name} Test Evaluation ===")
    print(f"Test Cross-Entropy Loss : {test_loss:.4f} (Perplexity: {np.exp(test_loss):.2f})")
    print(f"Test Top-1 Accuracy     : {top1_acc:.2%}")
    print(f"Test Top-5 Accuracy     : {top5_acc:.2%}")

    # Plot 8 test drawings predicting a stroke step before EOS
    n_show = min(8, len(x_test))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.ravel()):
        if i >= n_show:
            ax.axis("off")
            continue

        # Find steps where the ground-truth target is an actual stroke (not PAD and not EOS)
        stroke_steps = np.where((test_split["y"][i] != PAD_TOKEN) & (test_split["y"][i] != eos_token))[0]
        t = int(stroke_steps[-1]) if len(stroke_steps) > 0 else 0

        for s_idx in range(t + 1):
            tok = int(x_test[i, s_idx])
            if 1 <= tok <= len(centroids):
                stroke = centroids[tok - 1]
                ax.plot(
                    stroke[:, 0] * 255.0,
                    stroke[:, 1] * 255.0,
                    color="#64748b",
                    linewidth=2,
                    alpha=0.7,
                    label="Context Strokes" if (i == 0 and s_idx == 0) else None,
                )

        gt_tok = int(test_split["y"][i, t])
        step_logits = logits[i, t].clone()
        step_logits[PAD_TOKEN] = -1e9
        step_logits[eos_token] = -1e9
        pred_tok = int(step_logits.argmax().item())

        if 1 <= gt_tok <= len(centroids):
            gt = centroids[gt_tok - 1]
            ax.plot(gt[:, 0] * 255.0, gt[:, 1] * 255.0, color="#16a34a", linewidth=2.5, label="True Next Stroke" if i == 0 else None)
        if 1 <= pred_tok <= len(centroids):
            pr = centroids[pred_tok - 1]
            ax.plot(pr[:, 0] * 255.0, pr[:, 1] * 255.0, color="#dc2626", linewidth=2.5, linestyle="--", label="Predicted Next Stroke" if i == 0 else None)

        style_canvas(ax)

        match_note = f"exact match (#{gt_tok})" if pred_tok == gt_tok else f"true #{gt_tok}, pred #{pred_tok}"
        ax.set_title(f"Drawing {i + 1} (given {t + 1} strokes)\n{match_note}", fontsize=8.5)

    fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.94), frameon=False, fontsize=9)
    fig.suptitle(f"{model_name}: Next-Stroke Prediction on Test Drawings", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_path = output_dir / f"{model_name.lower().replace(' ', '_')}_predictions.pdf"
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved prediction grid to {save_path}")

    return {"loss": test_loss, "top1_acc": top1_acc, "top5_acc": top5_acc}
