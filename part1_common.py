"""Shared utilities for Part 1 (Sketch Classification)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

from helpers import get_device, set_seed

# =============================================================================
# Configuration
# =============================================================================
# You do NOT need to edit these values!
DATA_DIR = Path("data")
PART1_OUTPUT_DIR = Path("part1_outputs")
IMAGE_SIZE = 28
RANDOM_SEED = 42


# =============================================================================
# Dataset Loading & DataLoader Helpers
# =============================================================================
def load_part1_splits(output_dir=PART1_OUTPUT_DIR):
    """Load preprocessed train, validation, and test splits for Part 1.

    Args:
        output_dir: Directory containing train.npz, val.npz, test.npz, and metadata.json.

    Returns:
        ((X_train, y_train), (X_val, y_val), (X_test, y_test), metadata)
    """
    output_dir = Path(output_dir)
    if not (output_dir / "train.npz").exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found in '{output_dir}'. "
            "Run `python3 part1_preprocess.py` first."
        )

    train_data = np.load(output_dir / "train.npz")
    val_data = np.load(output_dir / "val.npz")
    test_data = np.load(output_dir / "test.npz")

    X_train, y_train = train_data["X"].astype(np.float32), train_data["y"].astype(np.int64)
    X_val, y_val = val_data["X"].astype(np.float32), val_data["y"].astype(np.int64)
    X_test, y_test = test_data["X"].astype(np.float32), test_data["y"].astype(np.int64)

    with open(output_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), metadata


def make_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    """Wrap train, validation, and test NumPy arrays into PyTorch DataLoaders."""
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=batch_size,
    )
    return train_loader, val_loader, test_loader


# =============================================================================
# Evaluation & Visualization
# =============================================================================
def evaluate_classifier(
    y_true,
    y_pred,
    categories,
    model_name,
    output_dir=PART1_OUTPUT_DIR,
):
    """Print classification metrics and save a confusion matrix plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n=== {model_name} Test Evaluation ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=categories, digits=4))

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=categories,
        cmap="Blues",
        xticks_rotation=45,
        ax=ax,
    )
    ax.set_title(f"{model_name} Confusion Matrix (Acc: {acc:.3f})")
    fig.tight_layout()

    save_path = output_dir / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.pdf"
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {save_path}")
    return {"accuracy": float(acc), "macro_f1": float(macro_f1)}


# =============================================================================
# PyTorch Training & Evaluation Helpers
# =============================================================================
def _run_epoch(model, loader, criterion, optimizer, device):
    """Run one training or evaluation epoch and return (avg_loss, accuracy)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            if is_train:
                optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def train_pytorch_classifier(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    weight_decay,
    model_name,
    device,
):
    """Train a PyTorch classifier and return training/validation history."""
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\nTraining {model_name} on {device} for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, None, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"  Epoch {epoch:2d}/{epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )
    return history


@torch.no_grad()
def predict_pytorch_classifier(model, loader, device):
    """Return predicted class indices for all samples in a DataLoader."""
    model.eval()
    preds = []
    for X_batch, _ in loader:
        logits = model(X_batch.to(device))
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)
