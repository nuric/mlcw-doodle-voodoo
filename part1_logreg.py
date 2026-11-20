"""Part 1 Model 2: Logistic Regression Sketch Classifier."""

import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from part1_common import (
    IMAGE_SIZE,
    PART1_OUTPUT_DIR,
    RANDOM_SEED,
    evaluate_classifier,
    load_part1_splits,
    set_seed,
)

# =============================================================================
# Hyperparameters
# =============================================================================
pass


# =============================================================================
# Weight Visualization
# =============================================================================
def save_weight_maps(model, categories, output_dir=PART1_OUTPUT_DIR):
    """Visualize learned linear weights reshaped as 28x28 images per class."""
    weights = model.coef_  # shape: (num_classes, 784)
    n_cats = len(categories)
    cols = min(5, n_cats)
    rows = (n_cats + cols - 1) // cols

    vmax = np.abs(weights).max()
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, cat in enumerate(categories):
        ax = axes[idx]
        w_img = weights[idx].reshape(IMAGE_SIZE, IMAGE_SIZE)
        ax.imshow(w_img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(cat, fontsize=9)
        ax.axis("off")

    for idx in range(n_cats, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Logistic Regression Learned Class Weights", fontsize=11)
    fig.tight_layout()
    save_path = output_dir / "logreg_weights.pdf"
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved weight maps to {save_path}")


# =============================================================================
# Training & Validation Tuning
# =============================================================================
def main():
    set_seed(RANDOM_SEED)

    # Load preprocessed [0, 1] features
    (X_train, y_train), (X_val, y_val), (X_test, y_test), metadata = load_part1_splits()
    categories = metadata["categories"]
    print(f"Loaded dataset: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    logreg_model = None
    # Use the already imported LogisticRegression to train logreg_model. We have also imported accuracy_score to help.
    pass

    # Save the model
    checkpoint_path = PART1_OUTPUT_DIR / "logreg_model.pkl"
    with open(checkpoint_path, "wb") as f:
        pickle.dump(logreg_model, f)
    print(f"Saved LogReg model to {checkpoint_path}")
    save_weight_maps(logreg_model, categories)

    # Evaluate the model on the test set
    y_test_pred = logreg_model.predict(X_test)
    evaluate_classifier(y_test, y_test_pred, categories, model_name="LogReg")


if __name__ == "__main__":
    main()
