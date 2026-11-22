"""Part 1 Model 3: PyTorch Multi-Layer Perceptron (MLP) Sketch Classifier."""

import torch
import torch.nn as nn

from helpers import plot_training_history
from part1_common import (
    PART1_OUTPUT_DIR,
    RANDOM_SEED,
    evaluate_classifier,
    get_device,
    load_part1_splits,
    make_dataloaders,
    predict_pytorch_classifier,
    set_seed,
    train_pytorch_classifier,
)

# =============================================================================
# Hyperparameters
# =============================================================================
# We give these to speed up common ones to get the training plumbing working.
BATCH_SIZE = 128
# Add your hyperparameters here as usual
# ### BEGIN SOLUTION ###
pass
# ### END SOLUTION ###


# =============================================================================
# Model Definition
# =============================================================================
class SketchMLP(nn.Module):
    """MLP for flattened 28x28 sketch classification."""

    # Define your model def __init__ here.
    # ### BEGIN SOLUTION ###
    pass
    # ### END SOLUTION ###

    def forward(self, x):
        return self.net(x)

# =============================================================================
# Training & Evaluation
# =============================================================================
def main():
    set_seed(RANDOM_SEED)
    device = get_device()

    (X_train, y_train), (X_val, y_val), (X_test, y_test), metadata = load_part1_splits()
    categories = metadata["categories"]

    train_loader, val_loader, test_loader = make_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=BATCH_SIZE
    )
    input_dim = X_train.shape[1]  # 28*28=784
    num_classes = len(categories)

    mlp_model = None
    # Use the already defined SketchMLP class to train the mlp_model. We have also imported train_pytorch_classifier to train the model and plot_training_history to visualise the training process.
    # ### BEGIN SOLUTION ###
    pass
    # ### END SOLUTION ###

    # Save the model
    checkpoint_path = PART1_OUTPUT_DIR / "mlp_model.pt"
    torch.save(mlp_model, checkpoint_path)
    print(f"Saved MLP model to {checkpoint_path}")

    y_test_pred = predict_pytorch_classifier(mlp_model, test_loader, device)
    evaluate_classifier(y_test, y_test_pred, categories, model_name="MLP")


if __name__ == "__main__":
    main()
