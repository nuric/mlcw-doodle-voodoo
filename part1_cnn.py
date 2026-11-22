"""Part 1 Model 4: PyTorch Convolutional Neural Network (CNN) Sketch Classifier."""

import torch
import torch.nn as nn

from helpers import plot_training_history
from part1_common import (
    IMAGE_SIZE,
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
class SketchCNN(nn.Module):
    """2D CNN for 28x28 grayscale sketch classification."""

    # Define your model def __init__ here. This time also define the forward method to implement the forward pass of the CNN.
    # ### BEGIN SOLUTION ###
    pass
    # ### END SOLUTION ###



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
    num_classes = len(categories)

    cnn_model = None
    # Use the already defined SketchCNN class to train the cnn_model. We have also imported train_pytorch_classifier to train the model and plot_training_history to visualise the training process.
    # ### BEGIN SOLUTION ###
    pass
    # ### END SOLUTION ###

    # Save the model
    checkpoint_path = PART1_OUTPUT_DIR / "cnn_model.pt"
    torch.save(cnn_model, checkpoint_path)
    print(f"Saved CNN model to {checkpoint_path}")

    y_test_pred = predict_pytorch_classifier(cnn_model, test_loader, device)
    evaluate_classifier(y_test, y_test_pred, categories, model_name="CNN")


if __name__ == "__main__":
    main()
