"""Part 1 Model 1: k-Nearest Neighbors (k-NN) Sketch Classifier."""

import pickle

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from part1_common import (
    PART1_OUTPUT_DIR,
    RANDOM_SEED,
    evaluate_classifier,
    load_part1_splits,
    set_seed,
)

# =============================================================================
# Hyperparameters
# =============================================================================
# ### BEGIN SOLUTION ###
pass
# ### END SOLUTION ###


# =============================================================================
# Training & Validation Tuning
# =============================================================================
def main():
    set_seed(RANDOM_SEED)

    # Load preprocessed [0, 1] features
    (X_train, y_train), (X_val, y_val), (X_test, y_test), metadata = load_part1_splits()
    categories = metadata["categories"]
    print(f"Loaded dataset: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    knn_model = None
    # Use the already imported KNeighborsClassifier to train the knn_model. We have also imported accuracy_score to help.
    # ### BEGIN SOLUTION ###
    pass
    # ### END SOLUTION ###

    # Save the model
    checkpoint_path = PART1_OUTPUT_DIR / "knn_model.pkl"
    with open(checkpoint_path, "wb") as f:
        pickle.dump(knn_model, f)
    print(f"Saved k-NN model to {checkpoint_path}")

    # Evaluate the model on the test set
    y_test_pred = knn_model.predict(X_test)
    evaluate_classifier(y_test, y_test_pred, categories, model_name="kNN")


if __name__ == "__main__":
    main()
