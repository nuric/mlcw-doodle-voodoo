"""Part 2 Step 3 — Autoregressive GRU over Discrete K-Means Stroke Tokens."""

import torch
import torch.nn as nn

from helpers import plot_training_history
from part2_common import (
    PAD_TOKEN,
    PART2_OUTPUT_DIR,
    RANDOM_SEED,
    evaluate_token_predictor,
    get_device,
    load_tokenized_splits,
    make_dataloaders,
    set_seed,
    train_token_predictor,
)

# =============================================================================
# Hyperparameters
# =============================================================================
# We give BATCH_SIZE to get the data loading plumbing working.
BATCH_SIZE = 128
# Add your hyperparameters here as usual
# ### BEGIN SOLUTION ###
pass
# ### END SOLUTION ###


# =============================================================================
# Model Definition (Autoregressive Token GRU)
# =============================================================================
class NextStrokeRNN(nn.Module):
    """Autoregressive GRU predicting next stroke token ID t+1 from tokens 0..t."""

    # Define your model __init__ and forward methods here.
    # ### BEGIN SOLUTION ###
    pass
    # ### END SOLUTION ###


# =============================================================================
# Training & Evaluation
# =============================================================================
def main():
    set_seed(RANDOM_SEED)
    device = get_device()

    train_split, val_split, test_split, centroids, metadata = load_tokenized_splits()
    vocab_size = metadata["vocab_size"]
    eos_token = metadata["eos_token"]

    print(
        f"Loaded Part 2 tokenized dataset ({len(metadata['categories'])} categories, "
        f"vocab_size={vocab_size}): "
        f"train={len(train_split['x'])}, val={len(val_split['x'])}, test={len(test_split['x'])} drawings"
    )

    train_loader, val_loader, test_loader = make_dataloaders(
        train_split, val_split, test_split, batch_size=BATCH_SIZE
    )

    rnn_model = None
    # Use the already defined NextStrokeRNN class to train the rnn_model.
    # We have also imported train_token_predictor and plot_training_history to help.
    # ### BEGIN SOLUTION ###
    pass
    # ### END SOLUTION ###

    checkpoint_path = PART2_OUTPUT_DIR / "rnn_model.pt"
    torch.save(rnn_model, checkpoint_path)
    print(f"Saved RNN model to {checkpoint_path}")

    evaluate_token_predictor(
        model=rnn_model,
        test_loader=test_loader,
        test_split=test_split,
        centroids=centroids,
        eos_token=eos_token,
        device=device,
        model_name="Part2 RNN",
    )


if __name__ == "__main__":
    main()
