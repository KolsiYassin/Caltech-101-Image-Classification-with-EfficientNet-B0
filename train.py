import random
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from data_processing import (
    get_dataloaders,
    get_class_weights
)


# Configuration
RANDOM_SEED = 42

NUM_CLASSES = 101
BATCH_SIZE = 32
NUM_EPOCHS = 30

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 0.0

EXPERIMENT = "random_init"
AUGMENTATION = False

USE_CLASS_WEIGHTS = False

RESULTS_DIR = Path("./results") / EXPERIMENT
WEIGHTS_DIR = Path("./weights")


# Reproducibility
def set_seed(seed):
    """
    Set random seeds so that experiments are as reproducible
    as possible.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# Device
def get_device():
    """
    Select GPU if available, otherwise use CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# Model
def create_model():
    """
    Create EfficientNet-B0 with ImageNet pretrained weights.

    The original ImageNet classifier predicts 1000 classes.
    Caltech-101 has 101 classes, so we replace the classifier.
    """


    # remove this for random init
    #weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(
        weights=None
    ) # if not random init then set to weights


    # Replace the final classifier.
    #
    # EfficientNet-B0 has:
    #
    # classifier[0] -> Dropout
    # classifier[1] -> Linear
    #
    # The Linear layer originally outputs 1000 ImageNet classes.
    # We replace it with a layer that outputs 101 Caltech-101 classes.

    num_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        num_features,
        NUM_CLASSES
    )

    return model


# Training
def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device
):
    """
    Train the model for one complete epoch.

    Returns:
        average training loss
        training accuracy
    """

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        # Move data to GPU/CPU.
        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients from previous iteration.
        optimizer.zero_grad()

        # Forward pass.
        outputs = model(images)

        # Calculate loss.
        loss = criterion(outputs, labels)

        # Backpropagation.
        loss.backward()

        # Update model parameters.
        optimizer.step()

        # Accumulate loss.
        running_loss += loss.item() * images.size(0)

        # Calculate predictions.
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# Validation
def validate(
    model,
    val_loader,
    criterion,
    device
):
    """
    Evaluate the model on the validation set.

    No gradients are calculated during validation.
    """

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass.
            outputs = model(images)

            # Calculate loss.
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Predictions.
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy



def main():

    # Setup
    set_seed(RANDOM_SEED)

    device = get_device()

    print(f"Using device: {device}")

    # Create output directories.
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    WEIGHTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Data
    print("\nLoading data...")

    train_loader, val_loader, test_loader = get_dataloaders(augmentation=AUGMENTATION)

    print(f"Training batches:   {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches:       {len(test_loader)}")

    # Model
    print("\nCreating EfficientNet-B0...")

    model = create_model()

    model = model.to(device)

    print("Model created.")
    print(f"Number of classes: {NUM_CLASSES}")

    # Loss function
    if USE_CLASS_WEIGHTS:

        class_weights = get_class_weights()

        print("\nClass weights:")
        print(f"Minimum: {class_weights.min():.4f}")
        print(f"Maximum: {class_weights.max():.4f}")
        print(f"Mean:    {class_weights.mean():.4f}")

        class_weights = torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=device
        )

        criterion = nn.CrossEntropyLoss(
            weight=class_weights
        )

    else:

        criterion = nn.CrossEntropyLoss()


    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Training history
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    # Track the best validation accuracy.
    best_val_accuracy = 0.0

    best_model_path = Path("./weights") / f"{EXPERIMENT}_best.pth"

    # Training loop
    print("\nStarting training...")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print()

    for epoch in range(NUM_EPOCHS):

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )

        # Training.
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # Validation.
        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
            device
        )

        # Store results.
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        # Print results.
        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f}"
        )

        print(
            f"Val Loss:   {val_loss:.4f} | "
            f"Val Acc:   {val_accuracy:.4f}"
        )

        # Save best model
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss
                },
                best_model_path
            )

            print(
                f"Saved best model "
                f"(Val Acc: {val_accuracy:.4f})"
            )

        print()

    # Save training history
    history_path = RESULTS_DIR / "training_history.npy"

    np.save(
        history_path,
        history
    )

    # Save final model
    final_model_path = Path("./weights") / f"{EXPERIMENT}_final.pth"

    torch.save(
        model.state_dict(),
        final_model_path
    )

    # Summary
    print("Training complete.")

    print(
        f"\nBest validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Best model saved to: "
        f"{best_model_path}"
    )

    print(
        f"Final model saved to: "
        f"{final_model_path}"
    )

    print(
        f"Training history saved to: "
        f"{history_path}"
    )


if __name__ == "__main__":
    main()