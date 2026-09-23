import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from torchvision.models import efficientnet_b0

from data_processing import get_dataloaders, get_dataset_info


# Configuration
EXPERIMENT = "class_weighted"

NUM_CLASSES = 101

WEIGHTS_PATH = Path("./weights") / f"{EXPERIMENT}_best.pth"
RESULTS_DIR = Path("./results") / EXPERIMENT


# Device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# Model
def create_model():
    model = efficientnet_b0(weights=None)

    num_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        num_features,
        NUM_CLASSES
    )

    return model


# Load model
def load_model(device):
    model = create_model()

    checkpoint = torch.load(
        WEIGHTS_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Validation accuracy at checkpoint: "
        f"{checkpoint['val_accuracy']:.4f}"
    )

    return model


# Evaluation
def evaluate(
    model,
    test_loader,
    criterion,
    device
):
    model.eval()

    running_loss = 0.0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Loss
            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            # Predictions
            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    test_loss = (
        running_loss /
        len(test_loader.dataset)
    )

    test_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        test_loss,
        test_accuracy,
        macro_f1,
        np.array(all_labels),
        np.array(all_predictions)
    )


# Main
def main():

    print("=" * 60)
    print(f"EXPERIMENT: {EXPERIMENT}")
    print("=" * 60)

    # Device
    device = get_device()

    print(f"Using device: {device}")

    # Create results directory
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Check that checkpoint exists
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{WEIGHTS_PATH}"
        )

    # Data
    print("\nLoading data...")

    _, _, test_loader = get_dataloaders()

    dataset_info = get_dataset_info()

    class_names = dataset_info["class_names"]

    print(
        f"Test samples: {len(test_loader.dataset)}"
    )

    print(
        f"Number of classes: {len(class_names)}"
    )

    # Model
    print("\nLoading best model...")

    model = load_model(device)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Evaluation
    print("\nEvaluating on test set...")

    (
        test_loss,
        test_accuracy,
        macro_f1,
        labels,
        predictions
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    # Overall metrics
    print("\n" + "=" * 60)
    print(f"{EXPERIMENT.upper()} TEST RESULTS")
    print("=" * 60)

    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Macro F1:      {macro_f1:.4f}")

    # Classification report
    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(
        labels,
        predictions
    )

    np.save(
        RESULTS_DIR / "confusion_matrix.npy",
        cm
    )

    # Labels and predictions
    np.save(
        RESULTS_DIR / "test_labels.npy",
        labels
    )

    np.save(
        RESULTS_DIR / "test_predictions.npy",
        predictions
    )

    # Overall metrics
    metrics = {
        "experiment": EXPERIMENT,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "macro_f1": macro_f1
    }

    np.save(
        RESULTS_DIR / "test_metrics.npy",
        metrics
    )

    print(
        f"\nSaved test results to: {RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()