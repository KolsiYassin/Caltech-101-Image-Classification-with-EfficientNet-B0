import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# Configuration
EXPERIMENT = "class_weighted"

HISTORY_PATH = (
    Path("./results") /
    EXPERIMENT /
    "training_history.npy"
)

PLOTS_DIR = Path("./plots")

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Load training history
history = np.load(
    HISTORY_PATH,
    allow_pickle=True
).item()

epochs = range(
    1,
    len(history["train_loss"]) + 1
)


# Loss plot
plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["train_loss"],
    marker="o",
    label="Training loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    marker="o",
    label="Validation loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    f"{EXPERIMENT.replace('_', ' ').title()} "
    "Training and Validation Loss"
)

plt.xticks(list(epochs))
plt.legend()
plt.grid(True)

plt.tight_layout()

loss_path = (
    PLOTS_DIR /
    f"{EXPERIMENT}_loss.png"
)

plt.savefig(
    loss_path,
    dpi=300
)

plt.close()


# Accuracy plot
plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["train_accuracy"],
    marker="o",
    label="Training accuracy"
)

plt.plot(
    epochs,
    history["val_accuracy"],
    marker="o",
    label="Validation accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    f"{EXPERIMENT.replace('_', ' ').title()} "
    "Training and Validation Accuracy"
)

plt.xticks(list(epochs))
plt.legend()
plt.grid(True)

plt.tight_layout()

accuracy_path = (
    PLOTS_DIR /
    f"{EXPERIMENT}_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=300
)

plt.close()


print("Experiment:", EXPERIMENT)

print("Saved:")
print(f"  {loss_path}")
print(f"  {accuracy_path}")