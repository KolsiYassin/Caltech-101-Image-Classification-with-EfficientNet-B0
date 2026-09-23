from torchvision.datasets import Caltech101
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


DATA_ROOT = "./data"
RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


dataset = Caltech101(
    root=DATA_ROOT,
    download=True
)

print(f"Total images: {len(dataset)}")
print(f"Number of classes: {len(dataset.categories)}")


labels = [dataset[i][1] for i in range(len(dataset))]
indices = list(range(len(dataset)))

# Stratified train split

train_indices, temp_indices = train_test_split(
    indices,
    test_size=VAL_RATIO + TEST_RATIO,
    stratify=labels,
    random_state=RANDOM_SEED
)


# Stratified validation/test split

temp_labels = [labels[i] for i in temp_indices]

val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
    stratify=temp_labels,
    random_state=RANDOM_SEED
)


print("\nDataset split:")
print(f"Train:      {len(train_indices)}")
print(f"Validation: {len(val_indices)}")
print(f"Test:       {len(test_indices)}")

# Check number of classes

train_labels = [labels[i] for i in train_indices]
val_labels = [labels[i] for i in val_indices]
test_labels = [labels[i] for i in test_indices]

print("\nNumber of classes:")
print(f"Train:      {len(set(train_labels))}")
print(f"Validation: {len(set(val_labels))}")
print(f"Test:       {len(set(test_labels))}")

# Check class distribution

train_counts = Counter(train_labels)
val_counts = Counter(val_labels)
test_counts = Counter(test_labels)

print("\nClass distribution:")
for class_idx, class_name in enumerate(dataset.categories):
    print(
        f"{class_name:20s} "
        f"Train: {train_counts[class_idx]:3d} | "
        f"Val: {val_counts[class_idx]:3d} | "
        f"Test: {test_counts[class_idx]:3d}"
    )



# Plot class distribution
plots_dir = Path("./plots")
plots_dir.mkdir(exist_ok=True)

class_names = dataset.categories
x = np.arange(len(class_names))

train_values = [train_counts[i] for i in range(len(class_names))]
val_values = [val_counts[i] for i in range(len(class_names))]
test_values = [test_counts[i] for i in range(len(class_names))]

plt.figure(figsize=(20, 8))

plt.bar(x, train_values, label="Train")
plt.bar(
    x,
    val_values,
    bottom=train_values,
    label="Validation"
)
plt.bar(
    x,
    test_values,
    bottom=np.array(train_values) + np.array(val_values),
    label="Test"
)

plt.xlabel("Class")
plt.ylabel("Number of images")
plt.title("Caltech-101 Class Distribution After Stratified Split")

plt.xticks(x, class_names, rotation=90)
plt.legend()

plt.tight_layout()

plt.savefig(
    plots_dir / "split_class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)


splits_dir = Path("./splits")
splits_dir.mkdir(exist_ok=True)

np.save(splits_dir / "train_indices.npy", train_indices)
np.save(splits_dir / "val_indices.npy", val_indices)
np.save(splits_dir / "test_indices.npy", test_indices)

print("\nSplit indices saved.")