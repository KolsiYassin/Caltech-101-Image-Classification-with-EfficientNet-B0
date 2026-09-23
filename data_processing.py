import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import Caltech101


# Configuration
DATA_ROOT = "./data"
SPLITS_DIR = "./splits"

BATCH_SIZE = 32
NUM_WORKERS = 0


# ImageNet normalization used by pretrained EfficientNet-B0.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# Transforms
def get_transforms(augmentation=False):
    """
    Create the training and evaluation transforms.

    Parameters
    ----------
    augmentation : bool
        If True, apply data augmentation to training images.
        Validation and test images are never augmented.

    Returns
    -------
    train_transform : torchvision.transforms.Compose
        Transform used for training images.

    eval_transform : torchvision.transforms.Compose
        Transform used for validation and test images.
    """

    if augmentation:

        ### First augmentation

        # train_transform = transforms.Compose([
        #     # Convert grayscale images to 3-channel RGB.
        #     transforms.Lambda(
        #         lambda image: image.convert("RGB")
        #     ),

        #     # Randomly crop and resize the image to 224 x 224.
        #     transforms.RandomResizedCrop(
        #         224,
        #         scale=(0.8, 1.0)
        #     ),

        #     # Randomly flip images horizontally.
        #     transforms.RandomHorizontalFlip(
        #         p=0.5
        #     ),

        #     # Slightly change brightness, contrast,
        #     # saturation, and hue.
        #     transforms.ColorJitter(
        #         brightness=0.2,
        #         contrast=0.2,
        #         saturation=0.2,
        #         hue=0.05
        #     ),

        #     # Convert PIL image to PyTorch tensor.
        #     transforms.ToTensor(),

        #     # ImageNet normalization.
        #     transforms.Normalize(
        #         mean=IMAGENET_MEAN,
        #         std=IMAGENET_STD
        #     )
        # ])


        ### Augmentation 2
#         train_transform = transforms.Compose([
#             transforms.Lambda(lambda image: image.convert("RGB")),

#             transforms.Resize(256),

#             transforms.CenterCrop(224),

#             transforms.RandomHorizontalFlip(p=0.5),

#             transforms.RandomRotation(10),

#             transforms.ToTensor(),

#             transforms.Normalize(
#                 mean=IMAGENET_MEAN,
#                 std=IMAGENET_STD
#             )
# ])

        ### Augmentation 3
        train_transform = transforms.Compose([
            transforms.Lambda(lambda image: image.convert("RGB")),

            transforms.Resize(256),

            transforms.CenterCrop(224),

            transforms.RandomHorizontalFlip(p=0.5),

            transforms.RandomRotation(10),

            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            )
        ])


    else:

        # Baseline preprocessing.
        train_transform = transforms.Compose([
            # Convert grayscale images to 3-channel RGB.
            transforms.Lambda(
                lambda image: image.convert("RGB")
            ),

            # Resize the shorter side to 256 pixels.
            transforms.Resize(256),

            # Take the center 224 x 224 crop.
            transforms.CenterCrop(224),

            # Convert PIL image to PyTorch tensor.
            transforms.ToTensor(),

            # ImageNet normalization.
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            )
        ])

    # Validation and test preprocessing is always deterministic.
    eval_transform = transforms.Compose([
        # Convert grayscale images to 3-channel RGB.
        transforms.Lambda(
            lambda image: image.convert("RGB")
        ),

        # Resize the shorter side to 256 pixels.
        transforms.Resize(256),

        # Take the center 224 x 224 crop.
        transforms.CenterCrop(224),

        # Convert PIL image to PyTorch tensor.
        transforms.ToTensor(),

        # ImageNet normalization.
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])

    return train_transform, eval_transform


# Load split indices
def load_split_indices():
    """
    Load the previously created train, validation, and test
    indices.

    The same fixed split is reused for all experiments.
    """

    train_indices = np.load(
        Path(SPLITS_DIR) / "train_indices.npy"
    )

    val_indices = np.load(
        Path(SPLITS_DIR) / "val_indices.npy"
    )

    test_indices = np.load(
        Path(SPLITS_DIR) / "test_indices.npy"
    )

    return train_indices, val_indices, test_indices


# Create datasets
def get_datasets(augmentation=False):
    """
    Create the train, validation, and test datasets.

    Parameters
    ----------
    augmentation : bool
        Whether to use data augmentation for training.

    Training uses either the baseline or augmented transform.

    Validation and test always use the deterministic
    evaluation transform.
    """

    # Load saved split indices.
    train_indices, val_indices, test_indices = (
        load_split_indices()
    )

    # Get preprocessing transforms.
    train_transform, eval_transform = get_transforms(
        augmentation=augmentation
    )

    # Training dataset.
    train_dataset_full = Caltech101(
        root=DATA_ROOT,
        download=True,
        transform=train_transform
    )

    # Validation and test dataset.
    eval_dataset_full = Caltech101(
        root=DATA_ROOT,
        download=True,
        transform=eval_transform
    )

    # Apply the saved indices to create the subsets.
    train_dataset = Subset(
        train_dataset_full,
        train_indices
    )

    val_dataset = Subset(
        eval_dataset_full,
        val_indices
    )

    test_dataset = Subset(
        eval_dataset_full,
        test_indices
    )

    return train_dataset, val_dataset, test_dataset


# Create DataLoaders
def get_dataloaders(augmentation=False):
    """
    Create and return the train, validation, and test
    DataLoaders.

    Parameters
    ----------
    augmentation : bool
        Whether to use data augmentation for training.
    """

    train_dataset, val_dataset, test_dataset = (
        get_datasets(
            augmentation=augmentation
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_loader, val_loader, test_loader


# Dataset information
def get_dataset_info():
    """
    Return basic information about the dataset.
    """

    train_indices, val_indices, test_indices = (
        load_split_indices()
    )

    dataset = Caltech101(
        root=DATA_ROOT,
        download=True
    )

    return {
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "test_size": len(test_indices),
        "num_classes": len(dataset.categories),
        "class_names": dataset.categories
    }


def get_class_weights():
    train_indices, _, _ = load_split_indices()

    dataset = Caltech101(
        root=DATA_ROOT,
        download=True
    )

    train_labels = np.array(dataset.y)[train_indices]

    class_counts = np.bincount(
        train_labels,
        minlength=len(dataset.categories)
    )

    class_weights = (
        len(train_labels) /
        (len(dataset.categories) * class_counts)
    )

    return class_weights