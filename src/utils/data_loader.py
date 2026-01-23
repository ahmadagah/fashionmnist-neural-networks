"""
Data loading utilities for FashionMNIST dataset.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

# FashionMNIST class labels
CLASSES = (
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'
)

# Default data directory (relative to project root)
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


def get_transform():
    """Get the standard transform for FashionMNIST images."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def load_fashion_mnist(data_dir=None, batch_size=30, num_workers=0):
    """
    Load FashionMNIST train and test datasets.

    Args:
        data_dir: Directory to store/load data. Defaults to project data/ folder.
        batch_size: Batch size for data loaders.
        num_workers: Number of worker processes for data loading.

    Returns:
        trainloader: DataLoader for training set
        testloader: DataLoader for test set
        trainset: Raw training dataset
        testset: Raw test dataset
    """
    if data_dir is None:
        data_dir = DATA_DIR

    transform = get_transform()

    # Load datasets
    trainset = torchvision.datasets.FashionMNIST(
        data_dir,
        download=True,
        train=True,
        transform=transform
    )

    testset = torchvision.datasets.FashionMNIST(
        data_dir,
        download=True,
        train=False,
        transform=transform
    )

    # Create data loaders
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return trainloader, testloader, trainset, testset


def get_data_loaders(trainset, testset, batch_size=30, num_workers=0):
    """
    Create data loaders from existing datasets.

    Useful when you need to change batch size without reloading data.

    Args:
        trainset: Training dataset
        testset: Test dataset
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes

    Returns:
        trainloader: DataLoader for training set
        testloader: DataLoader for test set
    """
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return trainloader, testloader


def get_class_indices(dataset):
    """
    Get indices of samples for each class.

    Args:
        dataset: PyTorch dataset with targets attribute

    Returns:
        dict: Mapping from class index to list of sample indices
    """
    class_indices = {i: [] for i in range(10)}

    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    return class_indices


def get_class_indices_fast(dataset):
    """
    Fast version using targets tensor directly.

    Args:
        dataset: FashionMNIST dataset

    Returns:
        dict: Mapping from class index to list of sample indices
    """
    targets = dataset.targets
    if isinstance(targets, list):
        targets = torch.tensor(targets)

    class_indices = {}
    for class_idx in range(10):
        class_indices[class_idx] = (targets == class_idx).nonzero(as_tuple=True)[0].tolist()

    return class_indices


def create_polluted_dataset(trainset, pollution_fraction=0.01, seed=42):
    """
    Create a polluted version of the training dataset.

    For each class, randomly select pollution_fraction of images and
    redistribute them to other classes (relabeling).

    Args:
        trainset: Original training dataset
        pollution_fraction: Fraction of each class to pollute (default 0.01 = 1%)
        seed: Random seed for reproducibility

    Returns:
        polluted_data: Tensor of image data
        polluted_targets: Tensor of (polluted) labels
    """
    np.random.seed(seed)

    # Get class indices
    class_indices = get_class_indices_fast(trainset)

    # Copy original data and targets
    data = trainset.data.clone()
    targets = trainset.targets.clone()
    if isinstance(targets, list):
        targets = torch.tensor(targets)

    # For each class, select samples to pollute
    for class_idx in range(10):
        indices = class_indices[class_idx]
        n_samples = len(indices)
        n_pollute = int(n_samples * pollution_fraction)

        # Select random samples to redistribute
        pollute_indices = np.random.choice(indices, size=n_pollute * 9, replace=False)

        # Redistribute to other 9 classes (n_pollute samples each)
        other_classes = [c for c in range(10) if c != class_idx]
        for i, other_class in enumerate(other_classes):
            start_idx = i * n_pollute
            end_idx = (i + 1) * n_pollute
            sample_indices = pollute_indices[start_idx:end_idx]
            targets[sample_indices] = other_class

    return data, targets


class PollutedFashionMNIST(torch.utils.data.Dataset):
    """Dataset wrapper for polluted FashionMNIST data."""

    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]

        # Convert to PIL Image for transforms
        from PIL import Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform:
            img = self.transform(img)

        return img, target
