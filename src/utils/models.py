"""
Neural network model architectures for FashionMNIST classification.
"""

import torch
import torch.nn as nn


class SingleLayerFC(nn.Module):
    """
    Single hidden layer fully-connected network.

    Architecture: 784 -> 1024 -> 10

    Args:
        activation: 'relu' or 'sigmoid'
    """

    def __init__(self, activation='relu'):
        super(SingleLayerFC, self).__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 10)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}. Use 'relu' or 'sigmoid'.")

        self.activation_name = activation

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

    def __repr__(self):
        return f"SingleLayerFC(784 -> 1024 -> 10, activation={self.activation_name})"


class TwoLayerFC(nn.Module):
    """
    Two hidden layer fully-connected network.

    Architecture: 784 -> 1024 -> 1024 -> 10

    Args:
        activation: 'relu' or 'sigmoid'
    """

    def __init__(self, activation='relu'):
        super(TwoLayerFC, self).__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}. Use 'relu' or 'sigmoid'.")

        self.activation_name = activation

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x

    def __repr__(self):
        return f"TwoLayerFC(784 -> 1024 -> 1024 -> 10, activation={self.activation_name})"


def create_model(num_layers=2, activation='relu'):
    """
    Factory function to create FC models.

    Args:
        num_layers: Number of hidden layers (1 or 2)
        activation: 'relu' or 'sigmoid'

    Returns:
        model: PyTorch model
    """
    if num_layers == 1:
        return SingleLayerFC(activation=activation)
    elif num_layers == 2:
        return TwoLayerFC(activation=activation)
    else:
        raise ValueError(f"num_layers must be 1 or 2, got {num_layers}")


def get_device():
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path):
    """Save model state dict to file."""
    torch.save(model.state_dict(), path)


def load_model(model, path, device=None):
    """Load model state dict from file."""
    if device is None:
        device = get_device()
    model.load_state_dict(torch.load(path, map_location=device))
    return model
