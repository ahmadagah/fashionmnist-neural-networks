"""
Training utilities for neural network models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time


def train_model(
    model,
    trainloader,
    testloader=None,
    epochs=20,
    learning_rate=0.001,
    momentum=0.0,
    device=None,
    verbose=True
):
    """
    Train a model using SGD optimizer and CrossEntropy loss.

    Args:
        model: PyTorch model to train
        trainloader: DataLoader for training data
        testloader: Optional DataLoader for test data (for tracking accuracy)
        epochs: Number of training epochs
        learning_rate: Learning rate for SGD
        momentum: Momentum for SGD
        device: Device to train on (auto-detected if None)
        verbose: Whether to print progress

    Returns:
        history: Dictionary containing training metrics
            - train_loss: List of average loss per epoch
            - train_acc: List of training accuracy per epoch
            - test_acc: List of test accuracy per epoch (if testloader provided)
            - epoch_times: List of time per epoch
    """
    if device is None:
        from .models import get_device
        device = get_device()

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'epoch_times': []
    }

    for epoch in range(epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        iterator = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", disable=not verbose)

        for inputs, labels in iterator:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if verbose:
                iterator.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total
        epoch_time = time.time() - epoch_start

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['epoch_times'].append(epoch_time)

        # Evaluation phase
        if testloader is not None:
            test_acc = evaluate_model(model, testloader, device)
            history['test_acc'].append(test_acc)

            if verbose:
                print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, "
                      f"Train Acc={epoch_acc:.2f}%, Test Acc={test_acc:.2f}%, "
                      f"Time={epoch_time:.2f}s")
        else:
            if verbose:
                print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, "
                      f"Train Acc={epoch_acc:.2f}%, Time={epoch_time:.2f}s")

    return history


def evaluate_model(model, dataloader, device=None):
    """
    Evaluate model accuracy on a dataset.

    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on

    Returns:
        accuracy: Accuracy percentage (0-100)
    """
    if device is None:
        from .models import get_device
        device = get_device()

    model = model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def get_predictions(model, dataloader, device=None):
    """
    Get all predictions and true labels for a dataset.

    Args:
        model: PyTorch model
        dataloader: DataLoader for data
        device: Device to use

    Returns:
        all_preds: Tensor of predicted labels
        all_labels: Tensor of true labels
        all_probs: Tensor of prediction probabilities
    """
    if device is None:
        from .models import get_device
        device = get_device()

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.append(predicted.cpu())
            all_labels.append(labels)
            all_probs.append(probs.cpu())

    return torch.cat(all_preds), torch.cat(all_labels), torch.cat(all_probs)
