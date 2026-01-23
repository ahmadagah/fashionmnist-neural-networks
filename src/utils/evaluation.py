"""
Evaluation and visualization utilities.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime


def compute_confusion_matrix(predictions, labels, num_classes=10):
    """
    Compute confusion matrix.

    Args:
        predictions: Tensor of predicted labels
        labels: Tensor of true labels
        num_classes: Number of classes

    Returns:
        confusion_matrix: numpy array of shape (num_classes, num_classes)
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)

    for pred, label in zip(predictions, labels):
        cm[label.item()][pred.item()] += 1

    return cm


def compute_per_class_accuracy(confusion_matrix):
    """
    Compute per-class accuracy from confusion matrix.

    Args:
        confusion_matrix: numpy array of shape (num_classes, num_classes)

    Returns:
        per_class_acc: numpy array of accuracy for each class
    """
    return np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)


def plot_training_history(history, save_path=None, title="Training History"):
    """
    Plot training loss and accuracy curves.

    Args:
        history: Dictionary with train_loss, train_acc, test_acc keys
        save_path: Path to save the figure (optional)
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    if history.get('test_acc'):
        axes[1].plot(epochs, history['test_acc'], 'r-', label='Test Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.close()


def plot_comparison(histories, labels, save_path=None, title="Model Comparison"):
    """
    Plot comparison of multiple training runs.

    Args:
        histories: List of history dictionaries
        labels: List of labels for each history
        save_path: Path to save the figure
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    for hist, label, color in zip(histories, labels, colors):
        epochs = range(1, len(hist['train_loss']) + 1)

        # Loss
        axes[0].plot(epochs, hist['train_loss'], color=color, label=label)

        # Test accuracy (if available)
        if hist.get('test_acc'):
            axes[1].plot(epochs, hist['test_acc'], color=color, label=label)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].set_title('Test Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.close()


def plot_confusion_matrix(cm, class_names, save_path=None, title="Confusion Matrix"):
    """
    Plot confusion matrix as heatmap.

    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True label',
        xlabel='Predicted label'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.close()


def plot_accuracy_heatmap(results_df, x_col, y_col, value_col, save_path=None, title="Accuracy Heatmap"):
    """
    Plot accuracy as a heatmap for hyperparameter grid.

    Args:
        results_df: DataFrame with results
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        value_col: Column name for values
        save_path: Path to save the figure
        title: Plot title
    """
    pivot = results_df.pivot(index=y_col, columns=x_col, values=value_col)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)

    plt.colorbar(im, label='Accuracy (%)')

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    color='white' if val < 50 else 'black')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.close()


def save_results(results, path):
    """
    Save results dictionary to JSON file.

    Args:
        results: Dictionary of results
        path: Path to save JSON file
    """
    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    results_converted = convert(results)
    results_converted['timestamp'] = datetime.now().isoformat()

    with open(path, 'w') as f:
        json.dump(results_converted, f, indent=2)

    print(f"Saved results to {path}")


def load_results(path):
    """
    Load results from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        results: Dictionary of results
    """
    with open(path, 'r') as f:
        return json.load(f)


def print_summary(results, title="Results Summary"):
    """
    Print a formatted summary of results.

    Args:
        results: Dictionary containing experiment results
        title: Summary title
    """
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

    for key, value in results.items():
        if key == 'timestamp':
            continue
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], float):
                print(f"  {key}: [final: {value[-1]:.4f}]")
            else:
                print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")

    print("=" * 60 + "\n")
