"""
Q5: Circular Shift Experiment

Load trained 2-layer network and test on shifted images:
1. Circular shift RIGHT by 2 pixels → measure accuracy
2. Circular shift DOWN by 2 pixels (on right-shifted) → measure accuracy

NO training - just load model and evaluate.
"""

import os
import sys
import json
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from torch.utils.data import DataLoader, TensorDataset

from src.utils import (
    load_fashion_mnist,
    create_model,
    get_device,
)

# Path to saved model from Q2
MODEL_PATH = os.path.join(project_root, 'models', 'q2_two_layer.pth')
OUTPUT_DIR = os.path.join(project_root, 'outputs', 'q5')


def circular_shift_right(images, shift=2):
    """Circular shift RIGHT by `shift` pixels."""
    return torch.roll(images, shifts=shift, dims=-1)


def circular_shift_down(images, shift=2):
    """Circular shift DOWN by `shift` pixels."""
    return torch.roll(images, shifts=shift, dims=-2)


def evaluate_shifted(model, data, targets, device, batch_size=100):
    """Evaluate model on data."""
    model.eval()

    # Normalize same as training: (x/255 - 0.5) / 0.5
    data_norm = (data.float() / 255.0 - 0.5) / 0.5
    if data_norm.dim() == 3:
        data_norm = data_norm.unsqueeze(1)

    dataset = TensorDataset(data_norm, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def run_shift_experiment():
    """Run Q5 circular shift experiment."""
    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load saved model
    print(f"\nLoading trained model from: {MODEL_PATH}")
    model = create_model(num_layers=2, activation='relu')
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    print("Model loaded successfully!")

    # Load test data
    print("Loading FashionMNIST test data...")
    _, _, _, testset = load_fashion_mnist(batch_size=100)
    test_data = testset.data.clone()
    test_targets = testset.targets.clone()

    # Evaluate
    print(f"\n{'='*60}")
    print("Q5: CIRCULAR SHIFT EVALUATION")
    print(f"{'='*60}")

    # 1. Original
    original_acc = evaluate_shifted(model, test_data, test_targets, device)
    print(f"\n1. Original test accuracy: {original_acc:.2f}%")

    # 2. Shift RIGHT by 2
    shifted_right = circular_shift_right(test_data, shift=2)
    acc_right = evaluate_shifted(model, shifted_right, test_targets, device)
    print(f"2. After RIGHT shift (2px): {acc_right:.2f}%  (drop: {original_acc - acc_right:.2f}%)")

    # 3. Shift DOWN by 2 (on right-shifted)
    shifted_right_down = circular_shift_down(shifted_right, shift=2)
    acc_right_down = evaluate_shifted(model, shifted_right_down, test_targets, device)
    print(f"3. After RIGHT+DOWN shift:  {acc_right_down:.2f}%  (drop: {original_acc - acc_right_down:.2f}%)")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Original:         {original_acc:.2f}%")
    print(f"Right shift:      {acc_right:.2f}%  (Δ {original_acc - acc_right:+.2f}%)")
    print(f"Right+Down shift: {acc_right_down:.2f}%  (Δ {original_acc - acc_right_down:+.2f}%)")

    # Save results
    results = {
        'original_accuracy': original_acc,
        'after_right_shift': acc_right,
        'after_right_down_shift': acc_right_down,
        'drop_from_right': original_acc - acc_right,
        'drop_from_right_down': original_acc - acc_right_down
    }

    results_path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == '__main__':
    run_shift_experiment()
