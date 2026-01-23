"""
Q4: Pollution Experiment

Train a 2-layer FC network on polluted training data and test on clean test data.

Uses BEST parameters from Q3: batch_size=1, lr=0.001, activation=relu

Pollution: For each class, 9% of samples are relabeled (1% to each of the other 9 classes).
"""

import os
import sys
import json
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from torch.utils.data import DataLoader

from src.utils import (
    load_fashion_mnist,
    create_polluted_dataset,
    PollutedFashionMNIST,
    get_transform,
    create_model,
    get_device,
    train_model,
    evaluate_model,
)

# BEST parameters from Q3 (batch_size=1, lr=0.001, relu achieved 89.42%)
BEST_BATCH_SIZE = 1
BEST_LR = 0.001
BEST_ACTIVATION = 'relu'
EPOCHS = 20
MOMENTUM = 0.0
POLLUTION_FRACTION = 0.01  # 1% per class redistributed to each other class (9% total)

# Q3 baseline for comparison
Q3_BASELINE_ACC = 89.42  # Best from Q3: bs=1, lr=0.001, relu on clean data

OUTPUT_DIR = os.path.join(project_root, 'outputs', 'q4')


def run_pollution_experiment():
    """Run the pollution experiment."""
    device = get_device()
    print(f"Using device: {device}")
    print(f"\nBest parameters from Q3:")
    print(f"  Batch size: {BEST_BATCH_SIZE}")
    print(f"  Learning rate: {BEST_LR}")
    print(f"  Activation: {BEST_ACTIVATION}")
    print(f"  Q3 baseline (clean): {Q3_BASELINE_ACC}%")
    print(f"  Pollution: {POLLUTION_FRACTION * 100}% per class to each other class (9% total)")

    # Load original data
    print("\nLoading FashionMNIST data...")
    _, testloader, trainset, testset = load_fashion_mnist(batch_size=BEST_BATCH_SIZE)

    # Create polluted dataset
    print("Creating polluted training dataset...")
    polluted_data, polluted_targets = create_polluted_dataset(
        trainset,
        pollution_fraction=POLLUTION_FRACTION,
        seed=42
    )

    # Count pollution statistics
    original_targets = trainset.targets
    n_changed = (polluted_targets != original_targets).sum().item()
    print(f"  Total samples relabeled: {n_changed} ({100 * n_changed / len(trainset):.1f}%)")

    # Create polluted dataset and dataloader
    polluted_trainset = PollutedFashionMNIST(
        polluted_data,
        polluted_targets,
        transform=get_transform()
    )

    polluted_trainloader = DataLoader(
        polluted_trainset,
        batch_size=BEST_BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # Train model on polluted data
    print(f"\n{'='*60}")
    print("Training on POLLUTED data")
    print(f"{'='*60}")

    model = create_model(num_layers=2, activation=BEST_ACTIVATION)

    start_time = time.time()
    history = train_model(
        model=model,
        trainloader=polluted_trainloader,
        testloader=testloader,
        epochs=EPOCHS,
        learning_rate=BEST_LR,
        momentum=MOMENTUM,
        device=device,
        verbose=True
    )
    total_time = time.time() - start_time

    # Final evaluation on clean test set
    test_acc = evaluate_model(model, testloader, device)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Q3 baseline (clean data):   {Q3_BASELINE_ACC:.2f}% test accuracy")
    print(f"Q4 polluted data training:  {test_acc:.2f}% test accuracy")
    print(f"Accuracy drop from pollution: {Q3_BASELINE_ACC - test_acc:.2f}%")

    # Save results
    results = {
        'config': {
            'batch_size': BEST_BATCH_SIZE,
            'learning_rate': BEST_LR,
            'activation': BEST_ACTIVATION,
            'epochs': EPOCHS,
            'momentum': MOMENTUM,
            'pollution_fraction': POLLUTION_FRACTION
        },
        'pollution_stats': {
            'total_samples': len(trainset),
            'samples_relabeled': n_changed,
            'pollution_percentage': 100 * n_changed / len(trainset)
        },
        'q3_baseline': {
            'test_acc': Q3_BASELINE_ACC,
            'note': 'Best from Q3: batch_size=1, lr=0.001, relu on clean data'
        },
        'polluted_training': {
            'test_acc': test_acc,
            'train_acc': history['train_acc'][-1],
            'train_loss': history['train_loss'][-1],
            'total_time': total_time,
            'history': history
        },
        'comparison': {
            'accuracy_drop': Q3_BASELINE_ACC - test_acc,
            'pollution_impact': f"{Q3_BASELINE_ACC - test_acc:.2f}% accuracy loss from 9% label pollution"
        }
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == '__main__':
    run_pollution_experiment()
