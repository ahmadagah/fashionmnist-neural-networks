import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    load_fashion_mnist,
    get_data_loaders,
    create_model,
    get_device,
    count_parameters,
    save_model,
    train_model,
    evaluate_model,
    save_results,
    CLASSES
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'q3')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

device = get_device()
print(f"Using device: {device}")

BATCH_SIZES = [1, 10, 1000]
LEARNING_RATES = [1.0, 0.1, 0.01, 0.001]
ACTIVATIONS = ['relu', 'sigmoid']

EPOCHS = 20
MOMENTUM = 0.0

RESULTS_FILE = os.path.join(OUTPUT_DIR, 'all_results.json')

print(f"Batch sizes: {BATCH_SIZES}")
print(f"Learning rates: {LEARNING_RATES}")
print(f"Activations: {ACTIVATIONS}")
print(f"Total experiments: {len(BATCH_SIZES) * len(LEARNING_RATES) * len(ACTIVATIONS)}")

print("Loading FashionMNIST datasets...")
_, _, trainset, testset = load_fashion_mnist(batch_size=30)
print(f"Training samples: {len(trainset):,}")
print(f"Test samples: {len(testset):,}")

def load_existing_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_all_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {RESULTS_FILE}")

def is_experiment_done(results, batch_size, learning_rate, activation):
    for r in results:
        if (r['batch_size'] == batch_size and
            r['learning_rate'] == learning_rate and
            r['activation'] == activation):
            return True
    return False

def run_experiment(batch_size, learning_rate, activation, epochs=20, verbose=True):
    if verbose:
        print(f"\n{'='*60}")
        print(f" Batch={batch_size}, LR={learning_rate}, Activation={activation}")
        print(f"{'='*60}")

    trainloader, testloader = get_data_loaders(trainset, testset, batch_size=batch_size)
    model = create_model(num_layers=2, activation=activation)

    if verbose:
        print(f"Batches per epoch: {len(trainloader):,}")

    start_time = time.time()
    history = train_model(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        epochs=epochs,
        learning_rate=learning_rate,
        momentum=MOMENTUM,
        device=device,
        verbose=verbose
    )
    total_time = time.time() - start_time

    test_acc = evaluate_model(model, testloader, device)

    if verbose:
        print(f"\nFinal test accuracy: {test_acc:.2f}%")
        print(f"Total time: {total_time:.1f}s")

    return {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'activation': activation,
        'test_acc': test_acc,
        'train_acc': history['train_acc'][-1],
        'train_loss': history['train_loss'][-1],
        'total_time': total_time,
        'history': history
    }

def run_batch_experiments(batch_size):
    all_results = load_existing_results()

    total = len(LEARNING_RATES) * len(ACTIVATIONS)
    done = 0

    for activation in ACTIVATIONS:
        for lr in LEARNING_RATES:
            if is_experiment_done(all_results, batch_size, lr, activation):
                print(f"[SKIP] batch={batch_size}, lr={lr}, act={activation} (already done)")
                done += 1
                continue

            result = run_experiment(batch_size, lr, activation, EPOCHS, verbose=True)
            all_results.append(result)
            save_all_results(all_results)
            done += 1
            print(f"Progress: {done}/{total} experiments for batch_size={batch_size}")

    print(f"\nAll experiments for batch_size={batch_size} complete!")
    return all_results

# Check current progress
existing = load_existing_results()
print(f"Completed experiments: {len(existing)}/24")

if existing:
    print("\nCompleted:")
    for r in existing:
        print(f"  batch={r['batch_size']}, lr={r['learning_rate']}, act={r['activation']} → {r['test_acc']:.2f}%")

remaining = 24 - len(existing)
print(f"\nRemaining: {remaining} experiments")

print("="*60)
print(" Running all experiments for BATCH_SIZE = 1000")
print("="*60)

all_results = run_batch_experiments(batch_size=1000)

print("="*60)
print(" Running all experiments for BATCH_SIZE = 10")
print("="*60)

all_results = run_batch_experiments(batch_size=10)

print("="*60)
print(" Running all experiments for BATCH_SIZE = 1")
print("="*60)

all_results = run_batch_experiments(batch_size=1)

all_results = load_existing_results()
print(f"Total completed: {len(all_results)}/24 experiments\n")

if len(all_results) < 24:
    print("WARNING: Not all experiments completed yet!")
    print("Run the batch size cells above to complete remaining experiments.\n")

print("="*70)
print(" Q3 RESULTS TABLE: Test Accuracy (%)")
print("="*70)
print(f"\n{'Activation':<10} {'Batch':<8} {'LR=1.0':<10} {'LR=0.1':<10} {'LR=0.01':<10} {'LR=0.001':<10}")
print("-" * 68)

for activation in ACTIVATIONS:
    for batch_size in BATCH_SIZES:
        row = f"{activation:<10} {batch_size:<8}"
        for lr in LEARNING_RATES:
            found = False
            for r in all_results:
                if (r['batch_size'] == batch_size and
                    r['learning_rate'] == lr and
                    r['activation'] == activation):
                    row += f" {r['test_acc']:<10.2f}"
                    found = True
                    break
            if not found:
                row += f" {'--':<10}"
        print(row)
    print()

if len(all_results) == 24:
    sorted_results = sorted(all_results, key=lambda x: x['test_acc'], reverse=True)
    best = sorted_results[0]
    worst = sorted_results[-1]

    print("="*60)
    print(" BEST AND WORST CASES")
    print("="*60)

    print(f"\n BEST:")
    print(f"   Batch={best['batch_size']}, LR={best['learning_rate']}, Act={best['activation']}")
    print(f"   Test Accuracy: {best['test_acc']:.2f}%")

    print(f"\n WORST:")
    print(f"   Batch={worst['batch_size']}, LR={worst['learning_rate']}, Act={worst['activation']}")
    print(f"   Test Accuracy: {worst['test_acc']:.2f}%")

    print(f"\n Difference: {best['test_acc'] - worst['test_acc']:.2f}%")
else:
    print("Complete all 24 experiments first to see best/worst cases.")

if len(all_results) == 24:
    sorted_results = sorted(all_results, key=lambda x: x['test_acc'], reverse=True)
    best = sorted_results[0]
    worst = sorted_results[-1]

    final_results = {
        'config': {
            'batch_sizes': BATCH_SIZES,
            'learning_rates': LEARNING_RATES,
            'activations': ACTIVATIONS,
            'epochs': EPOCHS,
            'momentum': MOMENTUM
        },
        'experiments': all_results,
        'best_case': {
            'batch_size': best['batch_size'],
            'learning_rate': best['learning_rate'],
            'activation': best['activation'],
            'test_acc': best['test_acc']
        },
        'worst_case': {
            'batch_size': worst['batch_size'],
            'learning_rate': worst['learning_rate'],
            'activation': worst['activation'],
            'test_acc': worst['test_acc']
        }
    }
    save_results(final_results, os.path.join(OUTPUT_DIR, 'results.json'))

    import csv
    csv_path = os.path.join(OUTPUT_DIR, 'accuracy_table.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Activation', 'Batch Size', 'Learning Rate', 'Test Accuracy', 'Train Accuracy', 'Time (s)'])
        for r in sorted(all_results, key=lambda x: (x['activation'], x['batch_size'], -x['learning_rate'])):
            writer.writerow([r['activation'], r['batch_size'], r['learning_rate'],
                           f"{r['test_acc']:.2f}", f"{r['train_acc']:.2f}", f"{r['total_time']:.1f}"])
    print(f"CSV saved to: {csv_path}")

    best_config = {
        'batch_size': best['batch_size'],
        'learning_rate': best['learning_rate'],
        'activation': best['activation'],
        'epochs': EPOCHS,
        'momentum': MOMENTUM
    }
    with open(os.path.join(OUTPUT_DIR, 'best_config.json'), 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"Best config saved for Q4/Q5")

    print("\n" + "="*60)
    print(" Q3 COMPLETE!")
    print("="*60)
else:
    print(f"Only {len(all_results)}/24 experiments done. Complete all first.")

if len(all_results) == 24:
    sorted_results = sorted(all_results, key=lambda x: x['test_acc'], reverse=True)
    best = sorted_results[0]

    print(f"Retraining best model: batch={best['batch_size']}, lr={best['learning_rate']}, act={best['activation']}")

    trainloader, testloader = get_data_loaders(trainset, testset, batch_size=best['batch_size'])
    best_model = create_model(num_layers=2, activation=best['activation'])

    train_model(
        model=best_model,
        trainloader=trainloader,
        testloader=testloader,
        epochs=EPOCHS,
        learning_rate=best['learning_rate'],
        momentum=MOMENTUM,
        device=device,
        verbose=True
    )

    model_path = os.path.join(MODELS_DIR, 'q3_best_model.pth')
    save_model(best_model, model_path)
    print(f"\nBest model saved to: {model_path}")
else:
    print("Complete all 24 experiments first.")
