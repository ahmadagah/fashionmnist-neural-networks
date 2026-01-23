import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    load_fashion_mnist,
    create_model,
    get_device,
    count_parameters,
    save_model,
    train_model,
    evaluate_model,
    plot_training_history,
    plot_comparison,
    save_results,
    CLASSES
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'q2')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

device = get_device()
print(f"Using device: {device}")

BATCH_SIZE = 30
EPOCHS = 20
LEARNING_RATE = 0.001
MOMENTUM = 0.0
ACTIVATION = 'relu'

print("Configuration:")
print(f"  Batch size:     {BATCH_SIZE}")
print(f"  Epochs:         {EPOCHS}")
print(f"  Learning rate:  {LEARNING_RATE}")
print(f"  Momentum:       {MOMENTUM}")
print(f"  Activation:     {ACTIVATION}")

print(f"Loading FashionMNIST data (batch_size={BATCH_SIZE})...")

trainloader, testloader, trainset, testset = load_fashion_mnist(
    batch_size=BATCH_SIZE
)

print(f"Training samples: {len(trainset):,}")
print(f"Test samples:     {len(testset):,}")
print(f"Batches per epoch: {len(trainloader):,}")
print(f"Classes: {CLASSES}")

print("=" * 60)
print(" SINGLE-LAYER NETWORK (784 -> 1024 -> 10)")
print("=" * 60)

model_1layer = create_model(num_layers=1, activation=ACTIVATION)

print(f"\nModel architecture:")
print(model_1layer)
print(f"\nTotal parameters: {count_parameters(model_1layer):,}")

print("\nTraining single-layer network...")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Momentum: {MOMENTUM}")
print(f"  Epochs: {EPOCHS}")
print("-" * 60)

history_1layer = train_model(
    model=model_1layer,
    trainloader=trainloader,
    testloader=testloader,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    momentum=MOMENTUM,
    device=device,
    verbose=True
)

acc_1layer = evaluate_model(model_1layer, testloader, device)

print("\n" + "=" * 60)
print(" SINGLE-LAYER RESULTS")
print("=" * 60)
print(f"  Final test accuracy:  {acc_1layer:.2f}%")
print(f"  Final train accuracy: {history_1layer['train_acc'][-1]:.2f}%")
print(f"  Final train loss:     {history_1layer['train_loss'][-1]:.4f}")

plot_training_history(
    history_1layer,
    save_path=os.path.join(OUTPUT_DIR, 'single_layer_history.png'),
    title=f'Single-Layer Network (lr={LEARNING_RATE}, epochs={EPOCHS})'
)
print(f"\nPlot saved to: {os.path.join(OUTPUT_DIR, 'single_layer_history.png')}")

save_model(model_1layer, os.path.join(MODELS_DIR, 'q2_single_layer.pth'))
print(f"Model saved to: {os.path.join(MODELS_DIR, 'q2_single_layer.pth')}")

print("=" * 60)
print(" TWO-LAYER NETWORK (784 -> 1024 -> 1024 -> 10)")
print("=" * 60)

model_2layer = create_model(num_layers=2, activation=ACTIVATION)

print(f"\nModel architecture:")
print(model_2layer)
print(f"\nTotal parameters: {count_parameters(model_2layer):,}")

print("\nTraining two-layer network...")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Momentum: {MOMENTUM}")
print(f"  Epochs: {EPOCHS}")
print("-" * 60)

history_2layer = train_model(
    model=model_2layer,
    trainloader=trainloader,
    testloader=testloader,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    momentum=MOMENTUM,
    device=device,
    verbose=True
)

acc_2layer = evaluate_model(model_2layer, testloader, device)

print("\n" + "=" * 60)
print(" TWO-LAYER RESULTS")
print("=" * 60)
print(f"  Final test accuracy:  {acc_2layer:.2f}%")
print(f"  Final train accuracy: {history_2layer['train_acc'][-1]:.2f}%")
print(f"  Final train loss:     {history_2layer['train_loss'][-1]:.4f}")

plot_training_history(
    history_2layer,
    save_path=os.path.join(OUTPUT_DIR, 'two_layer_history.png'),
    title=f'Two-Layer Network (lr={LEARNING_RATE}, epochs={EPOCHS})'
)
print(f"\nPlot saved to: {os.path.join(OUTPUT_DIR, 'two_layer_history.png')}")

save_model(model_2layer, os.path.join(MODELS_DIR, 'q2_two_layer.pth'))
print(f"Model saved to: {os.path.join(MODELS_DIR, 'q2_two_layer.pth')}")

print("\n" + "=" * 60)
print(" COMPARISON: SINGLE-LAYER vs TWO-LAYER")
print("=" * 60)

print(f"\n{'Metric':<25} {'1-Layer':<15} {'2-Layer':<15} {'Difference':<15}")
print("-" * 70)
print(f"{'Test Accuracy':<25} {acc_1layer:<15.2f} {acc_2layer:<15.2f} {acc_2layer - acc_1layer:+.2f}")
print(f"{'Train Accuracy':<25} {history_1layer['train_acc'][-1]:<15.2f} {history_2layer['train_acc'][-1]:<15.2f} {history_2layer['train_acc'][-1] - history_1layer['train_acc'][-1]:+.2f}")
print(f"{'Final Loss':<25} {history_1layer['train_loss'][-1]:<15.4f} {history_2layer['train_loss'][-1]:<15.4f} {history_2layer['train_loss'][-1] - history_1layer['train_loss'][-1]:+.4f}")
print(f"{'Parameters':<25} {count_parameters(model_1layer):<15,} {count_parameters(model_2layer):<15,}")

better = "Two-Layer" if acc_2layer > acc_1layer else "Single-Layer"
print(f"\nBetter model: {better} (+{abs(acc_2layer - acc_1layer):.2f}% accuracy)")

plot_comparison(
    histories=[history_1layer, history_2layer],
    labels=['1-Layer (784→1024→10)', '2-Layer (784→1024→1024→10)'],
    save_path=os.path.join(OUTPUT_DIR, 'comparison.png'),
    title='Q2: Single Layer vs Two Layer Comparison'
)
print(f"Comparison plot saved to: {os.path.join(OUTPUT_DIR, 'comparison.png')}")

results = {
    'config': {
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'momentum': MOMENTUM,
        'activation': ACTIVATION
    },
    'single_layer': {
        'parameters': count_parameters(model_1layer),
        'final_train_loss': history_1layer['train_loss'][-1],
        'final_train_acc': history_1layer['train_acc'][-1],
        'final_test_acc': acc_1layer,
        'history': history_1layer
    },
    'two_layer': {
        'parameters': count_parameters(model_2layer),
        'final_train_loss': history_2layer['train_loss'][-1],
        'final_train_acc': history_2layer['train_acc'][-1],
        'final_test_acc': acc_2layer,
        'history': history_2layer
    },
    'comparison': {
        'accuracy_difference': acc_2layer - acc_1layer,
        'better_model': '2-layer' if acc_2layer > acc_1layer else '1-layer'
    }
}

save_results(results, os.path.join(OUTPUT_DIR, 'results.json'))

print("\n" + "=" * 60)
print(" Q2 COMPLETE!")
print("=" * 60)
print("\nAll outputs saved to:")
print(f"  - {os.path.join(OUTPUT_DIR, 'results.json')}")
print(f"  - {os.path.join(OUTPUT_DIR, 'single_layer_history.png')}")
print(f"  - {os.path.join(OUTPUT_DIR, 'two_layer_history.png')}")
print(f"  - {os.path.join(OUTPUT_DIR, 'comparison.png')}")
print(f"  - {os.path.join(MODELS_DIR, 'q2_single_layer.pth')}")
print(f"  - {os.path.join(MODELS_DIR, 'q2_two_layer.pth')}")
