# Q3: Hyperparameter Experiments Analysis

## Results Table: 24 Hyperparameter Experiments (3 batch sizes × 4 learning rates × 2 activations)

| Batch Size | Learning Rate | Activation | Test Accuracy | Train Accuracy | Status |
|------------|---------------|------------|---------------|----------------|--------|
| 1 | 1.0 | relu | 10.00% | 10.00% | Failed |
| 1 | 1.0 | sigmoid | 10.00% | 9.92% | Failed |
| 1 | 0.1 | relu | 10.00% | 10.00% | Failed |
| 1 | 0.1 | sigmoid | 86.71% | 90.11% |  |
| 1 | 0.01 | relu | 88.80% | 93.98% |  |
| 1 | 0.01 | sigmoid | 88.65% | 91.71% |  |
| 1 | 0.001 | relu | 89.42% | 94.67% | **BEST** |
| 1 | 0.001 | sigmoid | 85.79% | 87.14% |  |
| 10 | 1.0 | relu | 0.00% | 0.00% | **WORST** |
| 10 | 1.0 | sigmoid | 87.08% | 89.02% |  |
| 10 | 0.1 | relu | 88.60% | 94.24% |  |
| 10 | 0.1 | sigmoid | 88.31% | 91.70% |  |
| 10 | 0.01 | relu | 88.79% | 94.68% |  |
| 10 | 0.01 | sigmoid | 85.52% | 87.25% |  |
| 10 | 0.001 | relu | 86.76% | 88.91% |  |
| 10 | 0.001 | sigmoid | 77.82% | 78.26% |  |
| 1000 | 1.0 | relu | 0.00% | 0.00% | **WORST** |
| 1000 | 1.0 | sigmoid | 83.34% | 84.47% |  |
| 1000 | 0.1 | relu | 85.98% | 87.92% |  |
| 1000 | 0.1 | sigmoid | 77.23% | 77.21% |  |
| 1000 | 0.01 | relu | 80.68% | 81.69% |  |
| 1000 | 0.01 | sigmoid | 54.83% | 52.80% |  |
| 1000 | 0.001 | relu | 67.62% | 67.64% |  |
| 1000 | 0.001 | sigmoid | 49.37% | 39.42% |  |

## Summary

- **Best configuration:** batch_size=1, lr=0.001, activation=relu → **89.42%** test accuracy
- **Worst configuration:** batch_size=1000, lr=1.0, activation=relu → **0.00%** test accuracy

## Q3 Analysis Paragraph: Best vs Worst Cases

The best performing configuration (batch_size=1, lr=0.001, ReLU) achieved 89.42% test accuracy, while the worst configuration (batch_size=1000, lr=1.0, ReLU) completely failed with 0% accuracy due to gradient explosion (NaN losses). The stark difference stems from the interaction between batch size and learning rate. With a learning rate of 1.0 and ReLU activation, the gradients become excessively large, causing weights to overflow to NaN values—a classic case of gradient explosion. ReLU is particularly susceptible because it has no upper bound on its output, unlike sigmoid which saturates between 0 and 1. This explains why lr=1.0 with sigmoid (83.34% at bs=1000, 87.08% at bs=10) still converged while ReLU failed catastrophically. The best configuration benefits from stochastic gradient descent in its purest form: batch_size=1 provides maximum gradient noise which acts as implicit regularization and helps escape local minima, while lr=0.001 ensures stable, gradual weight updates. ReLU outperforms sigmoid at lower learning rates because it avoids the vanishing gradient problem that plagues sigmoid in deep networks. Notably, smaller batch sizes consistently outperformed larger ones across most configurations, as they provide more frequent weight updates per epoch (60,000 updates with bs=1 vs 60 updates with bs=1000), allowing finer optimization of the loss landscape.

## Configuration

- Network: 2 fully-connected layers (784→1024→1024→10)
- Epochs: 20
- Momentum: 0.0
- Loss: CrossEntropy
- Optimizer: SGD
