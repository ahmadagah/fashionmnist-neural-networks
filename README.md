# FashionMNIST Neural Network Experiments

Deep learning experiments on FashionMNIST using fully connected neural networks. This project explores network architecture, hyperparameter tuning, data quality effects, and model limitations.

## Project Overview

This project implements and evaluates fully connected (FC) neural networks on the FashionMNIST dataset (60,000 training images, 10,000 test images, 10 clothing categories). The experiments cover:

- **Q2**: Single-layer vs two-layer network comparison
- **Q3**: Hyperparameter grid search (24 configurations)
- **Q4**: Training with polluted/noisy labels
- **Q5**: Translation invariance testing via circular shifts
- **Q6**: Real-world photo classification

## Results Summary

| Experiment | Key Finding |
|------------|-------------|
| Q2 | Two-layer (84.25%) barely outperforms single-layer (84.13%) on this simple dataset |
| Q3 | Best: batch_size=1, lr=0.001, ReLU (89.42%). Worst: batch_size=1000, lr=1.0, ReLU (0%, exploded) |
| Q4 | 9% label noise actually improved test accuracy by 0.39% due to regularization effect |
| Q5 | 2-pixel shift drops accuracy from 84.25% to 57.17%, showing FC networks lack translation invariance |
| Q6 | Real shoe photo classified as "Ankle Boot" (61% confidence) after proper preprocessing |

## Project Structure

```
.
├── src/
│   ├── q2_single_vs_two_layer.py    # Architecture comparison
│   ├── q3_hyperparameter_grid.py    # 24 hyperparameter experiments
│   ├── q4_pollution.py              # Label noise experiment
│   ├── q5_circular_shift.py         # Translation invariance test
│   ├── q6_real_photo.py             # Real photo classification
│   └── utils/
│       ├── models.py                # Network definitions
│       ├── training.py              # Training loop
│       ├── data_loader.py           # FashionMNIST loading
│       └── evaluation.py            # Metrics and visualization
├── outputs/
│   ├── q2/                          # Results and training curves
│   ├── q3/                          # 24-experiment results
│   ├── q4/                          # Pollution experiment results
│   ├── q5/                          # Shift test results
│   └── q6/                          # Processed images and predictions
├── analysis/                        # Detailed analysis per question
├── answers.md                       # Consolidated answers to all questions
├── requirements.txt
└── shoe_original.png                # Real photo for Q6
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.8+ with PyTorch, torchvision, numpy, matplotlib, Pillow, and tqdm.

## Running Experiments

Each script runs independently and saves results to `outputs/`:

```bash
python src/q2_single_vs_two_layer.py   # ~10 min
python src/q3_hyperparameter_grid.py   # ~8 hours (24 experiments)
python src/q4_pollution.py             # ~1 hour
python src/q5_circular_shift.py        # ~1 min (uses saved model)
python src/q6_real_photo.py            # ~1 min (uses saved model)
```

Note: Q5 and Q6 require the model from Q2 to be trained first.

## Network Architecture

**Single-layer:** 784 (input) → 1024 (hidden, ReLU) → 10 (output)

**Two-layer:** 784 (input) → 1024 (hidden, ReLU) → 1024 (hidden, ReLU) → 10 (output)

Training uses SGD optimizer with CrossEntropy loss, running for 20 epochs.

## Detailed Answers

See [answers.md](answers.md) for complete analysis of each question including methodology, results tables, and explanations.
