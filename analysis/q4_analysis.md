# Q4: Pollution Experiment Analysis

## Experiment Setup

- **Best parameters from Q3:** batch_size=1, lr=0.001, activation=ReLU
- **Pollution:** 9% of training labels corrupted (1% of each class relabeled to each of the other 9 classes)
- **Total samples relabeled:** 5,400 out of 60,000 (9%)
- **Test dataset:** NOT polluted (clean)

## Results Summary

| Metric | Clean Training | Polluted Training |
|--------|---------------|-------------------|
| Test Accuracy | 88.59% | 88.98% |
| Train Accuracy | 94.81% | 85.84% |
| Final Train Loss | 0.140 | 0.555 |
| Training Time | 71.2 min | 70.1 min |

**Accuracy Difference:** Polluted model performed **0.39% BETTER** than clean model on test data.

## Q4 Analysis Paragraph

Surprisingly, the model trained on polluted data (88.98% test accuracy) slightly outperformed the model trained on clean data (88.59% test accuracy) by 0.39%. This counterintuitive result can be explained by examining the training dynamics. The clean model achieved 94.81% training accuracy but only 88.59% test accuracy, showing a 6.22% generalization gap indicative of overfitting—the model memorized the training data rather than learning generalizable patterns. In contrast, the polluted model achieved only 85.84% training accuracy (limited by the 9% incorrect labels) but 88.98% test accuracy, actually generalizing better than it fit the training data. This phenomenon occurs because the label noise acts as implicit regularization: the model cannot perfectly memorize the training data when 9% of labels are incorrect, forcing it to learn more robust, generalizable features instead. The higher training loss (0.555 vs 0.140) confirms the model struggled to fit the noisy labels, but this "struggle" prevented overfitting. This result demonstrates that moderate label noise can sometimes improve generalization, similar to techniques like label smoothing used in modern deep learning. However, this benefit has limits—excessive pollution would eventually degrade performance as the signal-to-noise ratio becomes too low for the model to learn meaningful patterns.

## Key Observations

1. **Overfitting in clean model:** Train acc (94.81%) >> Test acc (88.59%) shows memorization
2. **Regularization effect:** Pollution prevented overfitting, improving generalization
3. **Training loss:** Higher loss on polluted data (0.555 vs 0.140) indicates model couldn't fit noisy labels
4. **Robust features:** Polluted model learned features that transfer better to clean test data

## Configuration

- Network: 2 FC layers (784→1024→1024→10)
- Epochs: 20
- Batch size: 1
- Learning rate: 0.001
- Momentum: 0.0
- Activation: ReLU
- Optimizer: SGD
- Loss: CrossEntropy
