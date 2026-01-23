# Q5: Circular Shift Experiment Analysis

## Experiment Setup

- **Model:** Trained 2-layer FC network from Q2 (784→1024→1024→10, ReLU)
- **Test:** Apply circular shifts to test images, evaluate accuracy
- **No retraining:** Same model, just different test data

## Circular Shift Operations

1. **Right shift by 2 pixels:** Rightmost 2 pixels in each row wrap to become leftmost 2 pixels
2. **Down shift by 2 pixels:** Applied to already right-shifted images; bottom 2 rows wrap to top

```
Original:        Right shift:     Right+Down shift:
[A B C ... Y Z]  [Y Z A B C ...]  (rows also wrapped)
```

## Results Summary

| Test Data | Accuracy | Drop from Original |
|-----------|----------|-------------------|
| Original | 84.25% | - |
| Right shift (2px) | 57.17% | -27.08% |
| Right + Down shift | 47.83% | -36.42% |

## Q5 Analysis Paragraph

The circular shift experiment dramatically demonstrates the lack of translation invariance in fully-connected neural networks. A mere 2-pixel rightward shift caused accuracy to plummet from 84.25% to 57.17%—a devastating 27.08% drop. Adding a 2-pixel downward shift further degraded performance to 47.83%, representing a total accuracy loss of 36.42% from the original. This severe degradation occurs because FC networks learn to associate specific pixel positions with features. When the image shifts, the same pixel values now appear at different positions in the flattened 784-dimensional input vector, completely disrupting the learned weight patterns. For example, if the network learned that high values at position (14, 14) indicate a shoe, shifting moves those values to position (14, 16), where the network expects different features. In contrast, Convolutional Neural Networks (CNNs) achieve translation invariance through weight sharing (the same filter slides across all positions) and pooling layers (which aggregate features over spatial regions). This experiment illustrates why CNNs became the standard for image classification: real-world images often have objects at varying positions, and a classifier must recognize objects regardless of their location. The FC network's sensitivity to tiny shifts makes it impractical for real applications where exact object positioning cannot be guaranteed.

## Key Observations

1. **No translation invariance:** FC networks are extremely sensitive to spatial shifts
2. **Cumulative degradation:** Each shift compounds the accuracy loss
3. **Why CNNs are better:** Weight sharing and pooling provide translation invariance
4. **Practical implication:** FC networks require perfectly centered inputs to perform well

## Comparison to Q2/Q3 Results

- Q2 original test accuracy: 84.25%
- After 2px shift: 57.17% (32% relative drop)
- After 4px total shift: 47.83% (43% relative drop)

The network retains less than half its original accuracy with minimal shifts, highlighting a fundamental limitation of FC architectures for image tasks.
