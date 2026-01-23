# Q2: Single Layer vs Two Layer Network Comparison

## Results Summary

| Metric | Single-Layer (784→1024→10) | Two-Layer (784→1024→1024→10) |
|--------|---------------------------|------------------------------|
| Test Accuracy | 84.13% | 84.25% |
| Train Accuracy | 85.62% | 85.87% |
| Final Loss | 0.411 | 0.401 |
| Parameters | 814,090 | 1,863,690 |

## Q2 Analysis Paragraph

The two-layer fully connected network achieved a test accuracy of 84.25%, marginally outperforming the single-layer network which achieved 84.13%, resulting in a difference of only 0.12%. Despite having more than twice the number of parameters (1,863,690 vs 814,090), the two-layer network provided only a minimal improvement in classification accuracy. This modest gain can be attributed to the additional hidden layer's ability to learn more complex, hierarchical feature representations from the flattened image input. The extra layer allows the network to compose simple features learned in the first layer into more abstract patterns in the second layer, which can be beneficial for distinguishing between visually similar classes such as "Shirt" and "T-shirt/top." However, the relatively small improvement suggests that for the FashionMNIST dataset with its 28x28 grayscale images, a single hidden layer with 1024 neurons already captures most of the discriminative information needed for classification. The similarity in train and test accuracies for both models (approximately 1.5% gap) indicates that neither network is significantly overfitting, suggesting that the current model capacity is appropriate for this dataset. The diminishing returns from adding depth highlight that simply increasing network complexity does not guarantee proportional improvements in performance, especially when the input data is relatively simple and low-dimensional.

## Configuration Used

- Batch size: 30
- Epochs: 20
- Learning rate: 0.001
- Momentum: 0.0
- Activation: ReLU
- Loss: CrossEntropy
- Optimizer: SGD
