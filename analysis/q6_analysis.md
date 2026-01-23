# Q6 (Bonus): Real Photo Classification Analysis

## Experiment Setup

- **Model:** Trained 2-layer FC network from Q2 (784→1024→1024→10, ReLU)
- **Input image:** Real photo of a Nike sneaker taken with iPhone
- **Original image size:** 5712 × 4284 pixels
- **Processing:** Resize to 28×28, convert to grayscale

## Methodology

### 1. Size Handling
- Original iPhone photo: 5712 × 4284 pixels (high resolution)
- Resized to 28×28 pixels using Lanczos resampling
- Significant detail loss is unavoidable at this scale

### 2. Grayscale Conversion Methods Tested
Three different grayscale conversion formulas were tested:

| Method | Formula |
|--------|---------|
| Luminosity | 0.2126×R + 0.7152×G + 0.0722×B (standard) |
| Average | (R + G + B) / 3 |
| Lightness | (max(R,G,B) + min(R,G,B)) / 2 |

### 3. Inversion Testing
FashionMNIST images have **light objects on dark backgrounds**. Real photos typically have dark objects on light backgrounds, so we tested both:
- Normal: as-is grayscale
- Inverted: 255 - grayscale value

## Results Summary

| Grayscale Method | Inverted | Prediction | Confidence |
|------------------|----------|------------|------------|
| Luminosity | No | Bag | 26.16% |
| Luminosity | Yes | Ankle Boot | 60.82% |
| Average | No | Bag | 25.88% |
| Average | Yes | Ankle Boot | 61.36% |
| Lightness | No | Bag | 25.74% |
| Lightness | Yes | Ankle Boot | 61.24% |

### Best Prediction (Inverted, Any Method)
- **1st:** Ankle Boot (60-61%)
- **2nd:** Sneaker (16%)
- **3rd:** Sandal (9%)

## Q6 Analysis Paragraph

The real photo classification experiment reveals important insights about applying trained models to real-world data. The grayscale conversion method made virtually no difference in classification results—all three methods (luminosity, average, lightness) produced nearly identical predictions with less than 1% variation in confidence scores. However, image inversion proved critical: without inversion, the model predicted "Bag" with low confidence (~26%), while inverted images correctly identified footwear, predicting "Ankle Boot" (61%) with "Sneaker" as second choice (16%). This dramatic difference occurs because FashionMNIST training images feature light-colored objects on dark backgrounds, opposite to typical real-world photos. The model predicted "Ankle Boot" rather than "Sneaker" likely because the Nike shoe in the photo has a higher ankle cut similar to boots in the training set. The relatively high confidence (61%) on inverted images demonstrates that despite the extreme downsampling from 5712×4284 to 28×28 pixels, the model can still extract meaningful shape features. The consistent second-place ranking of "Sneaker" (16%) shows the model recognizes the image as footwear but struggles to distinguish between shoe subcategories at such low resolution. This experiment underscores the importance of matching input preprocessing to training data characteristics when deploying models on real-world images.

## Key Findings

### Does grayscale method make a difference?
**No.** All three methods produced nearly identical results (<1% difference).

### What matters most?
**Inversion.** Matching the light-on-dark format of FashionMNIST is critical.

### Classification accuracy
- Predicted: Ankle Boot (actual: Sneaker)
- Close but not exact—shoe subcategories are hard to distinguish at 28×28

## Processed Images

The 28×28 processed images are saved at:
- `outputs/q6/processed_luminosity_inverted.png`
- `outputs/q6/processed_average_inverted.png`
- `outputs/q6/processed_lightness_inverted.png`
- (and normal versions)

## Conclusion

For classifying real photos with a FashionMNIST-trained model:
1. Grayscale method doesn't matter—use any standard conversion
2. Invert the image to match training data format (light object, dark background)
3. Expect some misclassification between similar categories due to low resolution
