"""
Q6 (Bonus): Real Photo Classification

Classify a real photo of a garment using the trained 2 FC NN.
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils import create_model, get_device, CLASSES

# Paths
MODEL_PATH = os.path.join(project_root, 'models', 'q2_two_layer.pth')
IMAGE_PATH = os.path.join(project_root, 'shoe_original.png')
OUTPUT_DIR = os.path.join(project_root, 'outputs', 'q6')


def grayscale_luminosity(img):
    """Standard luminosity method: 0.2126*R + 0.7152*G + 0.0722*B"""
    img_array = np.array(img)
    if len(img_array.shape) == 3:
        gray = 0.2126 * img_array[:,:,0] + 0.7152 * img_array[:,:,1] + 0.0722 * img_array[:,:,2]
    else:
        gray = img_array
    return gray.astype(np.uint8)


def grayscale_average(img):
    """Simple average: (R + G + B) / 3"""
    img_array = np.array(img)
    if len(img_array.shape) == 3:
        gray = np.mean(img_array[:,:,:3], axis=2)
    else:
        gray = img_array
    return gray.astype(np.uint8)


def grayscale_lightness(img):
    """Lightness method: (max(R,G,B) + min(R,G,B)) / 2"""
    img_array = np.array(img)
    if len(img_array.shape) == 3:
        max_val = np.max(img_array[:,:,:3], axis=2)
        min_val = np.min(img_array[:,:,:3], axis=2)
        gray = (max_val.astype(np.float32) + min_val.astype(np.float32)) / 2
    else:
        gray = img_array
    return gray.astype(np.uint8)


def preprocess_image(img_gray, invert=True):
    """
    Preprocess grayscale image for FashionMNIST model.

    FashionMNIST: dark background, light object
    Real photo: might be opposite, so we may need to invert
    """
    # Resize to 28x28
    img_pil = Image.fromarray(img_gray)
    img_resized = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized)

    # Optionally invert (FashionMNIST has light objects on dark background)
    if invert:
        img_array = 255 - img_array

    # Normalize like FashionMNIST: (x/255 - 0.5) / 0.5
    img_normalized = (img_array.astype(np.float32) / 255.0 - 0.5) / 0.5

    # Convert to tensor: (1, 1, 28, 28)
    tensor = torch.tensor(img_normalized).unsqueeze(0).unsqueeze(0)

    return tensor, img_array


def classify_image(model, tensor, device):
    """Run classification and return probabilities."""
    model.eval()
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)

    return probabilities.cpu().numpy()[0]


def run_q6_experiment():
    """Run Q6 real photo classification."""
    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    print(f"\nLoading trained model from: {MODEL_PATH}")
    model = create_model(num_layers=2, activation='relu')
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    print("Model loaded successfully!")

    # Load image
    print(f"\nLoading image from: {IMAGE_PATH}")
    img = Image.open(IMAGE_PATH)
    print(f"Original size: {img.size}")

    # Test different grayscale methods
    grayscale_methods = {
        'luminosity': grayscale_luminosity,
        'average': grayscale_average,
        'lightness': grayscale_lightness
    }

    results = {}

    print(f"\n{'='*60}")
    print("Q6: REAL PHOTO CLASSIFICATION")
    print(f"{'='*60}")

    for method_name, method_func in grayscale_methods.items():
        print(f"\n--- Grayscale method: {method_name.upper()} ---")

        # Convert to grayscale
        img_gray = method_func(img)

        # Test both normal and inverted
        for invert in [False, True]:
            invert_label = "inverted" if invert else "normal"

            # Preprocess
            tensor, processed_img = preprocess_image(img_gray, invert=invert)

            # Save processed image for reference
            processed_pil = Image.fromarray(processed_img)
            save_path = os.path.join(OUTPUT_DIR, f'processed_{method_name}_{invert_label}.png')
            processed_pil.save(save_path)

            # Classify
            probs = classify_image(model, tensor, device)

            # Get prediction
            pred_class = np.argmax(probs)
            pred_prob = probs[pred_class]

            print(f"\n  {invert_label.capitalize()}:")
            print(f"  Prediction: {CLASSES[pred_class]} ({pred_prob*100:.2f}%)")
            print(f"  All probabilities:")
            for i, (class_name, prob) in enumerate(zip(CLASSES, probs)):
                bar = '█' * int(prob * 20)
                print(f"    {i}: {class_name:12s} {prob*100:5.2f}% {bar}")

            # Store results
            key = f"{method_name}_{invert_label}"
            results[key] = {
                'grayscale_method': method_name,
                'inverted': invert,
                'predicted_class': int(pred_class),
                'predicted_label': CLASSES[pred_class],
                'confidence': float(pred_prob),
                'all_probabilities': {CLASSES[i]: float(p) for i, p in enumerate(probs)}
            }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Does grayscale method matter?")
    print(f"{'='*60}")

    print("\n| Method | Inverted | Prediction | Confidence |")
    print("|--------|----------|------------|------------|")
    for key, res in results.items():
        inv = "Yes" if res['inverted'] else "No"
        print(f"| {res['grayscale_method']:10s} | {inv:8s} | {res['predicted_label']:10s} | {res['confidence']*100:5.2f}% |")

    # Save results
    results_path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    print(f"Processed images saved to: {OUTPUT_DIR}/")

    return results


if __name__ == '__main__':
    run_q6_experiment()
