import onnxruntime as ort
import numpy as np
from PIL import Image
import io

session = ort.InferenceSession("./mnist-12.onnx")
input_name = session.get_inputs()[0].name

# MNIST digits 0-9, but we filter for 1-9
SUPPORTED_DIGITS = [str(i) for i in range(1, 10)]  # "1" to "9"


def preprocess(image_bytes):
    """Preprocess image bytes for MNIST digit recognition."""
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # MNIST standard size
    img = np.array(image).astype(np.float32) / 255.0
    
    # Normalize to [-1, 1] range (common for MNIST)
    img = (img - 0.5) / 0.5
    
    # Reshape to [1, 1, 28, 28] for ONNX model input
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    
    return img


def softmax(logits):
    """Compute softmax probabilities from raw logits."""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)


def predict(image_bytes):
    """
    Classify a handwritten digit image (1-9).

    Returns a dict with:
      - label: digit "1" to "9"
      - confidence: float between 0 and 1
      - predictions: list of {label, confidence} for all digits 1-9
    """
    img = preprocess(image_bytes)

    outputs = session.run(None, {input_name: img})
    logits = outputs[0][0]

    probs = softmax(logits)

    # Get predictions for digits 1-9 (skip 0 at index 0)
    digit_predictions = []
    for digit_idx in range(1, 10):
        digit_str = str(digit_idx)
        confidence = float(probs[digit_idx])
        digit_predictions.append({"label": digit_str, "confidence": round(confidence, 4)})

    # Find the top prediction
    top_idx = np.argmax(probs[1:10]) + 1  # Add 1 to account for skipping digit 0
    top_label = str(top_idx)
    top_confidence = float(probs[top_idx])

    return {
        "label": top_label,
        "confidence": round(top_confidence, 4),
        "predictions": digit_predictions,
    }
