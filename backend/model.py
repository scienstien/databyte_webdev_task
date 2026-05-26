import onnxruntime as ort
import numpy as np
from PIL import Image
import io

session = ort.InferenceSession("./mobilenetv2-7.onnx")
input_name = session.get_inputs()[0].name

# ImageNet class index ranges for dogs and cats
# Dogs: indices 151-268 (various dog breeds)
# Cats: indices 281-285 (tabby, tiger cat, Persian, Siamese, Egyptian cat)
DOG_INDICES = list(range(151, 269))
CAT_INDICES = list(range(281, 286))

SUPPORTED_LABELS = ["cat", "dog"]


def preprocess(image_bytes):
    """Preprocess image bytes for MobileNetV2 inference."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    img = np.array(image).astype(np.float32) / 255.0
    # Normalize with ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img


def softmax(logits):
    """Compute softmax probabilities from raw logits."""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)


def predict(image_bytes):
    """
    Classify an image as 'cat' or 'dog'.

    Returns a dict with:
      - label: 'cat' or 'dog'
      - confidence: float between 0 and 1
      - predictions: list of {label, confidence} for both classes
    """
    img = preprocess(image_bytes)

    outputs = session.run(None, {input_name: img})
    logits = outputs[0][0]

    probs = softmax(logits)

    # Sum probabilities across all dog-related and cat-related ImageNet classes
    dog_confidence = float(np.sum(probs[DOG_INDICES]))
    cat_confidence = float(np.sum(probs[CAT_INDICES]))

    # Normalize so cat + dog = 1.0 for clearer comparison
    total = dog_confidence + cat_confidence
    if total > 0:
        dog_confidence /= total
        cat_confidence /= total
    else:
        dog_confidence = 0.5
        cat_confidence = 0.5

    # Determine the top prediction
    if dog_confidence >= cat_confidence:
        label = "dog"
        confidence = dog_confidence
    else:
        label = "cat"
        confidence = cat_confidence

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "predictions": [
            {"label": "dog", "confidence": round(dog_confidence, 4)},
            {"label": "cat", "confidence": round(cat_confidence, 4)},
        ],
    }
