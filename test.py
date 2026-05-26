import model
import urllib.request
import sys


def test_local_image():
    """Test with the local image.png file if it exists."""
    try:
        with open("image.png", "rb") as f:
            image_bytes = f.read()
        result = model.predict(image_bytes)
        print(f"[Local image.png] Predicted: {result['label']} "
              f"(confidence: {result['confidence']:.2%})")
        print(f"  Details: {result['predictions']}")
        return result
    except FileNotFoundError:
        print("[Local image.png] File not found, skipping.")
        return None


def test_dog_image():
    """Test that a dog image is classified as 'dog'."""
    print("\n--- Dog Image Test ---")
    try:
        with open('dog.png', 'rb') as img:
           image_bytes = img.read()
    except Exception as e:
        print(f"  Could not download dog image: {e}")
        return None

    result = model.predict(image_bytes)
    print(f"  Predicted: {result['label']} (confidence: {result['confidence']:.2%})")
    print(f"  Details: {result['predictions']}")

    assert result["label"] == "dog", (
        f"FAIL: Expected 'dog' but got '{result['label']}'"
    )
    print("  ✓ PASSED: Correctly classified as dog")
    return result


def test_cat_image():
    """Test that a cat image is classified as 'cat'."""
    print("\n--- Cat Image Test ---")
    try:
         with open('image.png', 'rb') as img:
           image_bytes = img.read()
    except Exception as e:
        print(f"  Could not download cat image: {e}")
        return None

    result = model.predict(image_bytes)
    print(f"  Predicted: {result['label']} (confidence: {result['confidence']:.2%})")
    print(f"  Details: {result['predictions']}")

    assert result["label"] == "cat", (
        f"FAIL: Expected 'cat' but got '{result['label']}'"
    )
    print("  ✓ PASSED: Correctly classified as cat")
    return result


if __name__ == "__main__":
    print("=" * 50)
    print("  Cat vs Dog Classifier - Test Suite")
    print("=" * 50)

    passed = 0
    failed = 0

    # Test local image
    test_local_image()

    # Test dog classification
    try:
        if test_dog_image():
            passed += 1
    except AssertionError as e:
        print(f"  \u2717 {e}")
        failed += 1
    except Exception as e:
        print(f"  \u2717 Error: {e}")
        failed += 1

    # Test cat classification
    try:
        if test_cat_image():
            passed += 1
    except AssertionError as e:
        print(f"  \u2717 {e}")
        failed += 1
    except Exception as e:
        print(f"  \u2717 Error: {e}")
        failed += 1

    print("\n" + "=" * 50)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed > 0:
        sys.exit(1)

