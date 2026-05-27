import model


def test_digit_1_image():
    """Test that a digit 1 image is classified as '1'."""
    print("\n--- Digit 1 Image Test ---")
    try:
        with open('digit_1.png', 'rb') as img:
            image_bytes = img.read()
    except Exception as e:
        print(f"  Could not load digit_1.png: {e}")
        return None

    result = model.predict(image_bytes)
    print(f"  Predicted: {result['label']} (confidence: {result['confidence']:.2%})")
    print(f"  Details: {result['predictions']}")

    assert result["label"] == "1", (
        f"FAIL: Expected '1' but got '{result['label']}'"
    )
    print("  ✓ PASSED: Correctly classified as 1")
    return result


def test_digit_5_image():
    """Test that a digit 5 image is classified as '5'."""
    print("\n--- Digit 5 Image Test ---")
    try:
        with open('digit_5.png', 'rb') as img:
            image_bytes = img.read()
    except Exception as e:
        print(f"  Could not load digit_5.png: {e}")
        return None

    result = model.predict(image_bytes)
    print(f"  Predicted: {result['label']} (confidence: {result['confidence']:.2%})")
    print(f"  Details: {result['predictions']}")

    assert result["label"] == "5", (
        f"FAIL: Expected '5' but got '{result['label']}'"
    )
    print("  ✓ PASSED: Correctly classified as 5")
    return result




if __name__ == "__main__":
    print("=" * 50)
    print("  Handwritten Digit Classifier - Test Suite")
    print("=" * 50)

    passed = 0
    failed = 0

    # Test digit 1 classification
    try:
        if test_digit_1_image():
            passed += 1
    except AssertionError as e:
        print(f"  ✗ {e}")
        failed += 1
    except Exception as e:
        print(f"  ✗ Error: {e}")
        failed += 1

    # Test digit 5 classification
    try:
        if test_digit_5_image():
            passed += 1
    except AssertionError as e:
        print(f"  ✗ {e}")
        failed += 1
    except Exception as e:
        print(f"  ✗ Error: {e}")
        failed += 1

    print("\n" + "=" * 50)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed > 0:
        exit(1)

