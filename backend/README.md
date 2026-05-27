# � Backend — Handwritten Digit Classifier (1-9)

A lightweight ONNX-based image classifier that recognizes handwritten digits from **1 to 9**, powered by **MNIST model** and served via **Uvicorn**.

---

## 📁 Folder Structure

```
backend/
├── model.py            # Core inference logic (preprocessing + ONNX prediction)
├── test.py             # Test suite for handwritten digit classification
├── requirements.txt    # All Python dependencies
├── mnist-12.onnx       # Pre-trained MNIST ONNX model (included in repository)
├── digit_1.png         # Sample digit test images (included in repository)
├── digit_5.png         # Sample digit test images (included in repository)
└── .gitignore          # Excludes venv and bytecode
```

> **Note:** The `.onnx` model file and test images are included in this repository and will be cloned automatically.

---

## ✅ Prerequisites

Before you begin, make sure the following are installed on your machine:

| Tool | Version | Check Command |
|------|---------|---------------|
| **Python** | 3.9 or higher | `python --version` |
| **Git Bash** | Any recent version | Comes with [Git for Windows](https://git-scm.com/download/win) |
| **pip** | Bundled with Python | `pip --version` |

> ⚠️ **Windows users:** All commands below are written for **Git Bash** (not PowerShell or CMD).  
> Open Git Bash by right-clicking in any folder and selecting **"Git Bash Here"**.

---

## 🚀 Setup Guide

### Step 1 — Clone the Repository

If you haven't already cloned the project, open **Git Bash** and run:

```bash
git clone https://github.com/scienstien/databyte_webdev_task
cd databyte_task/backend
```

> Replace `<YOUR_REPOSITORY_LINK_HERE>` with the actual GitHub/GitLab URL of this project.

If you already have the project on your machine, just navigate into the backend folder:

```bash
cd /c/Users/<your-username>/OneDrive/Documents/Desktop/databyte_task/backend
```

---

### Step 2 — Create a Virtual Environment
**Tutorial** https://youtu.be/Y21OR1OPC9A
A **virtual environment (venv)** isolates this project's dependencies from your system Python, preventing version conflicts.

Run this command inside the `backend/` directory in **Git Bash**:

```bash
python -m venv venv
```

This creates a new folder called `venv/` inside `backend/`. It contains a private copy of Python and pip.

> 💡 The `venv/` folder is listed in `.gitignore` and will **never** be committed to Git — this is intentional.

---

### Step 3 — Activate the Virtual Environment

You must activate the venv **every time** you open a new terminal before running any Python commands.

**In Git Bash:**

```bash
source venv/Scripts/activate
```

When the venv is active, your terminal prompt will change to show `(venv)` at the beginning, like this:

```
(venv) user@machine ~/databyte_task/backend
$
```

> ⚠️ **Important:** If you see an error like `bash: venv/Scripts/activate: No such file or directory`, make sure you are inside the `backend/` directory and that Step 2 completed successfully.

---

### Step 4 — Install All Dependencies

With the venv **activated**, install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---------|---------|
| `onnxruntime` | Runs the MobileNetV2 `.onnx` model |
| `numpy` | Array and numerical operations |
| `pillow` | Image loading and preprocessing |
| `uvicorn` | ASGI server (for serving the API) |
| `click` | CLI utilities (uvicorn dependency) |
| `colorama` | Cross-platform colored terminal output |
| `flatbuffers` | Serialization (onnxruntime dependency) |
| `h11` | HTTP/1.1 protocol library |
| `packaging` | Version parsing utilities |
| `protobuf` | Protocol Buffers (onnxruntime dependency) |

Installation typically takes **1–3 minutes** depending on your internet speed.

To verify that packages were installed correctly:

```bash
pip list
```

---

### Step 5 — Verify Files Are Present

The following files are **included in the repository** and will be cloned automatically:

| File | Description |
|------|-------------|
| `mnist-12.onnx` | Pre-trained MNIST model weights |
| `digit_1.png` | Sample handwritten digit images for testing |
| `digit_5.png` | Sample handwritten digit images for testing |

> These files are already included when you clone the repository. No additional setup needed.

---

## ▶️ Running the Code

> Make sure your virtual environment is **activated** (`source venv/Scripts/activate`) before running any of the following commands.

### Run the Test Suite

The `test.py` script runs classification tests on handwritten digit images, and prints results with pass/fail status:

```bash
python test.py
```

**Expected output:**

```
==================================================
  Handwritten Digit Classifier - Test Suite
==================================================

--- Digit 1 Image Test ---
  Predicted: 1 (confidence: 95.23%)
  Details: [{'label': '1', 'confidence': 0.9523}, {'label': '2', 'confidence': 0.02}, ...]
  ✓ PASSED: Correctly classified as 1

--- Digit 5 Image Test ---
  Predicted: 5 (confidence: 92.15%)
  Details: [{'label': '5', 'confidence': 0.9215}, {'label': '3', 'confidence': 0.015}, ...]
  ✓ PASSED: Correctly classified as 5

==================================================
  Results: 2 passed, 0 failed
==================================================
```

---

### Use the Model in Your Own Script

You can import `model.py` directly and call `predict()` with raw image bytes:

```python
import model

with open("handwritten_digit.png", "rb") as f:
    image_bytes = f.read()

result = model.predict(image_bytes)

print(result["label"])       # "1" to "9"
print(result["confidence"])  # float between 0.0 and 1.0
print(result["predictions"]) # list with scores for all digits 1-9
```

---

## 🔄 Deactivating the Virtual Environment

When you're done working, deactivate the venv with:

```bash
deactivate
```

Your prompt will return to normal (no `(venv)` prefix).

---

## 🔁 Returning to the Project Later

Every time you come back to work on the backend, follow these two steps:

1. Open **Git Bash** and navigate to the `backend/` folder:
   ```bash
   cd /c/Users/<your-username>/OneDrive/Documents/Desktop/databyte_task/backend
   ```

2. Activate the virtual environment:
   ```bash
   source venv/Scripts/activate
   ```

You do **not** need to run `pip install` again unless `requirements.txt` has changed.

---

## 🛠️ Troubleshooting

### `python` is not recognized
Make sure Python is installed and added to your PATH. Run `python --version` in Git Bash. If it fails, try `python3 --version` and substitute `python3` in all commands above.

### `source venv/Scripts/activate` gives an error
- Confirm you are in the `backend/` directory: run `pwd`
- Confirm the `venv/` folder exists: run `ls`
- If missing, re-run `python -m venv venv`

### `ModuleNotFoundError: No module named 'onnxruntime'`
The virtual environment is not activated. Run `source venv/Scripts/activate` first, then retry.

### `FileNotFoundError: mobilenetv2-7.onnx`
The ONNX model file is missing. Place `mobilenetv2-7.onnx` inside the `backend/` folder.

### `AssertionError: FAIL: Expected 'dog' but got 'cat'`
The model misclassified the test image. Verify that `dog.png` contains a clear dog photo with good lighting. The model uses ImageNet class probabilities and may struggle with unusual angles or low-quality images.

---

## 📦 Updating Dependencies

If you add new packages, update `requirements.txt` so others can reproduce your environment:

```bash
pip freeze > requirements.txt
```

Then commit the updated file:

```bash
git add requirements.txt
git commit -m "chore: update requirements.txt"
```

---

## 📝 Notes

- The `venv/` folder is intentionally excluded from Git via `.gitignore`. Never commit it.
- The `.onnx` model and image files are also excluded — share them via a file link, cloud storage, or attach them to a release.
- All inference runs **locally** — no internet connection is needed after setup.
