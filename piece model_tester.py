import os
import onnxruntime as ort
import numpy as np
import cv2
import logging

logging.basicConfig(level=logging.INFO)

# Folder containing images to test
TEST_FOLDER = "/Users/macintoshhd/Downloads/MSE24/2025_semester_02/MLE501.9/final/templates"
IMAGE_SIZE = 224  # Expected input size for YOLOv8 classification

# Label mapping from class index to label name (adjust to match your training set)
idx2label = [
    "B", "K", "N", "P", "Q", "R",   # 0–5
    "b", "k", "n", "p", "q", "r"    # 6–11
]

# Fix label map (based on your rules)
fix_map = {
    "B": "b", "K": "board", "N": "k", "P": "n", "Q": "p", "R": "q",
    "b": "B", "board": "r", "k": "K", "n": "N", "p": "P", "q": "Q", "r": "R"
}

# Load ONNX model
session = ort.InferenceSession("/Users/macintoshhd/Downloads/MSE24/2025_semester_02/MLE501.9/final/model_dataset/PieceDetection/best.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
logging.info("Model loaded.")

# Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# Run classification on images
for fname in os.listdir(TEST_FOLDER):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    path = os.path.join(TEST_FOLDER, fname)
    img_raw = cv2.imread(path)
    img = cv2.resize(img_raw, (IMAGE_SIZE, IMAGE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[None]  # NCHW
    img = np.ascontiguousarray(img)

    outputs = session.run(None, {input_name: img})[0]
    logits = outputs[0]
    probs = softmax(logits)

    top_idx = int(np.argmax(probs))
    conf = float(probs[top_idx])
    label = idx2label[top_idx]
    fixed_label = fix_map.get(label, label)

    logging.info(f"{fname} → {fixed_label} (raw: {label}, confidence: {conf:.2f})")
    logging.debug(f"  Logits: {logits}")
    logging.debug(f"  Probabilities: {probs}")
