import os
import logging
import numpy as np
import cv2
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Paths
MODEL_PATH = "/Users/macintoshhd/Downloads/MSE24/2025_semester_02/MLE501.9/final/model_dataset/PieceDetection/yolo_piece_classifier.pt"
image_dir = "/Users/macintoshhd/Downloads/MSE24/2025_semester_02/MLE501.9/final/templates"
threshold = 0.6

def load_yolo_model(model_path=MODEL_PATH):
    """Load a trained YOLOv8 classification model."""
    if not os.path.exists(model_path):
        logging.error(f"Model not found at: {model_path}")
        raise FileNotFoundError(model_path)
    
    logging.info(f"Loading YOLOv8 classification model from: {model_path}")
    model = YOLO(model_path)
    
    # Extract class index mapping
    class_names = list(model.names.values())
    class_indices = {name: idx for idx, name in enumerate(class_names)}
    
    return model, class_indices, class_names

def predict_piece(image_path, model, class_indices, threshold=0.6, debug=False):
    """Predict chess piece using YOLOv8 classification model."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            logging.warning(f"Image failed to load: {image_path}")
            return None

        img_resized = cv2.resize(img, (224, 224))  # Match training size
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        results = model.predict(img_rgb, imgsz=224, verbose=False, device="mps")
        probs = results[0].probs  # Classification mode uses .probs

        if probs is None or probs.top1conf < threshold:
            if debug:
                logging.debug(f"No confident prediction for {image_path}")
            return "board"

        top1_idx = int(probs.top1)
        class_name = model.names[top1_idx]

        if debug:
            logging.info(f"{os.path.basename(image_path)} → Predicted: {class_name}, Confidence: {probs.top1conf:.2f}")
        
        return class_name
    except Exception as e:
        logging.warning(f"Failed prediction on {image_path}: {e}")
        return None

def verify_model(image_dir, expected_labels, model, class_indices, threshold=0.6):
    """Evaluate model accuracy against known-labeled images."""
    correct = 0
    total = 0
    num_classes = len(class_indices)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    idx_map = {v: k for k, v in class_indices.items()}

    for filename, true_label in expected_labels.items():
        image_path = os.path.join(image_dir, filename)
        if not os.path.exists(image_path):
            logging.warning(f"Missing image: {image_path}")
            continue
        
        predicted = predict_piece(image_path, model, class_indices, threshold, debug=True)
        if predicted is None:
            continue
        
        true_idx = class_indices[true_label]
        pred_idx = class_indices.get(predicted, class_indices["board"])
        confusion_matrix[true_idx, pred_idx] += 1

        if predicted == true_label:
            correct += 1
        total += 1
        logging.info(f"{filename}: True={true_label}, Predicted={predicted}")
    
    accuracy = correct / total if total else 0
    logging.info(f"\nFinal Accuracy: {accuracy:.2%} ({correct}/{total})")
    logging.info("Confusion Matrix:")
    for i, class_name in idx_map.items():
        row = " ".join(str(x) for x in confusion_matrix[i])
        logging.info(f"{class_name:<6}: {row}")
    
    return accuracy, confusion_matrix

if __name__ == "__main__":
    expected_labels = {
        "B_g.png": "B", "K_g.png": "K", "N_g.png": "N", "P_g.png": "P",
        "Q_g.png": "Q", "R_g.png": "R", "b_g.png": "b", "k_g.png": "k",
        "n_g.png": "n", "p_g.png": "p", "q_g.png": "q", "r_g.png": "r",
        "empty_1.png": "board"
    }

    model, class_indices, class_names = load_yolo_model()
    accuracy, confusion = verify_model(image_dir, expected_labels, model, class_indices, threshold)
    print(f"✅ Model verified. Accuracy: {accuracy:.2%}")
