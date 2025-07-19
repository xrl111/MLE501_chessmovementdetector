import os
import logging
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = "/Users/macintoshhd/Downloads/MSE24/2025_semester_02/MLE501.9/final/model_dataset/PieceDetection/yolo_piece_classifier.pt"

def load_yolo_model(model_path=MODEL_PATH):
    """Load the pre-trained YOLOv8 classification model and verify class names."""
    piece_classes = ['b', 'board', 'k', 'n', 'p', 'q', 'B', 'K', 'N', 'P', 'Q', 'R', 'r']
    
    if not os.path.exists(model_path):
        logger.error(f"Pre-trained model not found at {model_path}")
        raise ValueError(f"Pre-trained model not found at {model_path}")
    
    logger.info(f"Loading pre-trained YOLOv8 model from {model_path}")
    try:
        model = YOLO(model_path)
        model_classes = list(model.names.values())
        if model_classes != piece_classes:
            logger.warning(f"Model classes {model_classes} differ from expected {piece_classes}")
            logger.info("Using model classes")
            piece_classes = model_classes
        class_indices = {piece: idx for idx, piece in enumerate(piece_classes)}
        return model, class_indices, piece_classes
    except Exception as e:
        logger.error(f"Failed to load YOLOv8 model: {e}")
        raise

def predict_piece(image_path, model, class_indices, threshold=0.6, debug=False):
    """Predict the chess piece in a single-piece image using classification."""
    try:
        square_img = cv2.imread(image_path)
        if square_img is None or square_img.size == 0:
            logger.warning(f"Failed to load image: {image_path}")
            return None

        square_resized = cv2.resize(square_img, (224, 224))  # Match training imgsz
        square_rgb = cv2.cvtColor(square_resized, cv2.COLOR_BGR2RGB)
        
        results = model.predict(square_rgb, imgsz=224, conf=threshold, verbose=False, device="mps")
        probs = results[0].probs
        
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()
        predicted_class = class_indices.get(top1_idx, "board")
        
        if top1_conf < threshold:
            if debug:
                logger.debug(f"Confidence {top1_conf:.2f} below threshold {threshold} for {image_path}")
            return "board"
        
        if debug:
            logger.info(f"{image_path}: Predicted {predicted_class} (Confidence: {top1_conf:.2f})")
            plt.figure(figsize=(5, 5))
            plt.imshow(square_rgb)
            plt.title(f"Predicted: {predicted_class} ({top1_conf:.2f})")
            plt.axis('off')
            save_path = os.path.join("./test_results", f"result_{os.path.basename(image_path)}")
            os.makedirs("./test_results", exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        
        return predicted_class
    except Exception as e:
        logger.warning(f"Prediction failed for {image_path}: {e}")
        return None

def verify_model(image_dir, expected_labels, model, class_indices, threshold=0.6):
    """Verify model performance against a set of single-piece images with known labels."""
    correct = 0
    total = 0
    confusion_matrix = np.zeros((len(class_indices), len(class_indices)), dtype=int)
    class_names = list(class_indices.keys())
    
    os.makedirs("./test_results", exist_ok=True)
    
    for image_file, true_label in expected_labels.items():
        image_path = os.path.join(image_dir, image_file)
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            continue
        
        predicted_label = predict_piece(image_path, model, class_indices, threshold, debug=True)
        if predicted_label is None:
            continue
        
        true_idx = class_indices.get(true_label, class_indices["board"])
        pred_idx = class_indices.get(predicted_label, class_indices["board"])
        confusion_matrix[true_idx][pred_idx] += 1
        
        if predicted_label == true_label:
            correct += 1
        total += 1
        logger.info(f"{image_file}: True={true_label}, Predicted={predicted_label}")
    
    accuracy = correct / total if total > 0 else 0
    logger.info(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    logger.info("Confusion Matrix:")
    for i in range(len(class_names)):
        row = [f"{confusion_matrix[i][j]}" for j in range(len(class_names))]
        logger.info(f"{class_names[i]}: {' '.join(row)}")
    
    logger.info(f"Test results saved to ./test_results")
    
    return accuracy, confusion_matrix

if __name__ == "__main__":
    image_dir = "/Users/macintoshhd/Downloads/MSE24/2025_semester_02/MLE501.9/final/templates"
    threshold = 0.6
    
    # Expected labels for single-piece images (updated to match remapped labels)
    expected_labels = {
        "b_g.png": "b",  # BlackBishop (was B)
        "k_g.png": "k",  # BlackKing (was N)
        "n_g.png": "n",  # BlackKnight (was P)
        "p_g.png": "p",  # BlackPawn (was Q)
        "q_g.png": "q",  # BlackQueen (was R)
        "B_g.png": "B",  # WhiteBishop (was b)
        "K_g.png": "K",  # WhiteKing (was k)
        "N_g.png": "N",  # WhiteKnight (was n)
        "P_g.png": "P",  # WhitePawn (was p)
        "Q_g.png": "Q",  # WhiteQueen (was q)
        "R_g.png": "R",  # WhiteRook (was r)
    }
    
    model, class_indices, piece_classes = load_yolo_model()
    accuracy, confusion_matrix = verify_model(image_dir, expected_labels, model, class_indices, threshold)
    print(f"Model verification complete. Accuracy: {accuracy:.2%}")
